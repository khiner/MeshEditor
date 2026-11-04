#include "mesh/Simplify.h"

#include "Profile.h"
#include "gpu/MeshTopologyPushConstants.h"
#include "gpu/Simplify.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "metal/Dispatch.h"
#include "state/Registry.h"

#include <bit>
#include <cmath>

namespace {
constexpr uint32_t InfinityBits = 0x7f800000u;
// The histograms that select a cost's bits from the highest, as the mask of the bits already selected and the lowest bit each one bins.
constexpr std::array<std::pair<uint32_t, uint32_t>, 3> HistogramLevels{{{0u, 19u}, {0xfff80000u, 7u}, {0xffffff80u, 0u}}};

struct Rounds {
    const MeshPipelines &Pipelines;
    const MeshStore &Meshes;
    uint32_t Id;
    mtl::ComputeChain Chain;
    SimplifyJob Job{};
    Range Record;
    uint32_t EdgeCapacity{};
    SimplifyPushConstants Pc{};

    Rounds(state::Registry &r, uint32_t id, const GeometrySelection &selection) : Pipelines(GetMeshPipelines(r)), Meshes(r.Resources.get<const MeshStore>()), Id(id), Chain{Meshes.BufferContext()} {
        const auto &a = Meshes.Arenas();
        const auto &record = Meshes.Get(id);
        const auto vertices = a.Vertices.HandleBound(record.Vertices);
        auto &scratch = Chain.Scratch;
        Record = scratch.Allocate(uint32_t(sizeof(SimplifyJob) / sizeof(uint32_t)));
        Job.Quadrics = scratch.Allocate(SimplifyQuadricWords * vertices).Offset;
        Job.Flags = scratch.Allocate(vertices).Offset;
        Job.Counts = scratch.Allocate(SimplifyCountWords).Offset;
        Job.Histogram = scratch.Allocate(SimplifyHistogramBins).Offset;
        const auto selected = scratch.Allocate((vertices + 31u) / 32u);
        auto bits = scratch.GetMutable(selected);
        std::ranges::fill(bits, 0u);
        for (const auto v : selection.Vertices) bits[v / 32u] |= 1u << (v % 32u);
        Pc.Job = {scratch.Buffer.Slot, Record.Offset};
        Pc.Arenas = TopologyPushConstants(Meshes).Source;
        Pc.CornerAttributes = record.CornerAttributes;
        Pc.Selected = {scratch.Buffer.Slot, selected.Offset};
        EdgeRuns();
        std::ranges::fill(scratch.GetMutable({Job.Counts, SimplifyCountWords}), 0u);
        Upload();
        Dispatch(MeshStore::ElementDomain::Vertex, Pipelines[MeshPass::SimplifyVertexSetup]);
        Dispatch(MeshStore::ElementDomain::Edge, Pipelines[MeshPass::SimplifyEdgeLocks]);
        Dispatch(MeshStore::ElementDomain::Face, Pipelines[MeshPass::SimplifyFaceTriangles]);
        Chain.Submit();
    }
    // The connectivity and layer references follow the record's runs, which each edit may move.
    void EdgeRuns() {
        Pc.Connectivity = Meshes.GetConnectivityRef(Id);
        Pc.Arenas.Layers = Meshes.Layers(Id);
        const auto edges = Meshes.Arenas().EdgeHalfedges.HandleBound(Meshes.Get(Id).EdgeData);
        if (edges <= EdgeCapacity) return;
        EdgeCapacity = edges;
        auto &scratch = Chain.Scratch;
        Job.Errors = scratch.Allocate(edges).Offset;
        Job.Positions = scratch.Allocate(3u * edges).Offset;
        Job.Reductions = scratch.Allocate(edges).Offset;
        Job.Winners = scratch.Allocate(edges).Offset;
    }
    std::span<const uint32_t> Counts() { return Chain.Scratch.Get({Job.Counts, SimplifyCountWords}); }
    void Upload() { Chain.Scratch.Buffer.Update(std::as_bytes(std::span{&Job, 1u}), uint64_t(Record.Offset) * sizeof(uint32_t)); }
    void Dispatch(MeshStore::ElementDomain domain, const mtl::ComputePipeline &pipeline) {
        Pc.Blocks = BlocksOf(Meshes, Id, domain);
        if (Pc.Blocks.Count) Chain.Threads(pipeline, Pc, Pc.Blocks.Count * MeshElementBlockSize);
    }
    // Elects the proposals at most `threshold` greedily in priority order, a pass at a time until a pass elects nothing.
    std::pair<uint32_t, uint32_t> Elect(float threshold) {
        Job.Threshold = threshold;
        Upload();
        std::ranges::fill(Chain.Scratch.GetMutable({Job.Counts, 3u}), 0u);
        Dispatch(MeshStore::ElementDomain::Edge, Pipelines[MeshPass::SimplifyElect]);
        for (Chain.Submit(); Counts()[2]; Chain.Submit()) {
            Chain.Scratch.GetMutable({Job.Counts + 2u, 1u})[0] = 0u;
            Dispatch(MeshStore::ElementDomain::Edge, Pipelines[MeshPass::SimplifyBlock]);
            Dispatch(MeshStore::ElementDomain::Edge, Pipelines[MeshPass::SimplifyElect]);
        }
        const auto counts = Counts();
        return {counts[0], counts[1]};
    }
    // Selects the least floored cost bits at which the reductions of the proposals or winners at most them reach `target`, a digit per histogram from the highest.
    // Returns the bits, or infinity's when the total falls short, with the reductions below the bits and at them.
    std::array<uint32_t, 3> Select(uint32_t target, bool winners) {
        uint32_t bits = 0u, below = 0u, at = 0u;
        Job.HistogramWinners = winners ? 1u : 0u;
        for (const auto [mask, shift] : HistogramLevels) {
            Job.HistogramMask = mask;
            Job.HistogramShift = shift;
            Job.HistogramPrefix = bits;
            Upload();
            std::ranges::fill(Chain.Scratch.GetMutable({Job.Histogram, SimplifyHistogramBins}), 0u);
            Dispatch(MeshStore::ElementDomain::Edge, Pipelines[MeshPass::SimplifyHistogram]);
            Chain.Submit();
            const auto bins = Chain.Scratch.Get({Job.Histogram, SimplifyHistogramBins});
            uint32_t bin = 0u;
            for (; bin < SimplifyHistogramBins && below + bins[bin] < target; ++bin) below += bins[bin];
            if (bin == SimplifyHistogramBins) return {InfinityBits, below, 0u};
            bits |= bin << shift;
            at = bins[bin];
        }
        return {bits, below, at};
    }
    // Proposes the collapses costing below `max_cost` and elects one round's at up to 1.5 times the crossing.
    // The crossing is the least floored cost at which the proposals reduce the triangles by the budget.
    // The 1.5 factor bounds a pass as meshoptimizer does, since each winner blocks the proposals near it.
    std::pair<uint32_t, float> Round(float max_cost, float floor, uint32_t budget, const GeometrySelection &selection, const std::function<void(const MeshTopologyTask &)> &run) {
        EdgeRuns();
        Job.MaxCost = max_cost;
        Job.Floor = floor;
        Job.Threshold = std::numeric_limits<float>::infinity();
        Upload();
        Dispatch(MeshStore::ElementDomain::Edge, Pipelines[MeshPass::SimplifyCandidates]);
        const auto crossing = std::bit_cast<float>(Select(budget, false)[0]);
        auto [collapses, reduced] = Elect(1.5f * crossing);
        if (!collapses) return {0u, floor};
        if (reduced > budget) {
            // The winners rank by their own costs, which the floor would merge.
            Job.Floor = 0.f;
            const auto [bits, below, at] = Select(budget + 1u, true);
            Job.Kept = below ? bits - 1u : bits;
            reduced = below ? below : at;
            Upload();
            Dispatch(MeshStore::ElementDomain::Edge, Pipelines[MeshPass::SimplifyTrim]);
            Chain.Submit();
        }
        run({.SourceId = Id, .Op = MeshTopologyOp::Decimate, .Simplify = Pc.Job, .Selection = selection});
        return {reduced, crossing};
    }
};
} // namespace

bool SimplifyGeometry(state::Registry &r, uint32_t id, const GeometrySelection &selection, float tolerance, float ratio, const std::function<void(const MeshTopologyTask &)> &run) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const Mesh mesh{meshes, id};
    const profile::CpuScope scope{"Simplify"};
    if (!std::isfinite(tolerance) || !std::isfinite(ratio) || (tolerance <= 0.f && ratio >= 1.f) || !mesh.FaceCount()) return false;
    Rounds rounds{r, id, selection};
    GeometrySelection live{.Vertices = selection.Vertices};
    const auto run_live = [&](const MeshTopologyTask &task) {
        run(task);
        const auto &vertices = meshes.Arenas().Vertices;
        const auto entries = vertices.Entries(meshes.Get(id).Vertices);
        std::erase_if(live.Vertices, [&](uint32_t v) { return !vertices.Contains(entries, ElementHandle{v}); });
    };
    bool changed = false;
    if (tolerance > 0.f) {
        while (rounds.Round(tolerance * tolerance, 0.f, UINT32_MAX, live, run_live).first) changed = true;
    }
    auto triangles = rounds.Counts()[3];
    const auto target = uint32_t(std::ceil(double(triangles) * std::max(0.f, ratio)));
    float threshold = 0.f;
    while (ratio < 1.f && triangles > target) {
        const auto [reduced, reached] = rounds.Round(std::numeric_limits<float>::infinity(), threshold, triangles - target, live, run_live);
        if (!reduced) break;
        threshold = reached;
        triangles -= std::min(reduced, triangles);
        changed = true;
    }
    return changed;
}
