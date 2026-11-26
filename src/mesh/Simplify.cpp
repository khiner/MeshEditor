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
#include <cstddef>

namespace {
// The Block and Elect pass pairs an election records before the host reads whether it converged.
// The pairs after an Elect pass that elected nothing exit at once.
constexpr uint32_t ElectionPairs{16u};

// One record's simplify: its job in the chain's scratch with the job's host fields, and the state of its rounds.
struct Simplification {
    state::Registry &R;
    mtl::ComputeChain &Chain;
    uint32_t Id;
    float Tolerance, Ratio;
    std::function<void(const MeshTopologyTask &, std::function<void()>)> Decimate;
    std::function<void(bool)> Done;
    // The selected vertices that survive the rounds.
    GeometrySelection Live;
    SlotOffset At{};
    SimplifyJob Job{};
    uint32_t EdgeCapacity{};
    // The rounds raise the cost toward the target, after the tolerance's rounds.
    bool Reducing{};
    bool Changed{};
    // The selection's triangles the rounds leave, and the count the ratio targets.
    uint32_t Triangles{}, Target{};
    // The number of the election's last recorded pass.
    uint32_t Pass{};
};
using Run = std::shared_ptr<Simplification>;

void Upload(Simplification &s) { s.Chain.Scratch.Buffer.Update(std::as_bytes(std::span{&s.Job, 1u}).first(offsetof(SimplifyJob, Threshold)), uint64_t(s.At.Offset) * sizeof(uint32_t)); }
// The job word the passes wrote at `offset` bytes.
uint32_t Written(const Simplification &s, size_t offset) { return s.Chain.Scratch.Get({s.At.Offset + uint32_t(offset / sizeof(uint32_t)), 1u})[0]; }

// The record's blocks, references and edge runs, which each edit may move or grow.
void Renew(Simplification &s) {
    using Domain = MeshStore::ElementDomain;
    const auto &meshes = s.R.Resources.get<const MeshStore>();
    s.Job.Vertices = BlocksOf(meshes, s.Id, Domain::Vertex);
    s.Job.Edges = BlocksOf(meshes, s.Id, Domain::Edge);
    s.Job.Faces = BlocksOf(meshes, s.Id, Domain::Face);
    s.Job.Connectivity = meshes.GetConnectivityRef(s.Id);
    s.Job.Layers = meshes.Layers(s.Id);
    const auto edges = meshes.Arenas().EdgeHalfedges.HandleBound(meshes.Get(s.Id).EdgeData);
    if (edges <= s.EdgeCapacity) return;
    s.EdgeCapacity = edges;
    auto &scratch = s.Chain.Scratch;
    s.Job.Errors = scratch.Allocate(edges).Offset;
    s.Job.Positions = scratch.Allocate(3u * edges).Offset;
    s.Job.Reductions = scratch.Allocate(edges).Offset;
    s.Job.Winners = scratch.Allocate(edges).Offset;
}

// Records the pass over `tiles` threadgroups of the job with `parameter`, with the jobs the other pending lanes gather for it at the same step.
void Record(Simplification &s, MeshPass pass, uint32_t tiles, uint32_t parameter) {
    const SimplifyPass job{s.At, parameter};
    mtl::GatherPass(s.Chain, GetMeshPipelines(s.R)[pass], std::span{&job, 1u}, std::array{tiles}, JobTileThreads);
}

void Read(const Run &);

// Records the election's next pass pairs, then the trim of its winners to the budget once it converged, and reads it after the chain completes them.
void Elect(const Run &s) {
    s->Chain.Lane(s->Id, [&] {
        const auto edges = s->Job.Edges.Count;
        for (uint32_t pair = 0u; pair < ElectionPairs; ++pair) {
            ++s->Pass;
            Record(*s, MeshPass::SimplifyBlock, edges, s->Pass);
            Record(*s, MeshPass::SimplifyElect, edges, s->Pass);
        }
        const uint32_t trim = SimplifySelectWinners | ((s->Pass & 1u) ? SimplifyLastParity : 0u);
        for (uint32_t level = 0u; level < 3u; ++level) {
            Record(*s, MeshPass::SimplifyHistogram, edges, trim | level);
            Record(*s, MeshPass::SimplifySelect, 1u, trim | level);
        }
        Record(*s, MeshPass::SimplifyTrim, edges, trim);
    });
    s->Chain.AfterSubmit([s] { Read(s); });
}

// Records a round over the proposals costing below `max_cost`, electing them at up to 1.5 times the crossing, the least floored cost at which their reductions reach the budget.
// The 1.5 factor bounds a pass as meshoptimizer does, since each winner blocks the proposals near it.
void Round(const Run &s, float max_cost, float floor, uint32_t budget) {
    Renew(*s);
    s->Job.MaxCost = max_cost;
    s->Job.Budget = budget;
    s->Job.Floor = floor;
    Upload(*s);
    s->Pass = 0u;
    s->Chain.Lane(s->Id, [&] {
        const auto edges = s->Job.Edges.Count;
        Record(*s, MeshPass::SimplifyCandidates, edges, 0u);
        for (uint32_t level = 0u; level < 3u; ++level) {
            Record(*s, MeshPass::SimplifyHistogram, edges, level);
            Record(*s, MeshPass::SimplifySelect, 1u, level);
        }
        Record(*s, MeshPass::SimplifyElect, edges, 0u);
    });
    Elect(s);
}

// Starts the rounds that raise the cost from `floor` until the selection's triangles fall to the target, or ends the simplify.
void Reduce(const Run &s, float floor) {
    s->Reducing = true;
    if (s->Ratio < 1.f && s->Triangles > s->Target) Round(s, std::numeric_limits<float>::infinity(), floor, s->Triangles - s->Target);
    else s->Done(s->Changed);
}

// Continues after a round's Decimate removed `reduced` triangles.
void Decimated(const Run &s, uint32_t reduced, float crossing) {
    const auto &meshes = s->R.Resources.get<const MeshStore>();
    const auto &vertices = meshes.Arenas().Vertices;
    const auto entries = vertices.Entries(meshes.Get(s->Id).Vertices);
    std::erase_if(s->Live.Vertices, [&](uint32_t v) { return !vertices.Contains(entries, ElementHandle{v}); });
    s->Triangles -= std::min(reduced, s->Triangles);
    if (!reduced) {
        if (s->Reducing) s->Done(s->Changed);
        else Reduce(s, 0.f);
        return;
    }
    s->Changed = true;
    if (s->Reducing) Reduce(s, crossing);
    else Round(s, s->Tolerance * s->Tolerance, 0.f, UINT32_MAX);
}

// Continues the election when it has not converged, and otherwise decimates the round's winners, or ends the round's phase when there are none.
void Read(const Run &s) {
    const auto counts = s->Chain.Scratch.Get({s->Job.Counts, SimplifyCountWords});
    if (counts[SimplifyElected + (s->Pass & 1u)]) return Elect(s);
    if (!counts[0]) {
        if (s->Reducing) s->Done(s->Changed);
        else Reduce(s, 0.f);
        return;
    }
    const uint32_t reduced = counts[1] > s->Job.Budget ? Written(*s, offsetof(SimplifyJob, Reduced)) : counts[1];
    const float crossing = std::bit_cast<float>(Written(*s, offsetof(SimplifyJob, Crossing)));
    s->Decimate({.SourceId = s->Id, .Op = MeshTopologyOp::Decimate, .Simplify = s->At, .Selection = s->Live}, [s, reduced, crossing] { Decimated(s, reduced, crossing); });
}
} // namespace

bool RecordSimplify(state::Registry &r, mtl::ComputeChain &chain, uint32_t id, const GeometrySelection &selection, float tolerance, float ratio, std::function<void(const MeshTopologyTask &, std::function<void()>)> decimate, std::function<void(bool)> done) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    if (!std::isfinite(tolerance) || !std::isfinite(ratio) || (tolerance <= 0.f && ratio >= 1.f) || !Mesh{meshes, id}.FaceCount()) return false;
    const auto s = std::make_shared<Simplification>(Simplification{.R = r, .Chain = chain, .Id = id, .Tolerance = tolerance, .Ratio = ratio, .Decimate = std::move(decimate), .Done = std::move(done), .Live = {.Vertices = selection.Vertices}});
    const auto &record = meshes.Get(id);
    const auto vertices = meshes.Arenas().Vertices.HandleBound(record.Vertices);
    auto &scratch = chain.Scratch;
    const auto zeroed = [&](uint32_t words) {
        const auto range = scratch.Allocate(words);
        std::ranges::fill(scratch.GetMutable(range), 0u);
        return range;
    };
    s->At = {scratch.Buffer.Slot, zeroed(uint32_t(sizeof(SimplifyJob) / sizeof(uint32_t))).Offset};
    s->Job.Quadrics = scratch.Allocate(SimplifyQuadricWords * vertices).Offset;
    s->Job.Flags = scratch.Allocate(vertices).Offset;
    s->Job.Counts = zeroed(SimplifyCountWords).Offset;
    s->Job.Histogram = zeroed(SimplifyHistogramBins).Offset;
    const auto selected = zeroed((vertices + 31u) / 32u);
    auto bits = scratch.GetMutable(selected);
    for (const auto v : selection.Vertices) bits[v / 32u] |= 1u << (v % 32u);
    s->Job.Arenas = TopologyPushConstants(meshes).Source;
    s->Job.CornerAttributes = record.CornerAttributes;
    s->Job.Selected = {scratch.Buffer.Slot, selected.Offset};
    Renew(*s);
    Upload(*s);
    chain.Lane(id, [&] {
        Record(*s, MeshPass::SimplifyVertexSetup, s->Job.Vertices.Count, 0u);
        Record(*s, MeshPass::SimplifyEdgeLocks, s->Job.Edges.Count, 0u);
        Record(*s, MeshPass::SimplifyFaceTriangles, s->Job.Faces.Count, 0u);
    });
    chain.AfterSubmit([s] {
        s->Triangles = s->Chain.Scratch.Get({s->Job.Counts + 2u, 1u})[0];
        s->Target = uint32_t(std::ceil(double(s->Triangles) * std::max(0.f, s->Ratio)));
        if (s->Tolerance > 0.f) Round(s, s->Tolerance * s->Tolerance, 0.f, UINT32_MAX);
        else Reduce(s, 0.f);
    });
    return true;
}

bool SimplifyGeometry(state::Registry &r, uint32_t id, const GeometrySelection &selection, float tolerance, float ratio, const std::function<void(const MeshTopologyTask &)> &run) {
    const profile::CpuScope scope{"Simplify"};
    mtl::ComputeChain chain{r.Resources.get<const MeshStore>().BufferContext()};
    bool changed = false;
    const auto decimate = [&run](const MeshTopologyTask &task, std::function<void()> next) {
        run(task);
        next();
    };
    if (!RecordSimplify(r, chain, id, selection, tolerance, ratio, decimate, [&changed](bool simplified) { changed = simplified; })) return false;
    chain.Drain();
    return changed;
}
