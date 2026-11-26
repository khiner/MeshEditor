#include "mesh/SelectedChains.h"

#include "mesh/MeshStore.h"
#include "mesh/ScratchPasses.h"

namespace {
SelectedChainsJob ChainsJob(const MeshStore &meshes, uint32_t id) {
    return {
        .Connectivity = meshes.GetConnectivityRef(id),
        .VertexSlot = meshes.Arenas().Vertices.Buffer.Slot,
        .EdgeBound = meshes.Arenas().EdgeHalfedges.HandleBound(meshes.Get(id).EdgeData),
    };
}
// Takes the runs of the listed edges' facts, and returns the fill that zeroes their listed face counts.
ScratchFillRun TakeFacts(mtl::ComputeChain &chain, SelectedChainsJob &job, std::span<const uint32_t> edges) {
    job.Edges = chain.Scratch.Allocate(edges).Offset;
    job.EdgeCount = uint32_t(edges.size());
    job.Counts = TakeScratch(chain, job.EdgeBound).Offset;
    job.EdgeInfo = TakeScratch(chain, sizeof(EdgeFacts) / sizeof(uint32_t) * edges.size()).Offset;
    return {job.Counts, job.EdgeBound, 0u};
}
void Upload(mtl::ComputeChain &chain, SelectedChainsJob &job, std::span<const CornerLoop> loops) {
    std::vector<uint32_t> records, pairs;
    for (const auto &loop : loops) {
        records.insert(records.end(), {uint32_t(pairs.size() / 2u), uint32_t(loop.size()), 0u});
        for (const auto &[vertex, source] : loop) pairs.insert(pairs.end(), {vertex, source});
    }
    job.Records = chain.Scratch.Allocate(records).Offset;
    job.Output = chain.Scratch.Allocate(pairs).Offset;
    job.ChainCount = uint32_t(loops.size());
}
} // namespace

std::vector<uint32_t> VerticesOf(const CornerLoop &loop) {
    std::vector<uint32_t> vertices;
    for (const auto &[vertex, source] : loop) vertices.push_back(vertex);
    return vertices;
}

void RecordSelectedEdgeChains(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t id, std::span<const uint32_t> edges, bool boundary_only, std::function<void(SelectedChains)> then) {
    if (edges.empty()) return then({});
    auto job = ChainsJob(meshes, id);
    job.Blocks = BlocksOf(meshes, id, MeshStore::ElementDomain::Edge);
    if (!job.Blocks.Count) return then({});
    const uint32_t oriented = 2u * job.EdgeBound, vertices = meshes.Arenas().Vertices.HandleBound(meshes.Get(id).Vertices);
    job.BoundaryOnly = boundary_only ? 1u : 0u;
    const auto counts = TakeFacts(chain, job, edges);
    job.Degree = TakeScratch(chain, vertices).Offset;
    job.Lowest = TakeScratch(chain, vertices).Offset;
    job.Highest = TakeScratch(chain, vertices).Offset;
    job.Junctions = TakeScratch(chain, 1u).Offset;
    job.Candidate = TakeScratch(chain, oriented).Offset;
    job.Next = TakeScratch(chain, oriented).Offset;
    job.Previous = TakeScratch(chain, oriented).Offset;
    for (auto *run : {&job.Jumps, &job.PreviousJumps, &job.KeyIndex, &job.Tails, &job.Heads, &job.Distances, &job.Successors}) *run = TakeScratch(chain, 2ull * oriented).Offset;
    for (auto *run : {&job.Key, &job.Reverse}) *run = TakeScratch(chain, 4ull * oriented).Offset;
    job.Closed = TakeScratch(chain, oriented).Offset;
    job.Lengths = TakeScratch(chain, uint64_t(oriented) + 1u).Offset;
    job.Flags = TakeScratch(chain, uint64_t(oriented) + 1u).Offset;
    job.Parity = TakeScratch(chain, 1u).Offset;
    job.Output = TakeScratch(chain, 2ull * (uint64_t(oriented) + 2ull * edges.size())).Offset;
    const uint32_t threads = job.Blocks.Count * MeshElementBlockSize;
    chain.Lane([&] {
        FillScratch(chain, pipelines, std::to_array<ScratchFillRun>({counts, {job.Degree, vertices, 0u}, {job.Lowest, vertices, InvalidOffset}, {job.Highest, vertices, 0u}, {job.Junctions, 1u, 0u}, {job.Candidate, oriented, 0u}, {job.Previous, oriented, InvalidOffset}}));
        mtl::GatherPass(chain, pipelines[MeshPass::SelectedChainEdgeFacts], job, job.EdgeCount);
        mtl::GatherPass(chain, pipelines[MeshPass::SelectedChainDegrees], job, threads);
        mtl::GatherPass(chain, pipelines[MeshPass::SelectedChainLink], job, threads);
        mtl::GatherPass(chain, pipelines[MeshPass::SelectedChainHeads], job, threads);
        GatherLoopRanks(chain, pipelines, {MeshPass::SelectedChainLabel, MeshPass::SelectedChainRankInit, MeshPass::SelectedChainRank, MeshPass::SelectedChainCount}, job, job.Blocks.Count, [](const SelectedChainsJob &gathered) { return LoopList{2u * gathered.EdgeBound, gathered.Lengths, gathered.Flags}; });
    });
    chain.AfterSubmit([&pipelines, &chain, job, oriented, threads, then = std::move(then)] mutable {
        const auto &scratch = chain.Scratch;
        const uint32_t count = scratch.Get({job.Flags + oriented, 1u})[0];
        SelectedChains result{.Edges = ReadScratch<EdgeFacts>(chain, job.EdgeInfo, job.EdgeCount), .Junction = scratch.Get({job.Junctions, 1u})[0] != 0u};
        if (!count) return then(std::move(result));
        job.Records = TakeScratch(chain, 3ull * count).Offset;
        chain.Lane([&] { mtl::GatherPass(chain, pipelines[MeshPass::SelectedChainOutput], job, threads); });
        chain.AfterSubmit([&chain, job, count, result = std::move(result), then = std::move(then)] mutable {
            const auto records = chain.Scratch.Get({job.Records, 3u * count});
            result.Chains.resize(count);
            for (uint32_t c = 0u; c < count; ++c) {
                const auto length = records[3u * c + 1u];
                auto &out = result.Chains[c];
                out.Closed = records[3u * c + 2u] != 0u;
                out.Pairs = ReadScratch<std::array<uint32_t, 2>>(chain, job.Output + 2u * records[3u * c], length);
                if (!out.Closed && length > 1u && out.Pairs.back()[0] == out.Pairs.front()[0]) out.Pairs.pop_back();
            }
            std::ranges::sort(result.Chains, [](const SelectedChain &a, const SelectedChain &b) {
                const auto key = [](const SelectedChain &c) { return std::tuple{c.Closed, c.Pairs[0][0], c.Pairs.size() > 1u ? c.Pairs[1][0] : 0u}; };
                return key(a) < key(b);
            });
            then(std::move(result));
        });
    });
}

void RecordSelectedEdgeFacts(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t id, std::span<const uint32_t> edges, std::span<const uint32_t> faces, std::function<void(std::vector<EdgeFacts>)> then) {
    if (edges.empty()) return then({});
    auto job = ChainsJob(meshes, id);
    const auto facts = TakeFacts(chain, job, edges);
    if (!faces.empty()) {
        job.Faces = chain.Scratch.Allocate(faces).Offset;
        job.ListedFaces = uint32_t(faces.size());
    }
    chain.Lane(id, [&] {
        FillScratch(chain, pipelines, std::array{facts});
        if (!faces.empty()) mtl::GatherPass(chain, pipelines[MeshPass::SelectedChainFaceEdges], job, job.ListedFaces);
        mtl::GatherPass(chain, pipelines[MeshPass::SelectedChainEdgeFacts], job, job.EdgeCount);
    });
    chain.AfterSubmit([&chain, info = job.EdgeInfo, count = job.EdgeCount, then = std::move(then)] { then(ReadScratch<EdgeFacts>(chain, info, count)); });
}
std::vector<EdgeFacts> SelectedEdgeFacts(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> edges, std::span<const uint32_t> faces) {
    return mtl::Drained<std::vector<EdgeFacts>>(meshes.BufferContext(), [&](mtl::ComputeChain &chain, auto then) { RecordSelectedEdgeFacts(meshes, pipelines, chain, id, edges, faces, std::move(then)); });
}

void RecordSummarizeLoops(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t id, std::span<const CornerLoop> loops, std::function<void(std::vector<ChainSummary>)> then) {
    if (loops.empty()) return then({});
    auto job = ChainsJob(meshes, id);
    Upload(chain, job, loops);
    constexpr uint32_t Words = sizeof(ChainSummary) / sizeof(uint32_t);
    job.Summaries = TakeScratch(chain, uint64_t(Words) * loops.size()).Offset;
    chain.Lane(id, [&] { mtl::GatherPass(chain, pipelines[MeshPass::SelectedChainSummaries], job, job.ChainCount); });
    chain.AfterSubmit([&chain, summaries = job.Summaries, count = loops.size(), then = std::move(then)] { then(ReadScratch<ChainSummary>(chain, summaries, count)); });
}
std::vector<ChainSummary> SummarizeLoops(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const CornerLoop> loops) {
    return mtl::Drained<std::vector<ChainSummary>>(meshes.BufferContext(), [&](mtl::ComputeChain &chain, auto then) { RecordSummarizeLoops(meshes, pipelines, chain, id, loops, std::move(then)); });
}

void RecordNearestInLoop(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t id, const CornerLoop &loop, vec3 point, std::function<void(uint32_t)> then) {
    auto job = ChainsJob(meshes, id);
    Upload(chain, job, std::span{&loop, 1u});
    job.Point = point;
    job.Nearest = TakeScratch(chain, 1u).Offset;
    chain.Lane(id, [&] { mtl::GatherPass(chain, pipelines[MeshPass::SelectedChainNearest], job, 1u); });
    chain.AfterSubmit([&chain, nearest = job.Nearest, then = std::move(then)] { then(chain.Scratch.Get({nearest, 1u})[0]); });
}

std::optional<std::vector<float>> PointCloudAngles(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> vertices) {
    if (vertices.empty()) return {};
    mtl::ComputeChain chain{meshes.BufferContext()};
    auto job = ChainsJob(meshes, id);
    job.Points = chain.Scratch.Allocate(vertices).Offset;
    job.PointCount = uint32_t(vertices.size());
    job.Angles = TakeScratch(chain, uint64_t(vertices.size()) + 1u).Offset;
    mtl::GatherPass(chain, pipelines[MeshPass::SelectedChainPointAngles], job, 1u);
    chain.Submit();
    if (!chain.Scratch.Get({job.Angles + uint32_t(vertices.size()), 1u})[0]) return {};
    return ReadScratch<float>(chain, job.Angles, vertices.size());
}
