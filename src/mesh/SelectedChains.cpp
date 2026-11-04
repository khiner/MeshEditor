#include "mesh/SelectedChains.h"

#include "mesh/MeshStore.h"
#include "mesh/ScratchPasses.h"

namespace {
struct Chains {
    mtl::ComputeChain Chain;
    SelectedChainsPushConstants Pc{};

    Chains(const MeshStore &meshes, uint32_t id) : Chain{meshes.BufferContext()} {
        const auto &a = meshes.Arenas();
        const auto &record = meshes.Get(id);
        Pc.Connectivity = meshes.GetConnectivityRef(id);
        Pc.VertexSlot = a.Vertices.Buffer.Slot;
        Pc.EdgeBound = a.EdgeHalfedges.HandleBound(record.EdgeData);
        Pc.ScratchSlot = Chain.Scratch.Buffer.Slot;
    }
    void Facts(const MeshPipelines &pipelines, std::span<const uint32_t> edges, std::span<const uint32_t> faces) {
        Pc.Edges = Chain.Scratch.Allocate(edges).Offset;
        Pc.EdgeCount = uint32_t(edges.size());
        Pc.Counts = TakeScratch(Chain, Pc.EdgeBound).Offset;
        Pc.EdgeInfo = TakeScratch(Chain, sizeof(EdgeFacts) / sizeof(uint32_t) * edges.size()).Offset;
        FillScratch(Chain, pipelines, Pc.Counts, Pc.EdgeBound, 0u);
        if (!faces.empty()) {
            Pc.Faces = Chain.Scratch.Allocate(faces).Offset;
            Pc.ListedFaces = uint32_t(faces.size());
            Chain.Threads(pipelines[MeshPass::SelectedChainFaceEdges], Pc, Pc.ListedFaces);
        }
        Chain.Threads(pipelines[MeshPass::SelectedChainEdgeFacts], Pc, Pc.EdgeCount);
    }
    std::vector<EdgeFacts> ReadFacts() const { return ReadScratch<EdgeFacts>(Chain, Pc.EdgeInfo, Pc.EdgeCount); }
    void Upload(std::span<const CornerLoop> loops) {
        std::vector<uint32_t> records, pairs;
        for (const auto &loop : loops) {
            records.insert(records.end(), {uint32_t(pairs.size() / 2u), uint32_t(loop.size()), 0u});
            for (const auto &[vertex, source] : loop) pairs.insert(pairs.end(), {vertex, source});
        }
        Pc.Records = Chain.Scratch.Allocate(records).Offset;
        Pc.Output = Chain.Scratch.Allocate(pairs).Offset;
        Pc.ChainCount = uint32_t(loops.size());
    }
};
} // namespace

std::vector<uint32_t> VerticesOf(const CornerLoop &loop) {
    std::vector<uint32_t> vertices;
    for (const auto &[vertex, source] : loop) vertices.push_back(vertex);
    return vertices;
}

SelectedChains SelectedEdgeChains(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> edges, bool boundary_only) {
    SelectedChains result;
    if (edges.empty()) return result;
    Chains chains{meshes, id};
    auto &pc = chains.Pc;
    auto &chain = chains.Chain;
    auto &scratch = chain.Scratch;
    pc.Blocks = BlocksOf(meshes, id, MeshStore::ElementDomain::Edge);
    if (!pc.Blocks.Count) return result;
    const uint32_t oriented = 2u * pc.EdgeBound, vertices = meshes.Arenas().Vertices.HandleBound(meshes.Get(id).Vertices);
    pc.BoundaryOnly = boundary_only ? 1u : 0u;
    chains.Facts(pipelines, edges, {});
    pc.Degree = TakeScratch(chain, vertices).Offset;
    pc.Lowest = TakeScratch(chain, vertices).Offset;
    pc.Highest = TakeScratch(chain, vertices).Offset;
    pc.Junctions = TakeScratch(chain, 1u).Offset;
    pc.Candidate = TakeScratch(chain, oriented).Offset;
    pc.Next = TakeScratch(chain, oriented).Offset;
    pc.Previous = TakeScratch(chain, oriented).Offset;
    for (auto *run : {&pc.Jumps, &pc.PreviousJumps, &pc.KeyIndex, &pc.Tails, &pc.Heads, &pc.Distances, &pc.Successors}) *run = TakeScratch(chain, 2ull * oriented).Offset;
    for (auto *run : {&pc.Key, &pc.Reverse}) *run = TakeScratch(chain, 4ull * oriented).Offset;
    pc.Closed = TakeScratch(chain, oriented).Offset;
    pc.Lengths = TakeScratch(chain, uint64_t(oriented) + 1u).Offset;
    pc.Flags = TakeScratch(chain, uint64_t(oriented) + 1u).Offset;
    pc.Output = TakeScratch(chain, 2ull * (uint64_t(oriented) + 2ull * edges.size())).Offset;
    const uint32_t threads = pc.Blocks.Count * MeshElementBlockSize;
    FillScratch(chain, pipelines, pc.Degree, vertices, 0u);
    FillScratch(chain, pipelines, pc.Lowest, vertices, InvalidOffset);
    FillScratch(chain, pipelines, pc.Highest, vertices, 0u);
    FillScratch(chain, pipelines, pc.Junctions, 1u, 0u);
    FillScratch(chain, pipelines, pc.Candidate, oriented, 0u);
    FillScratch(chain, pipelines, pc.Previous, oriented, InvalidOffset);
    chain.Threads(pipelines[MeshPass::SelectedChainDegrees], pc, threads);
    chain.Threads(pipelines[MeshPass::SelectedChainLink], pc, threads);
    chain.Threads(pipelines[MeshPass::SelectedChainHeads], pc, threads);
    auto round_pc = RecordLoopRanks(chain, pipelines, pc, {MeshPass::SelectedChainLabel, MeshPass::SelectedChainRankInit, MeshPass::SelectedChainRank, MeshPass::SelectedChainCount}, oriented, threads, pc.Lengths, pc.Flags);
    chain.Submit();
    const uint32_t count = scratch.Get({pc.Flags + oriented, 1u})[0];
    result.Junction = scratch.Get({pc.Junctions, 1u})[0] != 0u;
    result.Edges = chains.ReadFacts();
    if (!count) return result;
    round_pc.Records = TakeScratch(chain, 3ull * count).Offset;
    chain.Threads(pipelines[MeshPass::SelectedChainOutput], round_pc, threads);
    chain.Submit();
    const auto records = scratch.Get({round_pc.Records, 3u * count});
    result.Chains.resize(count);
    for (uint32_t c = 0u; c < count; ++c) {
        const auto length = records[3u * c + 1u];
        auto &out = result.Chains[c];
        out.Closed = records[3u * c + 2u] != 0u;
        out.Pairs = ReadScratch<std::array<uint32_t, 2>>(chain, pc.Output + 2u * records[3u * c], length);
        if (!out.Closed && length > 1u && out.Pairs.back()[0] == out.Pairs.front()[0]) out.Pairs.pop_back();
    }
    std::ranges::sort(result.Chains, [](const SelectedChain &a, const SelectedChain &b) {
        const auto key = [](const SelectedChain &c) { return std::tuple{c.Closed, c.Pairs[0][0], c.Pairs.size() > 1u ? c.Pairs[1][0] : 0u}; };
        return key(a) < key(b);
    });
    return result;
}

std::vector<EdgeFacts> SelectedEdgeFacts(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> edges, std::span<const uint32_t> faces) {
    if (edges.empty()) return {};
    Chains chains{meshes, id};
    chains.Facts(pipelines, edges, faces);
    chains.Chain.Submit();
    return chains.ReadFacts();
}

std::vector<ChainSummary> SummarizeLoops(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const CornerLoop> loops) {
    if (loops.empty()) return {};
    Chains chains{meshes, id};
    chains.Upload(loops);
    constexpr uint32_t Words = sizeof(ChainSummary) / sizeof(uint32_t);
    chains.Pc.Summaries = TakeScratch(chains.Chain, uint64_t(Words) * loops.size()).Offset;
    chains.Chain.Threads(pipelines[MeshPass::SelectedChainSummaries], chains.Pc, uint32_t(loops.size()));
    chains.Chain.Submit();
    return ReadScratch<ChainSummary>(chains.Chain, chains.Pc.Summaries, loops.size());
}

uint32_t NearestInLoop(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, const CornerLoop &loop, vec3 point) {
    Chains chains{meshes, id};
    chains.Upload(std::span{&loop, 1u});
    chains.Pc.Point = point;
    chains.Pc.Nearest = TakeScratch(chains.Chain, 1u).Offset;
    chains.Chain.Threads(pipelines[MeshPass::SelectedChainNearest], chains.Pc, 1u);
    chains.Chain.Submit();
    return chains.Chain.Scratch.Get({chains.Pc.Nearest, 1u})[0];
}

std::optional<std::vector<float>> PointCloudAngles(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> vertices) {
    if (vertices.empty()) return {};
    Chains chains{meshes, id};
    chains.Pc.Points = chains.Chain.Scratch.Allocate(vertices).Offset;
    chains.Pc.PointCount = uint32_t(vertices.size());
    chains.Pc.Angles = TakeScratch(chains.Chain, uint64_t(vertices.size()) + 1u).Offset;
    chains.Chain.Groups(pipelines[MeshPass::SelectedChainPointAngles], chains.Pc, 1u);
    chains.Chain.Submit();
    if (!chains.Chain.Scratch.Get({chains.Pc.Angles + uint32_t(vertices.size()), 1u})[0]) return {};
    return ReadScratch<float>(chains.Chain, chains.Pc.Angles, vertices.size());
}
