#pragma once
#include "gpu/SelectedChainsJob.h"

#include <array>
#include <functional>
#include <optional>
#include <span>
#include <vector>

struct MeshStore;
struct MeshPipelines;
namespace mtl {
struct ComputeChain;
}

// A loop of vertices, each paired with the halfedge supplying the corner before it, or InvalidOffset for a new edge.
using CornerLoop = std::vector<std::array<uint32_t, 2>>;
std::vector<uint32_t> VerticesOf(const CornerLoop &);

struct SelectedChain {
    CornerLoop Pairs;
    bool Closed;
};
// The chains of the sorted selected edges in Blender's order.
// Open chains come first, each from its lower end and ordered by that end, then closed ones from their lowest vertex toward its lower neighbor.
// A chain back to a vertex with more than two selected edges is open and omits the repeated vertex, and Junction records that such a vertex exists.
struct SelectedChains {
    std::vector<SelectedChain> Chains;
    std::vector<EdgeFacts> Edges;
    bool Junction{};
};
// Records the passes of the selected edges' chains in lanes of the chain, which gather with the other lanes' passes, and calls `then` with the chains after the submits that complete them.
// With `boundary_only`, only edges with at most one face side join the chains, and the others supply no source.
void RecordSelectedEdgeChains(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t id, std::span<const uint32_t> edges, bool boundary_only, std::function<void(SelectedChains)> then);
// The facts of the sorted edges, counting only the listed faces.
std::vector<EdgeFacts> SelectedEdgeFacts(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> edges, std::span<const uint32_t> faces);
// Records the passes of SelectedEdgeFacts in the record's lane of the chain and calls `then` with the facts after the submit that completes them.
void RecordSelectedEdgeFacts(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t id, std::span<const uint32_t> edges, std::span<const uint32_t> faces, std::function<void(std::vector<EdgeFacts>)> then);
std::vector<ChainSummary> SummarizeLoops(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const CornerLoop> loops);
// Records the pass of SummarizeLoops in the record's lane of the chain and calls `then` with the summaries after the submit that completes it.
void RecordSummarizeLoops(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t id, std::span<const CornerLoop> loops, std::function<void(std::vector<ChainSummary>)> then);
// The index of the loop's vertex nearest the point, the lowest at a tie.
// Records the pass in the record's lane of the chain and calls `then` with the index after the submit that completes it.
void RecordNearestInLoop(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t id, const CornerLoop &loop, vec3 point, std::function<void(uint32_t)> then);
// Each vertex's angle about the cloud's center in the plane of its farthest point and the farthest point across that, or none when the cloud is a point or a line.
std::optional<std::vector<float>> PointCloudAngles(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> vertices);
