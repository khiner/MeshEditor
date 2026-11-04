#pragma once
#include "gpu/SelectedChainsPushConstants.h"

#include <array>
#include <optional>
#include <span>
#include <vector>

struct MeshStore;
struct MeshPipelines;

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
// With `boundary_only`, only edges with at most one face side join the chains, and the others supply no source.
SelectedChains SelectedEdgeChains(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> edges, bool boundary_only);
// The facts of the sorted edges, counting only the listed faces.
std::vector<EdgeFacts> SelectedEdgeFacts(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> edges, std::span<const uint32_t> faces);
std::vector<ChainSummary> SummarizeLoops(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const CornerLoop> loops);
// The index of the loop's vertex nearest the point, the lowest at a tie.
uint32_t NearestInLoop(const MeshStore &, const MeshPipelines &, uint32_t id, const CornerLoop &loop, vec3 point);
// Each vertex's angle about the cloud's center in the plane of its farthest point and the farthest point across that, or none when the cloud is a point or a line.
std::optional<std::vector<float>> PointCloudAngles(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> vertices);
