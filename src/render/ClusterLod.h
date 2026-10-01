#pragma once

#include "Range.h"
#include "gpu/LodNode.h"
#include "gpu/MeshletLimit.h"
#include "mesh/ElementAttributeView.h"
#include "mesh/CornerNormalView.h"
#include "numeric/vec3.h"
#include "render/CornerWeldKey.h"

#include <cstdint>
#include <span>
#include <vector>

// The visibility ID encoding and mesh-shader output contract fix these cluster limits.
inline constexpr uint32_t ClusterLodMaxVertices{uint32_t(MeshletLimit::MaxVertices)};
inline constexpr uint32_t ClusterLodMaxTriangles{uint32_t(MeshletLimit::MaxTriangles)};
// Merge clusters below this triangle count into a neighbor.
inline constexpr uint32_t ClusterLodMinTriangles{16};
// Maximum clusters per DAG group.
// A mesh at or below this count has no coarser level.
inline constexpr uint32_t ClusterLodPartitionSize{16};
// A face mesh with more level-zero clusters than one partition carries a coarser level.
inline constexpr bool ClusterLodApplies(bool face_topology, uint32_t level0_count) { return face_topology && level0_count > ClusterLodPartitionSize; }
// Render records one span-tree leaf covers.
inline constexpr uint32_t ClusterLodSpanLeafRecords{64};
// Children one span-tree node covers.
// A wider tree reduces depth while increasing the granularity of pruned record runs.
inline constexpr uint32_t ClusterLodSpanNodeWidth{64};
inline constexpr uint32_t ClusterLodInvalid{~0u};

struct meshopt_Bounds;
// Pack a meshopt cone axis and cutoff into the record's ConeAxisCutoff field.
uint32_t PackCone(const meshopt_Bounds &, bool cone_cull_safe);

struct ClusterLodPrimitive {
    uint32_t FirstTriangle{}, TriangleCount{};
    uint32_t FirstCluster{}, ClusterCount{};
    uint32_t Attributes{~0u};
};

// Describes one input cluster with the bounding sphere and error its group merges.
// A coarse cluster carries the sphere and error of the group it was simplified from.
struct ClusterLodSourceCluster {
    uint32_t FirstVertex{}, VertexCount{};
    uint32_t FirstLocalTriangle{}, TriangleCount{};
    vec3 Center{};
    float Radius{};
    float Error{};
    bool ConeCullSafe{};
};

// Provides mesh geometry to the DAG build.
// Every span is borrowed for the duration of the call.
struct ClusterLodMesh {
    TriangleVertexView CornerVertices;
    // Vertex positions, float3 in the first twelve bytes of each vertex.
    const float *Positions{};
    size_t PositionStride{};
    uint32_t VertexFirst{}; // Arena handle of the first position in the borrowed span.
    Range DenseVertices{}; // Existing dense owner span, and sparse owners derive primitive bounds.
    CornerNormalView Normals;

    CornerWeldSource Weld;

    std::span<const ClusterLodPrimitive> Primitives;
    std::span<const ClusterLodSourceCluster> Clusters;
    // Canonical corner handles (absolute index slots for external triangle soups).
    // Only local triangle bytes carry render flags.
    std::span<const uint32_t> SourceVertexCorners;
    std::span<const uint8_t> SourceLocalTriangles;
};

// Contains build-local offsets that integration rebases into mesh arenas.
// A coarse cluster names representative corners of its own group's members only, so every corner it references lies in a finest cluster its group refines.
struct ClusterLodCluster {
    uint32_t VertexOffset{}, VertexCount{};
    uint32_t LocalTriangleOffset{}, TriangleCount{};
    uint32_t Primitive{};
    uint32_t ConeAxisCutoff{}; // four packed s8, cutoff 127 never culls
    vec3 Center{};
    float Radius{};
    uint32_t GroupIndex{};
    uint32_t RefinedGroup{};
};

// A DAG group: the merged bounds of its members and the error its simplification introduces.
// A terminal group, which no coarser level replaces, carries error FLT_MAX.
struct ClusterLodGroup {
    vec3 Center{};
    float Radius{};
    float Error{};
    uint32_t FirstCluster{}, ClusterCount{};
    uint32_t Primitive{};
};

// Describes one primitive's part of the build.
// Coarse clusters and groups for a primitive are contiguous.
struct ClusterLodPrimitiveRange {
    uint32_t FirstCluster{}, ClusterCount{};
    uint32_t FirstGroup{}, GroupCount{};
    // RootNode covers the full record run.
    // FinestNode covers the original-geometry prefix.
    uint32_t RootNode{}, FinestNode{};
    // Converts normalized attribute weights to mesh units for every simplification of the primitive.
    float SimplifyScale{};
};

// Cluster IDs below Level0Count identify inputs.
// Higher IDs index Clusters after subtracting Level0Count.
struct ClusterLodBuild {
    std::vector<ClusterLodCluster> Clusters;
    std::vector<uint32_t> VertexCorners;
    std::vector<uint8_t> LocalTriangles;
    std::vector<ClusterLodGroup> Groups;
    std::vector<uint32_t> GroupClusters;
    // Stores one primitive-ordered span tree per primitive with contiguous record ranges and conservative bounds and error.
    // A zero child count marks a leaf.
    std::vector<LodNode> Nodes;
    std::vector<uint32_t> Level0Groups;
    std::vector<ClusterLodPrimitiveRange> PrimitiveRanges;
    uint32_t LevelCount{};
    uint32_t NodeDepth{};

    uint32_t Level0Count() const { return uint32_t(Level0Groups.size()); }
};

// `serial` uses the reference remap and processes groups sequentially while preserving parallel-run output bytes.
ClusterLodBuild BuildClusterLod(const ClusterLodMesh &, bool serial = false);

// Rebuilds the DAG above existing clusters with BuildClusterLod's level loop and the whole build's weights, limits and error scaling.
// The mesh's single primitive lists every cluster's local triangles in cluster order.
// `levels` holds each cluster's level, and a cluster joins the pending pool at its level.
// Every open edge that no other pooled cluster shares keeps its vertices, so the outline against untouched neighbors holds.
// The result follows BuildClusterLod's layout with the existing clusters as its inputs, and holds no span tree.
// `scale` is the primitive's SimplifyScale.
ClusterLodBuild RebuildClusterLod(const ClusterLodMesh &, std::span<const uint32_t> levels, float scale);
