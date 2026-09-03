#pragma once

#include "gpu/LodNode.h"
#include "gpu/MeshletLimit.h"
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
// Render records one span-tree leaf covers, which the frontier buffers size themselves against.
inline constexpr uint32_t ClusterLodSpanLeafRecords{64};
inline constexpr uint32_t ClusterLodInvalid{~0u};

struct ClusterLodPrimitive {
    uint32_t FirstTriangle{}, TriangleCount{};
    uint32_t FirstCluster{}, ClusterCount{};
};

// Describes one level-zero cluster and its render-record bounding sphere.
struct ClusterLodSourceCluster {
    uint32_t FirstVertex{}, VertexCount{};
    uint32_t FirstLocalTriangle{}, TriangleCount{};
    vec3 Center{};
    float Radius{};
    bool ConeCullSafe{};
};

// Provides mesh geometry to the DAG build.
// Every span is borrowed for the duration of the call.
struct ClusterLodMesh {
    std::span<const uint32_t> CornerVertices;
    // Vertex positions, float3 in the first twelve bytes of each vertex.
    const float *Positions{};
    size_t PositionStride{};
    // Resolved shading normal per corner.
    // An empty span requests area-weighted normals derived from geometry.
    std::span<const vec3> CornerNormals;

    CornerWeldSource Weld;

    std::span<const ClusterLodPrimitive> Primitives;
    std::span<const ClusterLodSourceCluster> Clusters;
    // DAG construction masks render-encoding flags from level-zero vertex corners and local triangle bytes.
    std::span<const uint32_t> SourceVertexCorners;
    std::span<const uint8_t> SourceLocalTriangles;
};

// Contains build-local offsets that integration rebases into mesh arenas.
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

// Partition, lock, and merge measure level wall time.
// Per-group phases accumulate group durations.
struct ClusterLodLevelStats {
    uint32_t Groups{}, Clusters{}, Triangles{};
    uint32_t StuckClusters{}, StuckTriangles{}, SingletonGroups{};
    float MeanRadius{};
    double PartitionMs{}, LockMs{}, SimplifyMs{}, ClusterizeMs{}, EmitMs{}, MergeMs{}, LevelMs{};
};

struct ClusterLodStats {
    double WeldMs{}, Level0Ms{}, HierarchyMs{}, TotalMs{};
    std::vector<ClusterLodLevelStats> Levels;
};

// Describes one primitive's part of the build.
// Coarse clusters and groups for a primitive are contiguous.
struct ClusterLodPrimitiveRange {
    uint32_t FirstCluster{}, ClusterCount{};
    uint32_t FirstGroup{}, GroupCount{};
    // RootNode covers the full record run.
    // FinestNode covers the original-geometry prefix.
    uint32_t RootNode{}, FinestNode{};
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
    ClusterLodStats Stats;

    uint32_t Level0Count() const { return uint32_t(Level0Groups.size()); }
};

// `serial` uses the reference remap and processes groups sequentially while preserving parallel-run output bytes.
ClusterLodBuild BuildClusterLod(const ClusterLodMesh &, bool serial = false);
