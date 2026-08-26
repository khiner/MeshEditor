#pragma once

#include "gpu/LodNode.h"
#include "numeric/vec3.h"
#include "render/CornerWeldKey.h"

#include <cstdint>
#include <span>
#include <vector>

// A cluster's vertex and triangle limits, fixed by the visibility id's six triangle bits and the
// mesh shader's output contract.
inline constexpr uint32_t ClusterLodMaxVertices{64};
inline constexpr uint32_t ClusterLodMaxTriangles{48};
// A cluster below this triangle count merges into a neighbour instead of standing on its own.
inline constexpr uint32_t ClusterLodMinTriangles{16};
// Clusters per DAG group. A mesh with no more clusters than this has no coarser level to reach.
inline constexpr uint32_t ClusterLodPartitionSize{16};
// Render records one span-tree leaf covers, which the frontier buffers size themselves against.
inline constexpr uint32_t ClusterLodSpanLeafRecords{64};
// Marks an absent cluster, group or node.
inline constexpr uint32_t ClusterLodInvalid{~0u};

struct ClusterLodPrimitive {
    uint32_t FirstTriangle{}, TriangleCount{};
    uint32_t FirstCluster{}, ClusterCount{};
};

// One level-0 cluster. Its render record carries this bounding sphere, which the span tree's leaves
// contain.
struct ClusterLodSourceCluster {
    uint32_t FirstVertex{}, VertexCount{}; // range in ClusterLodMesh::SourceVertexCorners
    uint32_t FirstLocalTriangle{}, TriangleCount{}; // three bytes per triangle in SourceLocalTriangles
    vec3 Center{};
    float Radius{};
    bool ConeCullSafe{};
};

// A mesh's geometry as the DAG build reads it. Every span is borrowed for the call.
struct ClusterLodMesh {
    // Three source vertex ids per triangle.
    std::span<const uint32_t> CornerVertices;
    // Vertex positions, float3 in the first twelve bytes of each vertex.
    const float *Positions{};
    size_t PositionStride{};
    // Resolved shading normal per corner. Empty derives area-weighted normals from the geometry.
    std::span<const vec3> CornerNormals;

    CornerWeldSource Weld;

    std::span<const ClusterLodPrimitive> Primitives;
    std::span<const ClusterLodSourceCluster> Clusters;
    // The level-zero meshlets' native geometry. Vertex corners and local triangle bytes may carry the
    // render encoding's high flag bits; the DAG reads only their corner and local-index fields.
    std::span<const uint32_t> SourceVertexCorners;
    std::span<const uint8_t> SourceLocalTriangles;
};

// A coarse cluster in engine record form. Offsets are local to the build, and integration rebases
// them into the mesh arenas.
struct ClusterLodCluster {
    uint32_t VertexOffset{}, VertexCount{}; // range in ClusterLodBuild::VertexCorners
    uint32_t LocalTriangleOffset{}, TriangleCount{}; // three local indices per triangle in ClusterLodBuild::LocalTriangles
    uint32_t Primitive{}; // primitive index within the mesh
    uint32_t ConeAxisCutoff{}; // four packed s8, cutoff 127 never culls
    vec3 Center{};
    float Radius{};
    uint32_t GroupIndex{}; // the group whose simplification replaces this cluster
    uint32_t RefinedGroup{}; // the group this cluster was simplified from, ClusterLodInvalid at level 0
};

// A DAG group: the merged bounds of its members and the error its simplification introduces.
// A terminal group, which no coarser level replaces, carries error FLT_MAX.
struct ClusterLodGroup {
    vec3 Center{};
    float Radius{};
    float Error{};
    uint32_t FirstCluster{}, ClusterCount{}; // range in ClusterLodBuild::GroupClusters
    uint32_t Primitive{};
};

// Partition, lock and merge are wall time over the level, and the three per-group phases sum the
// time each group spent.
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

// One primitive's share of the build. Coarse clusters and groups of a primitive are contiguous.
struct ClusterLodPrimitiveRange {
    uint32_t FirstCluster{}, ClusterCount{};
    uint32_t FirstGroup{}, GroupCount{};
    // Where a traversal of this primitive starts: the span tree over its whole record run, and the
    // leaf covering the original-geometry prefix a pinned instance draws.
    uint32_t RootNode{}, FinestNode{};
};

// A mesh's cluster LOD DAG. A build cluster id below Level0Count() names an input level-0 cluster,
// and above it names Clusters[id - Level0Count()].
struct ClusterLodBuild {
    std::vector<ClusterLodCluster> Clusters; // coarse clusters, in group order
    std::vector<uint32_t> VertexCorners; // primitive-local source corner id per cluster vertex
    std::vector<uint8_t> LocalTriangles; // three cluster-local vertex indices per triangle
    std::vector<ClusterLodGroup> Groups;
    std::vector<uint32_t> GroupClusters; // build cluster ids, grouped by group
    // Every primitive's span tree, in primitive order, each node covering a contiguous run of the
    // primitive's render records. A node's bounds contain the run's cluster and group spheres and its
    // error is the run's worst, and a zero child count marks a leaf.
    std::vector<LodNode> Nodes;
    std::vector<uint32_t> Level0Groups; // the group each input level-0 cluster belongs to
    std::vector<ClusterLodPrimitiveRange> PrimitiveRanges;
    uint32_t LevelCount{};
    uint32_t NodeDepth{}; // the deepest primitive's expansions from root to leaf
    ClusterLodStats Stats;

    uint32_t Level0Count() const { return uint32_t(Level0Groups.size()); }
};

// Builds a mesh's cluster LOD DAG from its level-0 clusters.
// `serial` uses the reference position remap and runs each level's groups one at a time, producing the
// same bytes as the parallel run.
ClusterLodBuild BuildClusterLod(const ClusterLodMesh &, bool serial = false);
