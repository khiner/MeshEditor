#pragma once

#include "gpu/DrawData.h"
#include "gpu/MeshletRecord.h"
#include "gpu/PrimitiveRecord.h"
#include "gpu/Vertex.h"
#include "mesh/MeshStore.h"
#include "render/ClusterLod.h"
#include "render/CornerWeldKey.h"

#include <span>
#include <vector>

static_assert(MaxWeldUvSets == MeshStore::MaxUvSets);

struct GpuBuffers;
struct MeshBuffers;

// Everything one mesh's meshlet build reads, gathered before any of the batch's builds run.
// The spans point into the mesh store's arenas and the shared face-index arena, which hold still
// until the whole batch commits.
struct MeshletBuildInputs {
    std::span<const uint32_t> Indices; // One corner index per triangle corner
    std::span<const Vertex> Vertices;
    std::span<const uint32_t> ElementPrimitives;
    std::vector<uint32_t> TriangleEditEdges; // Three canonical edge ids per source triangle; InvalidOffset for triangulation diagonals
    std::vector<PrimitiveTriangleRange> PrimitiveTriangleRanges;
    // The corner attributes the render-vertex weld keys on, shared with the cluster LOD build.
    CornerWeldSource Weld;
    uint32_t TriangleCount{}; // Triangles across every primitive
    uint32_t ElementCount{}; // Edges of a line mesh, vertices of a point mesh, zero with faces
    uint32_t EdgeCount{};
    uint32_t SourcePrimitiveCount{}; // Primitives a face-less mesh groups its elements into
    bool FaceTopology{}, LineTopology{};
    // Derived from the topology fields above.
    std::vector<DrawData> PrimitiveDraws{}; // One per entry of PrimitiveTriangleRanges
    DrawData ElementDraw{}; // Every line or point primitive of a face-less mesh draws through this
    std::vector<uint32_t> EdgeIndices{}; // Endpoint pairs of a line mesh's edges
};

// One mesh's finished meshlets, in the arena layout the commit places them at.
// Offsets are relative to the mesh's own ranges until the commit rebases them.
struct MeshletBuild {
    std::vector<MeshletRecord> Records{};
    std::vector<uint32_t> Vertices{};
    std::vector<PrimitiveRecord> Primitives{};
    std::vector<uint32_t> TriangleIds{};
    std::vector<uint8_t> LocalTriangles{};
    std::vector<uint32_t> EditEdges{};
    uint32_t TriangleIdCount{}, LocalTriangleCount{};
};

// Read every input the build needs while the arenas hold still.
MeshletBuildInputs CaptureMeshletInputs(const GpuBuffers &, const MeshBuffers &, const Mesh &, const MeshStore &);
// Clusterize into plain vectors, consuming TriangleEditEdges but touching no arena, so this runs on any thread.
MeshletBuild BuildMeshlets(MeshletBuildInputs &);
// Build the DAG over a finished level-0 build, on the same inputs and the same thread.
// A face-less mesh, and one whose clusters fit a single partition, returns an empty build.
ClusterLodBuild BuildMeshletClusterLod(const MeshletBuildInputs &, const MeshletBuild &);
// Release the mesh's previous meshlet ranges, place the finished build, and rebase its offsets.
// Serial, because arena offsets follow call order.
void CommitMeshlets(GpuBuffers &, MeshBuffers &, MeshletBuild &);
// Place a finished DAG: its groups and nodes in their arenas, and its coarse clusters behind each
// primitive's original geometry in a rewritten meshlet run. Serial, like the meshlet commit.
void CommitClusterLod(GpuBuffers &, MeshBuffers &, const ClusterLodBuild &);
