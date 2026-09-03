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

// Borrows stable arena spans until the batch commits.
struct MeshletBuildInputs {
    std::span<const uint32_t> Indices;
    std::span<const Vertex> Vertices;
    std::span<const uint32_t> ElementPrimitives;
    std::vector<uint32_t> TriangleEditEdges;
    std::vector<PrimitiveTriangleRange> PrimitiveTriangleRanges;
    // The corner attributes the render-vertex weld keys on, shared with the cluster LOD build.
    CornerWeldSource Weld;
    uint32_t TriangleCount{};
    uint32_t ElementCount{};
    uint32_t EdgeCount{};
    uint32_t SourcePrimitiveCount{};
    bool FaceTopology{}, LineTopology{};
    SlotOffset AuxIndices{};
    // Derived from the topology fields above.
    std::vector<DrawData> PrimitiveDraws{};
    DrawData ElementDraw{};
    std::vector<uint32_t> EdgeIndices{};
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

// Captures stable input spans for the duration of the batch.
MeshletBuildInputs CaptureMeshletInputs(const GpuBuffers &, const MeshBuffers &, const Mesh &, const MeshStore &);
// Builds meshlets in host vectors and consumes TriangleEditEdges.
MeshletBuild BuildMeshlets(MeshletBuildInputs &);
// A face-less mesh, and one whose clusters fit a single partition, returns an empty build.
ClusterLodBuild BuildMeshletClusterLod(const MeshletBuildInputs &, const MeshletBuild &);
// Release the mesh's previous meshlet ranges, place the finished build, and rebase its offsets.
// Serial, because arena offsets follow call order.
void CommitMeshlets(GpuBuffers &, MeshBuffers &, MeshletBuild &);
// Places a finished DAG serially and appends coarse clusters after each primitive's original geometry.
void CommitClusterLod(GpuBuffers &, MeshBuffers &, const ClusterLodBuild &);
