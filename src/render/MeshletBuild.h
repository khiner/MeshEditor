#pragma once

#include "gpu/DrawData.h"
#include "gpu/MeshletRecord.h"
#include "gpu/PrimitiveRecord.h"
#include "gpu/Vertex.h"
#include "mesh/MeshStore.h"

#include <span>
#include <vector>

struct GpuBuffers;
struct MeshBuffers;

// Everything one mesh's meshlet build reads, gathered before any of the batch's builds run.
// The spans point into the mesh store's arenas and the shared face-index arena, which hold still
// until the whole batch commits.
struct MeshletBuildInputs {
    std::span<const uint32_t> Indices; // One corner index per triangle corner
    std::span<const Vertex> Vertices;
    std::span<const uint32_t> CornerClasses; // Empty when every corner takes UniformClassWord
    std::span<const uint32_t> FaceIds; // 1-indexed source face per triangle
    std::span<const uint32_t> ElementPrimitives;
    std::span<const uvec2> CustomCornerMasks;
    std::span<const vec4> CornerTangents, CornerColors;
    std::array<std::span<const vec2>, MeshStore::MaxUvSets> CornerUvs;
    std::vector<PrimitiveTriangleRange> PrimitiveTriangleRanges;
    uint32_t CornerClassOffset{}; // UniformFaceOffset classifies every corner Face, anything else Vertex
    uint32_t TriangleCount{}; // Triangles across every primitive
    uint32_t ElementCount{}; // Edges of a line mesh, vertices of a point mesh, zero with faces
    uint32_t SourcePrimitiveCount{}; // Primitives a face-less mesh groups its elements into
    bool FaceTopology{}, LineTopology{}, MorphShadingAuthored{};
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
    uint32_t TriangleIdCount{}, LocalTriangleCount{};
};

// Read every input the build needs while the arenas hold still.
MeshletBuildInputs CaptureMeshletInputs(const GpuBuffers &, const MeshBuffers &, const Mesh &, const MeshStore &);
// Clusterize into plain vectors, touching no arena, so this runs on any thread.
MeshletBuild BuildMeshlets(const MeshletBuildInputs &);
// Release the mesh's previous meshlet ranges, place the finished build, and rebase its offsets.
// Serial, because arena offsets follow call order.
void CommitMeshlets(GpuBuffers &, MeshBuffers &, MeshletBuild &);
