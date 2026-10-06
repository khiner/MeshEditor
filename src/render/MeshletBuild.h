#pragma once

#include "Range.h"
#include "gpu/MeshRecord.h"
#include "gpu/MeshletRecord.h"
#include "gpu/PrimitiveRecord.h"
#include "gpu/Vertex.h"
#include "mesh/ElementAttributeView.h"
#include "mesh/MeshStore.h"
#include "render/ClusterLod.h"
#include "render/CornerWeldKey.h"

#include <span>
#include <vector>

static_assert(MaxWeldUvSets == MeshStore::MaxUvSets);

struct RenderArenas;

// Borrows stable arena spans until the batch commits.
struct MeshletBuildInputs {
    TriangleVertexView Indices;
    std::span<const Vertex> Vertices;
    uint32_t VertexFirst{};
    Range DenseVertices{};
    CornerNormalView Normals;
    // The corner attributes the render-vertex weld keys on, shared with the cluster LOD build.
    CornerWeldSource Weld;
    bool FaceTopology{};
};

// Captures stable input spans for the duration of the batch.
MeshletBuildInputs CaptureMeshletInputs(const Mesh &, const MeshStore &, TriangleCorners);
// Builds the DAG over the mesh's committed level-zero clusters.
// A face-less mesh, and one whose clusters fit a single partition, returns an empty build.
ClusterLodBuild BuildMeshletClusterLod(const MeshStore &, const MeshStore::Record &, const MeshletBuildInputs &, std::span<const uint32_t> primitive_triangle_counts = {});
// Places each owner's finished DAG, retaining finest identities and allocating only new coarse records.
void CommitClusterLods(state::Scene &, std::span<MeshStore::Record *const>, std::span<const ClusterLodBuild>);

// Allocates coarse records and their geometry, then publishes each group's
// member/proxy run and proxy IDs directly in UMA. Callers fill member IDs in
// their established order and retain the returned coarse cluster run.
Range PublishClusterLodStorage(RenderArenas &, const ClusterLodBuild &, std::span<const uint32_t> primitive_ids, Range &groups, Range &vertices, Range &local_triangles);
