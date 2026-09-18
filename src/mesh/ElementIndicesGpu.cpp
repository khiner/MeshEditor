#include "mesh/ElementIndicesGpu.h"

#include "gpu/ElementIndicesJob.h"
#include "gpu/TiledJobPushConstants.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchChunks.h"
#include "mesh/TiledJobBatch.h"
#include "state/Scene.h"

namespace {
constexpr std::array Passes{TiledPass{MeshPass::EdgeEndpointsWrite, 0}, TiledPass{MeshPass::TriangleIndicesWrite, 1}};
} // namespace

void WriteElementIndicesNow(state::Scene &r, std::span<const ElementIndicesWork> work) {
    if (work.empty()) return;
    const auto &meshes = r.ctx().get<const MeshStore>();
    const auto &arenas = meshes.Arenas();
    TiledJobBatch<ElementIndicesJob, 2> batch{meshes.BufferContext(), 1, uint32_t(work.size())};
    batch.Begin();
    for (const auto &item : work) {
        const auto &record = meshes.Get(item.StoreId);
        const auto corners = arenas.FaceCorners.Slotted(record.FaceCorners);
        const auto run = arenas.Connectivity.Slotted(record.Connectivity);
        const auto face_ids = arenas.TriangleFaceIds.Slotted(record.TriangleFaceIds);
        const auto first_triangles = arenas.FaceFirstTriangles.Slotted(record.FaceData);
        const auto edge_count = item.Endpoints.Count / 2u, triangle_count = item.Triangles.Count / 3u;
        batch.AddJob(
            ElementIndicesJob{
                .Corners = {corners.Slot, corners.Offset},
                .Connectivity = {run.Slot, run.Offset},
                .Endpoints = {item.Endpoints.Slot, item.Endpoints.Offset},
                .Triangles = {item.Triangles.Slot, item.Triangles.Offset},
                .TriangleFaceIds = {face_ids.Slot, face_ids.Offset},
                .FaceFirstTriangles = {first_triangles.Slot, first_triangles.Offset},
                .VertexCount = record.ConnectivityVertices,
                .HalfedgeCount = record.ConnectivityHalfedges,
                .EdgeCount = edge_count,
                .FaceCount = record.ConnectivityFaces,
                .TriangleCount = triangle_count,
                .FaceStarts = record.ConnectivityFaceStarts ? 1u : 0u,
            },
            {TileCount(edge_count, TileElements), TileCount(triangle_count, TileElements)}
        );
    }
    batch.Submit(r.ctx().get<const mtl::Context>(), r.ctx().get<const mtl::BindlessSet>(), GetMeshPipelines(r), TiledJobPushConstants{}, Passes);
}
