#include "render/MeshUpdates.h"
#include "mesh/MeshBvh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "project/Registry.h"
#include "render/GpuBufferOps.h"
#include "render/MeshBuffers.h"
#include <entt/entity/registry.hpp>
// Rebuild the closest-point hierarchy, curvature, and volume derived from a mesh.
void UpdateMeshBvh(entt::registry &r, entt::entity mesh_entity) {
    const auto mesh = GetMesh(r, mesh_entity);
    const auto indices = GetFaceIndices(r, mesh, r.get<const MeshBuffers>(mesh_entity));
    // A mesh of points or lines has no surface.
    if (indices.empty()) {
        project::Remove<MeshBvh>(r, mesh_entity);
        return;
    }
    auto bvh = BuildMeshBvh(mesh.GetVerticesSpan(), indices);
    bvh.MeanCurvature = mesh.CalcMeanCurvatures(r.ctx().get<const MeshStore>().GetEdgeSharpness(mesh.GetStoreId()));
    bvh.EnclosedVolume = mesh.CalcEnclosedVolume();
    project::EmplaceOrReplace<MeshBvh>(r, mesh_entity, std::move(bvh));
}
