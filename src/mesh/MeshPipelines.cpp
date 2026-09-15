#include "mesh/MeshPipelines.h"

#include "state/Scene.h"

MeshPipelines::MeshPipelines(mtl::LibraryCache &libraries)
    : VertexAdjacency{libraries}, VertexWeld{libraries}, MeshConnectivity{libraries} {}

MeshPipelines &GetMeshPipelines(state::Scene &r) {
    if (auto *pipelines = r.ctx().find<MeshPipelines>()) return *pipelines;
    return r.ctx().emplace<MeshPipelines>(r.ctx().get<mtl::LibraryCache>());
}
