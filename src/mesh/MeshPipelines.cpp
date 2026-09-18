#include "mesh/MeshPipelines.h"

#include "state/Scene.h"

namespace {
// The shader file and function of each pass, in MeshPass order.
constexpr std::array<std::pair<const char *, const char *>, size_t(MeshPass::Count)> PassFunctions{{
    {"MeshConnectivity.metal", "MeshConnectivityPrev"},
    {"MeshConnectivity.metal", "MeshConnectivityInit"},
    {"MeshConnectivity.metal", "MeshConnectivityInsert"},
    {"MeshConnectivity.metal", "MeshConnectivityResolve"},
    {"MeshConnectivity.metal", "MeshConnectivityLink"},
    {"MeshConnectivity.metal", "MeshConnectivityWordBlockSum"},
    {"MeshConnectivity.metal", "MeshConnectivityWordBlockPrefix"},
    {"MeshConnectivity.metal", "MeshConnectivityRanks"},
    {"MeshConnectivity.metal", "MeshConnectivityEdgeTables"},
    {"VertexAdjacency.metal", "VertexAdjacencyZero"},
    {"VertexAdjacency.metal", "VertexAdjacencyCount"},
    {"VertexAdjacency.metal", "VertexAdjacencyBlockSum"},
    {"VertexAdjacency.metal", "VertexAdjacencyBlockPrefix"},
    {"VertexAdjacency.metal", "VertexAdjacencyOffsets"},
    {"VertexAdjacency.metal", "VertexAdjacencyScatter"},
    {"VertexAdjacency.metal", "VertexAdjacencySort"},
    {"VertexWeld.metal", "VertexWeldTableInit"},
    {"VertexWeld.metal", "VertexWeldInsert"},
    {"VertexWeld.metal", "VertexWeldMarkReps"},
    {"VertexWeld.metal", "VertexWeldBlockSum"},
    {"VertexWeld.metal", "VertexWeldBlockPrefix"},
    {"VertexWeld.metal", "VertexWeldScan"},
    {"VertexWeld.metal", "VertexWeldEmit"},
    {"VertexWeld.metal", "VertexWeldRemapCorners"},
    {"VertexWeld.metal", "VertexWeldCompact"},
    {"VertexWeld.metal", "VertexWeldWriteBack"},
    {"ElementIndices.metal", "EdgeEndpointsWrite"},
    {"ElementIndices.metal", "TriangleIndicesWrite"},
    {"MeshTopology.metal", "TopologyFaceIndex"},
    {"MeshTopology.metal", "TopologyZero"},
    {"MeshTopology.metal", "TopologyMarkHalfedges"},
    {"MeshTopology.metal", "TopologyMarkFaces"},
    {"MeshTopology.metal", "TopologyLink"},
    {"MeshTopology.metal", "TopologyJump"},
    {"MeshTopology.metal", "TopologyConverge"},
    {"MeshTopology.metal", "TopologyDissolveRegions"},
    {"MeshTopology.metal", "TopologyDissolveWalk"},
    {"MeshTopology.metal", "TopologyDissolveRevert"},
    {"MeshTopology.metal", "TopologyJoinBest"},
    {"MeshTopology.metal", "TopologyJoinMatch"},
    {"MeshTopology.metal", "TopologyZeroVertices"},
    {"MeshTopology.metal", "TopologyMergeTable"},
    {"MeshTopology.metal", "TopologyMergeInsert"},
    {"MeshTopology.metal", "TopologyMergeQuery"},
    {"MeshTopology.metal", "TopologyDissolveLimitVertices"},
    {"MeshTopology.metal", "TopologyListFill"},
    {"MeshTopology.metal", "TopologyCountVertices"},
    {"MeshTopology.metal", "TopologyCountHalfedges"},
    {"MeshTopology.metal", "TopologyCountFaces"},
    {"MeshTopology.metal", "TopologyScanBlockSum"},
    {"MeshTopology.metal", "TopologyScanBlockPrefix"},
    {"MeshTopology.metal", "TopologyScanOffsets"},
    {"MeshTopology.metal", "TopologyZeroOutput"},
    {"MeshTopology.metal", "TopologyScatterVertices"},
    {"MeshTopology.metal", "TopologyScatterHalfedges"},
    {"MeshTopology.metal", "TopologyScatterFaces"},
    {"MeshTopology.metal", "TopologyFaceTables"},
    {"MeshTopology.metal", "TopologyGatherVertices"},
    {"MeshTopology.metal", "TopologyGatherCorners"},
    {"MeshTopology.metal", "TopologyCustomPopcount"},
    {"MeshTopology.metal", "TopologyCustomPack"},
    {"MeshTopology.metal", "TopologyEdgeAttributes"},
}};

template<size_t... I>
std::array<mtl::ComputePipeline, sizeof...(I)> CompilePasses(mtl::LibraryCache &libraries, std::index_sequence<I...>) {
    return {{mtl::ComputePipeline{libraries, {PassFunctions[I].first, PassFunctions[I].second}}...}};
}
} // namespace

MeshPipelines::MeshPipelines(mtl::LibraryCache &libraries)
    : Pipelines{CompilePasses(libraries, std::make_index_sequence<size_t(MeshPass::Count)>{})} {}

MeshPipelines &GetMeshPipelines(state::Scene &r) {
    if (auto *pipelines = r.ctx().find<MeshPipelines>()) return *pipelines;
    return r.ctx().emplace<MeshPipelines>(r.ctx().get<mtl::LibraryCache>());
}
