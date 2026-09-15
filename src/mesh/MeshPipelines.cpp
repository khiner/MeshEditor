#include "mesh/MeshPipelines.h"

#include "state/Scene.h"

namespace {
// The shader file and function of each pass, in MeshPass order.
constexpr std::array<std::pair<const char *, const char *>, size_t(MeshPass::Count)> PassFunctions{{
    {"MeshConnectivity.metal", "MeshConnectivityZero"},
    {"MeshConnectivity.metal", "MeshConnectivityCount"},
    {"MeshConnectivity.metal", "MeshConnectivityBlockSum"},
    {"MeshConnectivity.metal", "MeshConnectivityBlockPrefix"},
    {"MeshConnectivity.metal", "MeshConnectivityOffsets"},
    {"MeshConnectivity.metal", "MeshConnectivityScatter"},
    {"MeshConnectivity.metal", "MeshConnectivityPair"},
    {"MeshConnectivity.metal", "MeshConnectivityBits"},
    {"MeshConnectivity.metal", "MeshConnectivityWordBlockSum"},
    {"MeshConnectivity.metal", "MeshConnectivityWordBlockPrefix"},
    {"MeshConnectivity.metal", "MeshConnectivityRanks"},
    {"MeshConnectivity.metal", "MeshConnectivitySamples"},
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
