#include "mesh/MeshPipelines.h"

#include "Profile.h"
#include "state/Scene.h"

namespace {
// The shader file and function of each pass, in MeshPass order.
constexpr std::array<std::pair<const char *, const char *>, size_t(MeshPass::Count)> PassFunctions{{
    {"MeshConnectivity.metal", "MeshConnectivityFaces"},
    {"MeshConnectivity.metal", "MeshConnectivityInit"},
    {"MeshConnectivity.metal", "MeshConnectivityInsert"},
    {"MeshConnectivity.metal", "MeshConnectivityMatchEdges"},
    {"MeshConnectivity.metal", "MeshConnectivityClassifyEdges"},
    {"MeshConnectivity.metal", "MeshConnectivityResolve"},
    {"MeshConnectivity.metal", "MeshConnectivityLink"},
    {"MeshConnectivity.metal", "MeshConnectivityWordBlockSum"},
    {"MeshConnectivity.metal", "MeshConnectivityWordBlockPrefix"},
    {"MeshConnectivity.metal", "MeshConnectivityRanks"},
    {"MeshConnectivity.metal", "MeshConnectivityCounts"},
    {"MeshConnectivity.metal", "MeshConnectivityRetiredEdges"},
    {"MeshConnectivity.metal", "MeshConnectivityEdgeTables"},
    {"VertexFanBuild.metal", "VertexFanInit"},
    {"VertexFanBuild.metal", "VertexFanKeys"},
    {"VertexFanBuild.metal", "VertexFanTileScan"},
    {"VertexFanBuild.metal", "VertexFanTilePrefix"},
    {"VertexFanBuild.metal", "VertexFanHistogram"},
    {"VertexFanBuild.metal", "VertexFanPrefix"},
    {"VertexFanBuild.metal", "VertexFanScatter"},
    {"VertexFanBuild.metal", "VertexFanEmit"},
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
    {"VertexNormalDerive.metal", "VertexNormalDeriveKernel"},
    {"RetessellateFaces.metal", "RetessellateFaces"},
    {"EditSharpness.metal", "EditSharpnessKernel"},
    {"ElementWorkSeed.metal", "ElementWorkSeed"},
    {"CornerClassification.metal", "CornerClassificationCount"},
    {"CornerClassification.metal", "CornerClassificationPlan"},
    {"CornerClassification.metal", "CornerClassificationWrite"},
    {"CornerClassification.metal", "CornerClassificationBlocks"},
    {"MeshClosure.metal", "MeshClosureCount"},
    {"MeshClosure.metal", "MeshClosureExpand"},
    {"MeshClosure.metal", "MeshClosureEdgeVertices"},
    {"MeshClosure.metal", "MeshClosureTriangles"},
    {"SelectionUpdate.metal", "MarkSelectionNeighbors"},
    {"SelectionUpdate.metal", "UpdateSelectionBlocks"},
    {"SelectionUpdate.metal", "GatherSelectedElements"},
    {"ConnectivityEdit.metal", "ConnectivityEditWork"},
    {"SpatialFaceQuery.metal", "SpatialFaceQueryExpand"},
    {"TopologyIdentity.metal", "TopologyIdentityInit"},
    {"TopologyIdentity.metal", "TopologyIdentityRetain"},
    {"TopologyIdentity.metal", "TopologyIdentityReplaced"},
    {"TopologyIdentity.metal", "TopologyIdentityNew"},
    {"TopologyIdentity.metal", "TopologyIdentityAssign"},
    {"ElementWorkSort.metal", "ElementWorkHistogram"},
    {"ElementWorkSort.metal", "ElementWorkPrefix"},
    {"ElementWorkSort.metal", "ElementWorkScatter"},
    {"ElementWorkSort.metal", "ElementWorkFinish"},
    {"TopologyCollapse.metal", "TopologyCollapseKeys"},
    {"TopologyCollapse.metal", "TopologyCollapseHistogram"},
    {"TopologyCollapse.metal", "TopologyCollapsePrefix"},
    {"TopologyCollapse.metal", "TopologyCollapseScatter"},
    {"TopologyCollapse.metal", "TopologyCollapseReduce"},
    {"TopologyCollapse.metal", "TopologyCollapseCarry"},
    {"TopologyCollapse.metal", "TopologyCollapseCenters"},
    {"MeshTopology.metal", "TopologySelection"},
    {"MeshTopology.metal", "TopologyZero"},
    {"MeshTopology.metal", "TopologyMarkHalfedges"},
    {"MeshTopology.metal", "TopologyMarkFaces"},
    {"MeshTopology.metal", "TopologyMarkRetainedEdges"},
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
    {"MeshTopology.metal", "TopologyLineKeys"},
    {"MeshTopology.metal", "TopologyFinalizeVertices"},
    {"MeshTopology.metal", "TopologyListFill"},
    {"MeshTopology.metal", "TopologyCountVertices"},
    {"MeshTopology.metal", "TopologyCountHalfedges"},
    {"MeshTopology.metal", "TopologyCountFaces"},
    {"MeshTopology.metal", "TopologyScanBlockSum"},
    {"MeshTopology.metal", "TopologyScanBlockPrefix"},
    {"MeshTopology.metal", "TopologyScanOffsets"},
    {"MeshTopology.metal", "TopologyScatterVertices"},
    {"MeshTopology.metal", "TopologyScatterHalfedges"},
    {"MeshTopology.metal", "TopologyScatterFaces"},
    {"MeshTopology.metal", "TopologyFaceTables"},
    {"MeshTopology.metal", "TopologyGatherVertices"},
    {"InsetPreview.metal", "InsetPreviewPositions"},
    {"VertexPositionEdit.metal", "PositionVerticesGather"},
    {"EdgeChains.metal", "SpaceEvenlyGather"},
    {"EdgeChains.metal", "RelaxEdgeLoopsGather"},
    {"EdgeChains.metal", "CurveBetweenSelected"},
    {"Circularize.metal", "Circularize"},
    {"Flatten.metal", "FlattenGroups"},
    {"VertexPositionEdit.metal", "PositionStatisticsGather"},
    {"VertexPositionEdit.metal", "PositionStatisticsReduce"},
    {"VertexPositionEdit.metal", "VertexSlideReference"},
    {"RecalculateNormals.metal", "RecalculateNormalFaceCenters"},
    {"RecalculateNormals.metal", "RecalculateNormalCenters"},
    {"RecalculateNormals.metal", "RecalculateNormalCandidates"},
    {"RecalculateNormals.metal", "RecalculateNormalOrientation"},
    {"VertexPositionEdit.metal", "PlanarFacePlanes"},
    {"VertexPositionEdit.metal", "WriteEditedPositions"},
    {"FaceAttributeEdit.metal", "EditFaceUvs"},
    {"FaceAttributeEdit.metal", "EditFaceColors"},
    {"MeshletBoundsRefit.metal", "MeshletBoundsRefit"},
    {"MeshTopology.metal", "TopologyGatherCorners"},
    {"MeshTopology.metal", "TopologyCustomNormals"},
    {"MeshTopology.metal", "TopologyEdgeAttributes"},
    {"LodNodeRefit.metal", "LodNodeRefit"},
    {"MeshletOwners.metal", "MeshletOwners"},
    {"MeshletBuild.metal", "MeshletBuildElements"},
    {"MeshletBuild.metal", "MeshletBuildMaterials"},
    {"MeshletBuild.metal", "MeshletBuildInit"},
    {"MeshletBuild.metal", "MeshletBuildBounds"},
    {"MeshletBuild.metal", "MeshletBuildKeys"},
    {"MeshletBuild.metal", "MeshletBuildHistogram"},
    {"MeshletBuild.metal", "MeshletBuildHistogramPrefix"},
    {"MeshletBuild.metal", "MeshletBuildScatter"},
    {"MeshletBuild.metal", "MeshletBuildSegments"},
    {"MeshletBuild.metal", "MeshletBuildTiles"},
    {"MeshletBuild.metal", "MeshletBuildClusters"},
    {"MeshletBuild.metal", "MeshletBuildOffsets"},
    {"MeshletBuild.metal", "MeshletBuildEmit"},
    {"MeshletBuild.metal", "MeshletBuildPrimitives"},
    {"MeshClone.metal", "CopyByteRuns"},
    {"MeshClone.metal", "GatherByRank"},
    {"MeshClone.metal", "RebaseByBlock"},
    {"MeshClone.metal", "RebaseByRank"},
}};

} // namespace

MeshPipelines::MeshPipelines(mtl::LibraryCache &libraries) : Libraries(libraries) {}

void MeshPipelines::PrewarmAsync() {
    if (PrewarmWorker.joinable() || !Libraries.PipelineCompiler()) return;
    PrewarmWorker = std::jthread([this](std::stop_token stop) {
        try {
            auto cache = Libraries.PrewarmCache();
            for (size_t i = 0u; i < PassFunctions.size() && !stop.stop_requested(); ++i) {
                {
                    std::lock_guard lock{Mutex};
                    if (Pipelines[i]) continue;
                }
                try {
                    mtl::ComputePipeline pipeline{*cache, {PassFunctions[i].first, PassFunctions[i].second}};
                    std::lock_guard lock{Mutex};
                    if (!Pipelines[i]) Pipelines[i].emplace(std::move(pipeline));
                } catch (...) {
                    // Foreground use retains the normal error path for a shader
                    // edited after the offline archive was built.
                }
            }
        } catch (...) {
            // A missing archive does not change foreground shader behavior.
        }
    });
}

const mtl::ComputePipeline &MeshPipelines::operator[](MeshPass pass) const {
    const auto index = size_t(pass);
    std::lock_guard lock{Mutex};
    auto &pipeline = Pipelines.at(index);
    if (!pipeline) {
        const profile::CpuScope scope{"MeshPipelineCreate"};
        pipeline.emplace(Libraries, mtl::FunctionRef{PassFunctions[index].first, PassFunctions[index].second});
    }
    return *pipeline;
}

MeshPipelines &GetMeshPipelines(state::Scene &r) {
    if (auto *pipelines = r.Context.find<MeshPipelines>()) return *pipelines;
    return r.Context.emplace<MeshPipelines>(r.Context.get<mtl::LibraryCache>());
}
