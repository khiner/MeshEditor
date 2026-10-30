#include "mesh/MeshPipelines.h"

#include "Field.h"
#include "Profile.h"
#include "state/Registry.h"

namespace {
// Each pass runs the kernel named after it, in its family's shader file, with the family's prefix and suffix around the name.
struct PassFamily {
    MeshPass First, Last;
    const char *File;
    const char *Prefix{""}, *Suffix{""};
};
constexpr std::array Families{
    PassFamily{MeshPass::ConnectivityFaces, MeshPass::ConnectivityEdgeTables, "MeshConnectivity.metal", "Mesh"},
    PassFamily{MeshPass::VertexFanInit, MeshPass::VertexFanEmit, "VertexFanBuild.metal"},
    PassFamily{MeshPass::WeldTableInit, MeshPass::WeldWriteBack, "VertexWeld.metal", "Vertex"},
    PassFamily{MeshPass::VertexNormalDerive, MeshPass::VertexNormalDerive, "VertexNormalDerive.metal", "", "Kernel"},
    PassFamily{MeshPass::RetessellateFaces, MeshPass::RetessellateFaces, "RetessellateFaces.metal"},
    PassFamily{MeshPass::EditSharpness, MeshPass::EditSharpness, "EditSharpness.metal", "", "Kernel"},
    PassFamily{MeshPass::ElementWorkSeed, MeshPass::ElementWorkSeed, "ElementWorkSeed.metal"},
    PassFamily{MeshPass::CornerClassificationCount, MeshPass::CornerClassificationWrite, "CornerClassification.metal"},
    PassFamily{MeshPass::MeshClosureCount, MeshPass::MeshClosureTriangles, "MeshClosure.metal"},
    PassFamily{MeshPass::MarkSelectionNeighbors, MeshPass::GatherSelectedElements, "SelectionUpdate.metal"},
    PassFamily{MeshPass::ConnectivityEditWork, MeshPass::ConnectivityEditWork, "ConnectivityEdit.metal"},
    PassFamily{MeshPass::SpatialFaceQueryExpand, MeshPass::SpatialFaceQueryExpand, "SpatialFaceQuery.metal"},
    PassFamily{MeshPass::BooleanVertices, MeshPass::BooleanFaces, "MeshBoolean.metal"},
    PassFamily{MeshPass::FlagShortEdges, MeshPass::FlagShortEdges, "ShortEdges.metal"},
    PassFamily{MeshPass::TopologyIdentityInit, MeshPass::TopologyIdentityAssign, "TopologyIdentity.metal"},
    PassFamily{MeshPass::ElementWorkHistogram, MeshPass::ElementWorkFinish, "ElementWorkSort.metal"},
    PassFamily{MeshPass::TopologyCollapseKeys, MeshPass::TopologyCollapseCenters, "TopologyCollapse.metal"},
    PassFamily{MeshPass::TopologySelection, MeshPass::TopologyGatherVertices, "MeshTopology.metal"},
    PassFamily{MeshPass::PositionVerticesGather, MeshPass::PositionVerticesGather, "VertexPositionEdit.metal"},
    PassFamily{MeshPass::SpaceEvenlyGather, MeshPass::CurveBetweenSelected, "EdgeChains.metal"},
    PassFamily{MeshPass::Circularize, MeshPass::Circularize, "Circularize.metal"},
    PassFamily{MeshPass::FlattenGroups, MeshPass::FlattenGroups, "Flatten.metal"},
    PassFamily{MeshPass::PositionStatisticsGather, MeshPass::VertexSlideReference, "VertexPositionEdit.metal"},
    PassFamily{MeshPass::RecalculateNormalFaceCenters, MeshPass::RecalculateNormalOrientation, "RecalculateNormals.metal"},
    PassFamily{MeshPass::PlanarFacePlanes, MeshPass::WriteEditedPositions, "VertexPositionEdit.metal"},
    PassFamily{MeshPass::EditFaceUvs, MeshPass::EditFaceColors, "FaceAttributeEdit.metal"},
    PassFamily{MeshPass::AttributeRead, MeshPass::AttributeSelectionMasks, "AttributeKernel.metal"},
    PassFamily{MeshPass::MeshletBoundsRefit, MeshPass::MeshletBoundsRefit, "MeshletBoundsRefit.metal"},
    PassFamily{MeshPass::TopologyGatherCorners, MeshPass::TopologyEdgeAttributes, "MeshTopology.metal"},
    PassFamily{MeshPass::LodNodeRefit, MeshPass::LodNodeRefit, "LodNodeRefit.metal"},
    PassFamily{MeshPass::MeshletOwners, MeshPass::MeshletOwners, "MeshletOwners.metal"},
    PassFamily{MeshPass::MeshletBuildElements, MeshPass::MeshletBuildPrimitives, "MeshletBuild.metal"},
    PassFamily{MeshPass::CloneCopyByteRuns, MeshPass::CloneOffsetHandles, "MeshClone.metal"},
};
// Every pass belongs to one family, which this fails to compile without.
constexpr const PassFamily &FamilyOf(MeshPass pass) {
    for (const auto &family : Families) {
        if (pass >= family.First && pass <= family.Last) return family;
    }
    return Families[Families.size()];
}
template<size_t I> constexpr auto KernelName = [] {
    constexpr auto &family = FamilyOf(MeshPass(I));
    constexpr std::string_view prefix{family.Prefix}, name = field::detail::EnumeratorName<MeshPass(I)>(), suffix{family.Suffix};
    field::fixed_string<prefix.size() + name.size() + suffix.size() + 1> out{};
    size_t at = 0;
    for (const auto part : {prefix, name, suffix}) {
        for (const auto c : part) out.Chars[at++] = c;
    }
    return out;
}();
// The shader file and function of each pass, in MeshPass order.
constexpr auto PassFunctions = []<size_t... Is>(std::index_sequence<Is...>) {
    return std::array<std::pair<const char *, const char *>, sizeof...(Is)>{std::pair{FamilyOf(MeshPass(Is)).File, KernelName<Is>.c_str()}...};
}(std::make_index_sequence<size_t(MeshPass::Count)>{});
} // namespace

MeshPipelines::MeshPipelines(mtl::LibraryCache &libraries) : Libraries(libraries) {}

void MeshPipelines::PrewarmAsync() {
    if (PrewarmWorker.joinable()) return;
    PrewarmWorker = std::jthread([this](const std::stop_token &stop) {
        try {
            auto cache = Libraries.PrewarmCache();
            for (size_t i = 0u; i < PassFunctions.size() && !stop.stop_requested(); ++i) {
                {
                    const std::scoped_lock lock{Mutex};
                    if (Pipelines[i]) continue;
                }
                try {
                    mtl::ComputePipeline pipeline{*cache, {PassFunctions[i].first, PassFunctions[i].second}};
                    const std::scoped_lock lock{Mutex};
                    if (!Pipelines[i]) Pipelines[i].emplace(std::move(pipeline));
                } catch (...) {
                    // Foreground use retains the normal error path for a shader edited after the offline archive was built.
                }
            }
        } catch (...) {
            // A missing archive does not change foreground shader behavior.
        }
    });
}

const mtl::ComputePipeline &MeshPipelines::operator[](MeshPass pass) const {
    const auto index = size_t(pass);
    const std::scoped_lock lock{Mutex};
    auto &pipeline = Pipelines.at(index);
    if (!pipeline) {
        const profile::CpuScope scope{"MeshPipelineCreate"};
        pipeline.emplace(Libraries, mtl::FunctionRef{PassFunctions[index].first, PassFunctions[index].second});
    }
    return *pipeline;
}

MeshPipelines &GetMeshPipelines(state::Registry &r) {
    if (auto *pipelines = r.Resources.find<MeshPipelines>()) return *pipelines;
    return r.Resources.emplace<MeshPipelines>(r.Resources.get<mtl::LibraryCache>());
}
