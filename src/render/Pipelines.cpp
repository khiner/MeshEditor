#include "render/Pipelines.h"
#include "gpu/BackgroundConstant.h"
#include "gpu/EditOverlayConstant.h"
#include "gpu/MeshVertexConstant.h"
#include "gpu/NormalIndicatorConstant.h"
#include "gpu/PbrConstant.h"
#include "state/Scene.h"

#include <array>

using mtl::AdditiveBlend, mtl::Blend, mtl::NoBlend, mtl::NoWrite, mtl::PremultipliedBlend;
using mtl::BlendState, mtl::DepthState, mtl::FunctionConstant, mtl::FunctionRef, mtl::PassFormats, mtl::RenderPipeline;

namespace {
// Map host feature bits to generated shader function constants.
constexpr std::array PbrSpecFeatures{
    std::pair{PbrConstant::EnablePunctual, PbrFeature::Punctual},
    std::pair{PbrConstant::EnableTransmission, PbrFeature::Transmission},
    std::pair{PbrConstant::EnableDiffuseTrans, PbrFeature::DiffuseTrans},
    std::pair{PbrConstant::EnableClearcoat, PbrFeature::Clearcoat},
    std::pair{PbrConstant::EnableSheen, PbrFeature::Sheen},
    std::pair{PbrConstant::EnableAnisotropy, PbrFeature::Anisotropy},
    std::pair{PbrConstant::EnableIridescence, PbrFeature::Iridescence},
};

constexpr DepthState DepthTestWrite{};
constexpr DepthState DepthOff{.Test = false, .Write = false};
constexpr DepthState DepthTestLessEqual{.Compare = MTL::CompareFunctionLessEqual};
constexpr DepthState DepthTestNoWriteLessEqual{.Write = false, .Compare = MTL::CompareFunctionLessEqual};

constexpr mtl::FunctionConstant BoolConstant(auto index, bool value) {
    return {uint32_t(index), MTL::DataTypeBool, value ? 1u : 0u};
}
std::vector<mtl::FunctionConstant> MeshVertexConstants(bool non_triangle_topology = false) {
    return {
        BoolConstant(MeshVertexConstant::NonTriangleTopology, non_triangle_topology),
    };
}
FunctionRef NormalIndicatorMesh(bool faces) {
    return {"NormalIndicator.metal", "NormalIndicatorMesh", {BoolConstant(NormalIndicatorConstant::NormalIndicatorFaces, faces)}};
}

FunctionRef MeshletVertex(bool non_triangle_topology = false) {
    return {
        "MeshletTransform.metal", "MeshletForwardMesh",
        MeshVertexConstants(non_triangle_topology)
    };
}

FunctionRef MeshletVisibilityVertex() {
    return {"MeshletTransform.metal", "MeshletVisibilityMesh", MeshVertexConstants(false)};
}

RenderPipeline CreateMeshPipeline(
    mtl::LibraryCache &libraries, std::optional<FunctionRef> fragment, PassFormats formats,
    std::vector<BlendState> blends = {}, std::optional<DepthState> depth = {}, FunctionRef mesh = MeshletVertex()
) {
    return mtl::MakeMeshPipeline(libraries, std::move(mesh), std::move(fragment), std::move(formats), std::move(blends), depth);
}

FunctionRef PbrFragment(PbrFeatureMask mask, bool prepass, const char *fragment, bool non_triangle_topology) {
    std::vector<FunctionConstant> constants;
    constants.reserve(PbrSpecFeatures.size() + 2);
    for (const auto &[constant, feature] : PbrSpecFeatures) constants.push_back(BoolConstant(constant, HasFeature(mask, feature)));
    constants.push_back(BoolConstant(PbrConstant::TransmissionPrepass, prepass));
    constants.push_back(BoolConstant(PbrConstant::NonTriangleTopology, non_triangle_topology));
    return {"pbr.metal", fragment, std::move(constants)};
}

PassFormats SceneFormats() { return {{Format::HdrColor}, Format::Depth}; }
PassFormats OverlayFormats() { return {{Format::Color}, Format::Depth}; }

RenderPipeline StrokePipeline(mtl::LibraryCache &libraries, FunctionRef mesh, bool include_outer = false) {
    return CreateMeshPipeline(
        libraries, FunctionRef{"EdgeQuad.metal", "EdgeQuadFragment", {BoolConstant(EditOverlayConstant::IncludeOuter, include_outer)}}, OverlayFormats(), {PremultipliedBlend}, DepthTestNoWriteLessEqual,
        std::move(mesh)
    );
}

// `transmission_prepass` skips exposure, which the main pass applies after sampling.
RenderPipeline CreateBackgroundPipeline(mtl::LibraryCache &libraries, PassFormats formats, std::vector<mtl::BlendState> blends, bool transmission_prepass) {
    return mtl::MakeRenderPipeline(
        libraries, {"Background.metal", "BackgroundVertex"}, FunctionRef{"Background.metal", "BackgroundFragment", {BoolConstant(BackgroundConstant::TransmissionPrepass, transmission_prepass)}}, std::move(formats), std::move(blends), DepthOff
    );
}

RenderPipeline CreateQuadPipeline(mtl::LibraryCache &libraries, PassFormats formats, const char *fragment_file, const char *fragment_name, mtl::BlendState blend) {
    return mtl::MakeRenderPipeline(libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{fragment_file, fragment_name}, std::move(formats), {blend}, DepthOff);
}

// The element rasters share these formats: depth only, and no color, since ids reach the fragment stage as a varying.
PassFormats SelectionFormats() { return {{}, Format::Depth}; }

RenderPipeline MeshletElementRaster(mtl::LibraryCache &libraries, FunctionRef mesh, bool bitset_box, bool xray) {
    const FunctionRef fragment = bitset_box ?
        FunctionRef{"SelectionElementBitsetBox.metal", "SelectionElementBitsetBoxFragment"} :
        FunctionRef{"SelectionElementPick.metal", "SelectionElementPickFragment"};
    return CreateMeshPipeline(libraries, fragment, SelectionFormats(), {}, xray ? DepthOff : DepthTestNoWriteLessEqual, std::move(mesh));
}
} // namespace

PbrCompiler::PbrCompiler(mtl::LibraryCache &libraries, PassFormats scene) : Libraries(libraries), SceneFormats(std::move(scene)) {}

bool PbrCompiler::CompilePipelines(PbrFeatureMask mask, bool non_triangle_topology) {
    if (mask == Mask && non_triangle_topology == NonTriangleTopology && Visibility) return false;
    const auto visibility = [&](bool prepass) {
        return mtl::MakeRenderPipeline(Libraries, FunctionRef{"TexQuad.metal", "TexQuadVertex"}, PbrFragment(mask, prepass, "PbrVisibilityFragment", non_triangle_topology), SceneFormats, std::vector<BlendState>{Blend}, DepthOff);
    };
    Visibility = visibility(false);
    Prepass = ::HasFeature(mask, PbrFeature::Transmission) ? std::optional{visibility(true)} : std::nullopt;
    Transparent = mtl::MakeMeshPipeline(Libraries, MeshletVertex(non_triangle_topology), PbrFragment(mask, false, "PbrTransparentFragment", non_triangle_topology), SceneFormats, std::vector<BlendState>{NoWrite}, DepthTestNoWriteLessEqual);
    Mask = mask;
    NonTriangleTopology = non_triangle_topology;
    return true;
}

void PbrCompiler::BindMeshlets(MTL::RenderCommandEncoder *encoder) const { Transparent->Bind(encoder); }
void PbrCompiler::BindVisibility(MTL::RenderCommandEncoder *encoder, bool prepass) const { (prepass ? Prepass : Visibility)->Bind(encoder); }

MainPipeline::MainPipeline(mtl::LibraryCache &libraries)
    : Background{CreateBackgroundPipeline(libraries, SceneFormats(), {Blend}, false)},
      TransmissionComposite{CreateQuadPipeline(libraries, SceneFormats(), "TransmissionComposite.metal", "TransmissionCompositeFragment", PremultipliedBlend)},
      MotionBlurResolve{CreateQuadPipeline(libraries, SceneFormats(), "MotionBlurResolve.metal", "MotionBlurResolveFragment", NoBlend)},
      Grid{mtl::MakeRenderPipeline(libraries, {"GridLines.metal", "GridLinesVertex"}, FunctionRef{"GridLines.metal", "GridLinesFragment"}, OverlayFormats(), {Blend}, DepthState{.Write = false})},
      SilhouetteEdgeColor{mtl::MakeRenderPipeline(libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"SilhouetteEdgeColor.metal", "SilhouetteEdgeColorFragment"}, OverlayFormats(), {NoBlend}, DepthState{.Test = false})},
      PrepassBackground{CreateBackgroundPipeline(libraries, SceneFormats(), {Blend}, true)},
      ViewportComposite{CreateQuadPipeline(libraries, {{Format::Color}, MTL::PixelFormatInvalid}, "ViewportComposite.metal", "ViewportCompositeFragment", NoBlend)},
      MotionBlurAccumulate{CreateQuadPipeline(libraries, {{MTL::PixelFormatRGBA32Float}, MTL::PixelFormatInvalid}, "MotionBlurAccumulate.metal", "MotionBlurAccumulateFragment", AdditiveBlend)},
      MotionBlurGather{CreateQuadPipeline(libraries, {{Format::HdrColor}, MTL::PixelFormatInvalid}, "MotionBlurGather.metal", "MotionBlurGatherFragment", NoBlend)},
      MotionBlurTilesFlatten{libraries, {"MotionBlurTilesFlatten.metal", "MotionBlurTilesFlattenKernel"}},
      MotionBlurTilesDilate{libraries, {"MotionBlurTilesDilate.metal", "MotionBlurTilesDilateKernel"}},
      WorkspaceVisibility{mtl::MakeRenderPipeline(libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"WorkspaceLighting.metal", "WorkspaceVisibilityFragment"}, SceneFormats(), {Blend}, DepthOff)},
      TransparencyInit{CreateQuadPipeline(libraries, SceneFormats(), "Transparency.metal", "TransparencyInitFragment", NoWrite)},
      TransparencyResolve{CreateQuadPipeline(libraries, SceneFormats(), "Transparency.metal", "TransparencyResolveFragment", NoBlend)},
      MeshletVisibilityOpaque{CreateMeshPipeline(libraries, FunctionRef{"MeshletVisibility.metal", "MeshletVisibilityOpaqueFragment"}, {{Format::Uint}, Format::Depth}, {NoBlend}, DepthTestWrite, MeshletVisibilityVertex())},
      MeshletVisibilityCoverage{CreateMeshPipeline(libraries, FunctionRef{"MeshletVisibility.metal", "MeshletVisibilityPrimitiveFragment"}, {{Format::Uint}, Format::Depth}, {NoBlend}, DepthTestWrite, MeshletVisibilityVertex())},
      MeshletEditEdges{StrokePipeline(libraries, {"MeshletEditOverlay.metal", "MeshletEditEdgeMesh"}, true)},
      MeshletEditSmoothEdges{StrokePipeline(libraries, {"MeshletEditOverlay.metal", "MeshletEditEdgeMesh"})},
      MeshletEditPoint{CreateMeshPipeline(libraries, FunctionRef{"VertexPoint.metal", "VertexPointFragment"}, OverlayFormats(), {Blend}, DepthTestNoWriteLessEqual, {"MeshletEditOverlay.metal", "MeshletEditPointMesh"})},
      FaceNormalMesh{StrokePipeline(libraries, NormalIndicatorMesh(true))},
      VertexNormalMesh{StrokePipeline(libraries, NormalIndicatorMesh(false))},
      OverlayJobLines{StrokePipeline(libraries, {"OverlayJobLine.metal", "OverlayJobLineMesh"})},
      BoneFillMesh{CreateMeshPipeline(libraries, FunctionRef{"BoneSolid.metal", "BoneSolidFragment"}, OverlayFormats(), {Blend}, DepthTestWrite, {"BoneSolid.metal", "BoneSolidMesh"})},
      BoneWireMesh{StrokePipeline(libraries, {"BoneWire.metal", "BoneWireMesh"})},
      BoneSphereFillMesh{CreateMeshPipeline(libraries, FunctionRef{"BoneSphere.metal", "BoneSphereFragment"}, OverlayFormats(), {Blend}, DepthTestLessEqual, {"BoneSphere.metal", "BoneSphereMesh"})},
      BoneSphereWireMesh{StrokePipeline(libraries, {"BoneSphereWire.metal", "BoneSphereWireMesh"})},
      WireResolve{mtl::MakeRenderPipeline(libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"WireResolve.metal", "WireResolveFragment"}, OverlayFormats(), {PremultipliedBlend}, DepthOff)},
      Compiler{libraries, SceneFormats()} {}

SelectionFragmentPipeline::SelectionFragmentPipeline(mtl::LibraryCache &libraries)
    : MeshletFaces{
          MeshletElementRaster(libraries, MeshletVertex(), false, false),
          MeshletElementRaster(libraries, MeshletVertex(), true, false),
          CreateMeshPipeline(libraries, FunctionRef{"SelectionElementPick.metal", "SelectionElementPickFragment"}, SelectionFormats(), {}, DepthOff),
          CreateMeshPipeline(libraries, FunctionRef{"SelectionElementBitsetBox.metal", "SelectionElementBitsetBoxFragment"}, SelectionFormats(), {}, DepthOff),
      },
      MeshletVertices{
          MeshletElementRaster(libraries, {"MeshletEditOverlay.metal", "MeshletSelectPointMesh"}, false, false),
          MeshletElementRaster(libraries, {"MeshletEditOverlay.metal", "MeshletSelectPointMesh"}, true, false),
          MeshletElementRaster(libraries, {"MeshletEditOverlay.metal", "MeshletSelectPointMesh"}, false, true),
          MeshletElementRaster(libraries, {"MeshletEditOverlay.metal", "MeshletSelectPointMesh"}, true, true),
      },
      MeshletEdges{
          MeshletElementRaster(libraries, {"MeshletEditOverlay.metal", "MeshletSelectEdgeMesh"}, false, false),
          MeshletElementRaster(libraries, {"MeshletEditOverlay.metal", "MeshletSelectEdgeMesh"}, true, false),
          MeshletElementRaster(libraries, {"MeshletEditOverlay.metal", "MeshletSelectEdgeMesh"}, false, true),
          MeshletElementRaster(libraries, {"MeshletEditOverlay.metal", "MeshletSelectEdgeMesh"}, true, true),
      },
      MeshletFaceXRayPointsBitsetBox{MeshletElementRaster(libraries, {"MeshletEditOverlay.metal", "MeshletSelectFacePointMesh"}, true, true)},
      MeshletEdgeXRayPointsBitsetBox{MeshletElementRaster(libraries, {"MeshletEditOverlay.metal", "MeshletSelectEdgePointMesh"}, true, true)},
      ObjectPick{CreateMeshPipeline(libraries, FunctionRef{"VisibilitySelection.metal", "MeshletObjectPickFragment"}, SelectionFormats(), {}, DepthOff, MeshletVisibilityVertex())},
      OverlayJobLines{CreateMeshPipeline(libraries, FunctionRef{"SelectionFragment.metal", "SelectionStrokeFragment"}, SelectionFormats(), {}, DepthOff, {"OverlayJobLine.metal", "OverlayJobLineMesh"})},
      BoneSolid{CreateMeshPipeline(libraries, FunctionRef{"SelectionFragment.metal", "SelectionFragment"}, SelectionFormats(), {}, DepthOff, {"BoneSolid.metal", "BoneSolidMesh"})},
      BoneSphere{CreateMeshPipeline(libraries, FunctionRef{"SelectionFragment.metal", "SelectionFragment"}, SelectionFormats(), {}, DepthOff, {"BoneSphere.metal", "BoneSphereMesh"})} {}

const RenderPipeline &SelectionFragmentPipeline::ElementRaster(Element element, bool bitset_box, bool xray) const {
    const auto &variants = element == Element::Face ? MeshletFaces : element == Element::Vertex ? MeshletVertices :
                                                                                                  MeshletEdges;
    return variants[uint32_t(bitset_box) + 2u * uint32_t(xray)];
}

Pipelines::Pipelines(mtl::LibraryCache &libraries)
    : Main{libraries},
      Silhouette{CreateMeshPipeline(libraries, FunctionRef{"VisibilitySelection.metal", "MeshletSilhouetteFragment"}, PassFormats{{Format::Float2}, Format::Depth}, {NoBlend}, DepthTestNoWriteLessEqual, MeshletVisibilityVertex())},
      SelectionFragment{libraries},
      VisibilityObjectSelection{libraries, {"VisibilitySelection.metal", "VisibilityObjectSelectionKernel"}},
      PrepareEditSelection{libraries, {"EditSelectionTransaction.metal", "PrepareEditSelectionKernel"}},
      FillEditSelectionList{libraries, {"EditSelectionTransaction.metal", "FillEditSelectionListKernel"}},
      ResetEditSelectionSummary{libraries, {"EditSelectionTransaction.metal", "ResetEditSelectionSummaryKernel"}},
      DeriveEditSelection{libraries, {"EditSelectionTransaction.metal", "DeriveEditSelectionKernel"}},
      SumEditSelectionPosition{libraries, {"EditSelectionTransaction.metal", "SumEditSelectionPositionKernel"}},
      EditSharpness{libraries, {"EditSharpness.metal", "EditSharpnessKernel"}},
      CommitPosedGeometry{libraries, {"CommitPosedGeometry.metal", "CommitPosedGeometryKernel"}},
      GeometryWorkArgs{libraries, {"CommitPosedGeometry.metal", "GeometryWorkArgsKernel"}},
      PosePrepass{libraries, {"PosePrepass.metal", "PosePrepassKernel"}},
      PosedMeshletBounds{libraries, {"PosedMeshletBounds.metal", "PosedMeshletBoundsKernel"}},
      VertexNormalDerive{libraries, {"VertexNormalDerive.metal", "VertexNormalDeriveKernel"}},
      BoundsReduce{libraries, {"BoundsReduce.metal", "BoundsReduceKernel"}},
      BoundsCombine{libraries, {"BoundsCombine.metal", "BoundsCombineKernel"}},
      BoundsTree{libraries, {"BoundsTree.metal", "BoundsTreeKernel"}},
      WireRaster{libraries, {"WireRaster.metal", "WireRasterKernel"}},
      LodFrontierCount{libraries, {"MeshletCull.metal", "LodFrontierCount"}},
      LodFrontierPrefix{libraries, {"MeshletCull.metal", "LodFrontierPrefix"}},
      LodFrontierEmit{libraries, {"MeshletCull.metal", "LodFrontierEmit"}},
      MeshletCullBlockCount{libraries, {"MeshletCull.metal", "MeshletCullBlockCount"}},
      MeshletCullPrefix{libraries, {"MeshletCull.metal", "MeshletCullPrefix"}},
      MeshletCullEmit{libraries, {"MeshletCull.metal", "MeshletCullEmit"}},
      OverlayJobBlockCount{libraries, {"OverlayJobCull.metal", "OverlayJobBlockCount"}},
      OverlayJobPrefix{libraries, {"OverlayJobCull.metal", "OverlayJobPrefix"}},
      OverlayJobEmit{libraries, {"OverlayJobCull.metal", "OverlayJobEmit"}},
      DepthPyramidReduce{libraries, {"DepthPyramidReduce.metal", "DepthPyramidReduceKernel"}},
      EquirectToCubemap{libraries, {"IblPrefilter.metal", "EquirectToCubemapKernel"}},
      DiffuseIrradiance{libraries, {"IblPrefilter.metal", "DiffuseIrradianceKernel"}},
      SpecularPrefilter{libraries, {"IblPrefilter.metal", "SpecularPrefilterKernel"}} {}

Pipelines &GetPipelines(state::Scene &r) {
    if (auto *pipelines = r.ctx().find<Pipelines>()) return *pipelines;
    return r.ctx().emplace<Pipelines>(r.ctx().get<mtl::LibraryCache>());
}
