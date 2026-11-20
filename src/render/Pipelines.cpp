#include "render/Pipelines.h"
#include "gpu/BackgroundConstant.h"
#include "gpu/EditOverlayConstant.h"
#include "gpu/MeshVertexConstant.h"
#include "state/Registry.h"

using mtl::AdditiveBlend, mtl::Blend, mtl::NoBlend, mtl::NoWrite, mtl::PremultipliedBlend;
using mtl::BlendState, mtl::DepthState, mtl::FunctionConstant, mtl::FunctionRef, mtl::PassFormats, mtl::RenderPipeline;

namespace {
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
    mtl::LibraryCache &libraries, std::optional<FunctionRef> fragment, const PassFormats &formats,
    const std::vector<BlendState> &blends = {}, std::optional<DepthState> depth = {}, FunctionRef mesh = MeshletVertex()
) {
    return mtl::MakeMeshPipeline(libraries, std::move(mesh), std::move(fragment), formats, blends, depth);
}

FunctionRef PbrFunction(const char *name, PbrFeatureMask mask, const char *library = "pbr.metal") {
    FunctionRef function{library, name};
    for (uint32_t index = 0; index < uint32_t(PbrFeature::BoundedAreaShadows); ++index)
        function.Constants.push_back(BoolConstant(index, HasFeature(mask, PbrFeature(index))));
    return function;
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
RenderPipeline CreateBackgroundPipeline(mtl::LibraryCache &libraries, const PassFormats &formats, const std::vector<mtl::BlendState> &blends, bool transmission_prepass) {
    return mtl::MakeRenderPipeline(
        libraries, {"Background.metal", "BackgroundVertex"}, FunctionRef{"Background.metal", "BackgroundFragment", {BoolConstant(BackgroundConstant::TransmissionPrepass, transmission_prepass)}}, formats, blends, DepthOff
    );
}

RenderPipeline CreateQuadPipeline(mtl::LibraryCache &libraries, const PassFormats &formats, const char *fragment_file, const char *fragment_name, mtl::BlendState blend) {
    return mtl::MakeRenderPipeline(libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{fragment_file, fragment_name}, formats, {blend}, DepthOff);
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

const PbrVariant &PbrCompiler::Variant(PbrFeatureMask mask) {
    const auto it = std::ranges::find(Variants, mask, &PbrVariant::Mask);
    if (it != Variants.end()) return *it;
    const auto raster = mask & ~(FeatureBit(PbrFeature::TraceDiffuse) | PbrFeature::TraceReflections);
    const auto visibility = [&](PbrFeatureMask features) {
        return mtl::MakeRenderPipeline(Libraries, FunctionRef{"TexQuad.metal", "TexQuadVertex"}, PbrFunction("PbrVisibilityFragment", features), SceneFormats, std::vector<BlendState>{Blend}, DepthOff);
    };
    const bool transmission = HasFeature(mask, PbrFeature::Transmission);
    PbrVariant variant{
        .Mask = mask,
        .Transparent = mtl::MakeMeshPipeline(Libraries, MeshletVertex(HasFeature(mask, PbrFeature::NonTriangleTopology)), PbrFunction("PbrTransparentFragment", raster), SceneFormats, std::vector<BlendState>{NoWrite}, DepthTestNoWriteLessEqual),
        .Visibility = visibility(raster),
        .Prepass = transmission ? std::optional{visibility(raster | PbrFeature::TransmissionPrepass)} : std::nullopt,
    };
    if (HasFeature(mask, PbrFeature::RayTracing)) {
        const auto triangles = mask & ~FeatureBit(PbrFeature::NonTriangleTopology);
        const auto kernel = [&](const char *name, PbrFeatureMask features, const char *library = "RayLighting.metal") {
            return mtl::ComputePipeline{Libraries, PbrFunction(name, features, library)};
        };
        variant.RayDirect = kernel("PbrRayDirect", mask, HasFeature(mask, PbrFeature::BoundedAreaShadows) ? "RayDirectBounded.metal" : "RayLighting.metal");
        if (HasFeature(mask, PbrFeature::AreaLights) && !HasFeature(mask, PbrFeature::MultipleShadowSamples) && !transmission && !HasFeature(mask, PbrFeature::AlphaBlend)) {
            variant.RayDirectFiltered = kernel("PbrRayDirect", mask, "RayDirectFiltered.metal");
        }
        if (transmission) {
            variant.PreparePrimary = kernel("PbrRayDirect", mask | PbrFeature::PreparePrimary);
            variant.GlassWeights = kernel("PbrRayGlassWeights", triangles);
            if (HasFeature(mask, PbrFeature::OpaqueReceiver)) {
                variant.ShadeOpaque = kernel("PbrRayShade", triangles);
                variant.OpaqueLayer = kernel("PbrOpaqueLayer", mask);
            }
        }
        if ((transmission && HasFeature(mask, PbrFeature::AlphaBlend)) || (mask & ExtendedPbrMaterials)) variant.Special = kernel("PbrRaySpecial", triangles);
        variant.Generate = kernel("PbrRayGenerate", triangles);
        variant.Shade = kernel("PbrRayShade", triangles & ~FeatureBit(PbrFeature::OpaqueReceiver));
        variant.Reconstruct = kernel("RayReconstruct", triangles);
    }
    return Variants.emplace_back(std::move(variant));
}

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
      FastGiDepth{libraries, {"FastGI.metal", "FastGiDepth"}},
      FastGiDepthReduce{libraries, {"FastGI.metal", "FastGiDepthReduce"}},
      FastGiSetup{libraries, {"FastGI.metal", "FastGiSetup"}},
      FastGiReduce{libraries, {"FastGI.metal", "FastGiReduce"}},
      FastGiScan{libraries, {"FastGI.metal", "FastGiScan"}},
      FastGiDenoise{libraries, {"FastGI.metal", "FastGiDenoise"}},
      FastGiResolve{libraries, {"FastGI.metal", "FastGiResolve"}},
      RayCameraMedia{libraries, {"RayCameraMedia.metal", "RayCameraMedia"}},
      RayDeformation{libraries, {"RayDeformation.metal", "RayDeformation"}},
      RayTemporal{libraries, {"RayTemporal.metal", "RayTemporal"}},
      LightCull{libraries, {"LightCulling.metal", "LightCull"}},
      LightPrepare{libraries, {"LightCulling.metal", "LightPrepare"}},
      RayShadowBounds{libraries, {"RayShadowBounds.metal", "PrepareRayShadowBounds"}},
      RayBilateral{libraries, {"RayTemporal.metal", "RayBilateral"}},
      RayAccumulate{libraries, {"RayTemporal.metal", "RayAccumulate"}},
      RayFilmReproject{libraries, {"RayTemporal.metal", "RayFilmReproject", {BoolConstant(0u, false)}}},
      RayFilmReprojectLayer{libraries, {"RayTemporal.metal", "RayFilmReproject", {BoolConstant(0u, true)}}},
      RayEnvironmentRows{libraries, {"RayEnvironment.metal", "RayEnvironmentRowsKernel"}},
      RayEnvironmentMarginal{libraries, {"RayEnvironment.metal", "RayEnvironmentMarginalKernel"}},
      RayClassify{libraries, {"RayClassify.metal", "RayClassify"}},
      RayDispatch{libraries, {"RayClassify.metal", "RayDispatch"}},
      RayReset{libraries, {"RayClassify.metal", "RayReset"}},
      MotionBlurTilesFlatten{libraries, {"MotionBlurTilesFlatten.metal", "MotionBlurTilesFlattenKernel"}},
      MotionBlurTilesDilate{libraries, {"MotionBlurTilesDilate.metal", "MotionBlurTilesDilateKernel"}},
      WorkspaceVisibility{mtl::MakeRenderPipeline(libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"WorkspaceLighting.metal", "WorkspaceVisibilityFragment"}, SceneFormats(), {Blend}, DepthOff)},
      WorkspaceTransparent{CreateMeshPipeline(libraries, FunctionRef{"WorkspaceLighting.metal", "WorkspaceTransparentFragment"}, SceneFormats(), {NoWrite}, DepthOff)},
      TransparencyInit{CreateQuadPipeline(libraries, SceneFormats(), "Transparency.metal", "TransparencyInitFragment", NoWrite)},
      TransparencyResolve{CreateQuadPipeline(libraries, SceneFormats(), "Transparency.metal", "TransparencyResolveFragment", NoBlend)},
      MeshletVisibilityOpaque{CreateMeshPipeline(libraries, FunctionRef{"MeshletVisibility.metal", "MeshletVisibilityOpaqueFragment"}, {{Format::Uint}, Format::Depth}, {NoBlend}, DepthTestWrite, MeshletVisibilityVertex())},
      MeshletVisibilityCoverage{CreateMeshPipeline(libraries, FunctionRef{"MeshletVisibility.metal", "MeshletVisibilityPrimitiveFragment"}, {{Format::Uint}, Format::Depth}, {NoBlend}, DepthTestWrite, MeshletVisibilityVertex())},
      MeshletVisibilityLayer{CreateMeshPipeline(libraries, FunctionRef{"MeshletVisibility.metal", "MeshletVisibilityLayerFragment", {BoolConstant(0u, false)}}, {{Format::Uint}, Format::Depth}, {NoBlend}, DepthTestWrite, MeshletVisibilityVertex())},
      MeshletVisibilityMotion{CreateMeshPipeline(libraries, FunctionRef{"MeshletVisibility.metal", "MeshletVisibilityLayerFragment", {BoolConstant(0u, true)}}, {{Format::Uint}, Format::Depth}, {NoBlend}, DepthTestWrite, MeshletVisibilityVertex())},
      MeshletEditEdges{StrokePipeline(libraries, {"MeshletEditOverlay.metal", "MeshletEditEdgeMesh"}, true)},
      MeshletEditSmoothEdges{StrokePipeline(libraries, {"MeshletEditOverlay.metal", "MeshletEditEdgeMesh"})},
      VertexBlockPoints{CreateMeshPipeline(libraries, FunctionRef{"VertexPoint.metal", "VertexPointFragment"}, OverlayFormats(), {Blend}, DepthTestNoWriteLessEqual, {"VertexBlockOverlay.metal", "VertexBlockPointMesh"})},
      FaceNormalMesh{StrokePipeline(libraries, {"NormalIndicator.metal", "FaceNormalIndicatorMesh"})},
      VertexNormalMesh{StrokePipeline(libraries, {"NormalIndicator.metal", "VertexNormalIndicatorMesh"})},
      OverlayJobLines{StrokePipeline(libraries, {"OverlayJobLine.metal", "OverlayJobLineMesh"})},
      BoneFillMesh{CreateMeshPipeline(libraries, FunctionRef{"BoneSolid.metal", "BoneSolidFragment"}, OverlayFormats(), {Blend}, DepthTestWrite, {"BoneSolid.metal", "BoneSolidMesh"})},
      BoneWireMesh{StrokePipeline(libraries, {"BoneWire.metal", "BoneWireMesh"})},
      BoneSphereFillMesh{CreateMeshPipeline(libraries, FunctionRef{"BoneSphere.metal", "BoneSphereFragment"}, OverlayFormats(), {Blend}, DepthTestLessEqual, {"BoneSphere.metal", "BoneSphereMesh"})},
      BoneSphereWireMesh{StrokePipeline(libraries, {"BoneSphereWire.metal", "BoneSphereWireMesh"})},
      WireResolve{mtl::MakeRenderPipeline(libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"WireResolve.metal", "WireResolveFragment"}, OverlayFormats(), {PremultipliedBlend}, DepthOff)},
      ObjectOrigins{mtl::MakeRenderPipeline(libraries, {"ObjectOrigin.metal", "ObjectOriginVertex"}, FunctionRef{"ObjectOrigin.metal", "ObjectOriginFragment"}, OverlayFormats(), {PremultipliedBlend}, DepthOff)},
      Compiler{libraries, SceneFormats()} {}

SelectionFragmentPipeline::SelectionFragmentPipeline(mtl::LibraryCache &libraries)
    : MeshletFaces{
          CreateMeshPipeline(libraries, FunctionRef{"SelectionElementPick.metal", "SelectionFacePickFragment"}, SelectionFormats(), {}, DepthTestNoWriteLessEqual, {"MeshletEditOverlay.metal", "MeshletSelectFaceMesh"}),
          CreateMeshPipeline(libraries, FunctionRef{"SelectionElementBitsetBox.metal", "SelectionFaceBitsetBoxFragment"}, SelectionFormats(), {}, DepthTestNoWriteLessEqual, {"MeshletEditOverlay.metal", "MeshletSelectFaceMesh"}),
          CreateMeshPipeline(libraries, FunctionRef{"SelectionElementPick.metal", "SelectionElementPickFragment"}, SelectionFormats(), {}, DepthOff),
          CreateMeshPipeline(libraries, FunctionRef{"SelectionElementBitsetBox.metal", "SelectionElementBitsetBoxFragment"}, SelectionFormats(), {}, DepthOff),
      },
      VertexBlocks{
          MeshletElementRaster(libraries, {"VertexBlockOverlay.metal", "VertexBlockSelectMesh"}, false, false),
          MeshletElementRaster(libraries, {"VertexBlockOverlay.metal", "VertexBlockSelectMesh"}, true, false),
          MeshletElementRaster(libraries, {"VertexBlockOverlay.metal", "VertexBlockSelectMesh"}, false, true),
          MeshletElementRaster(libraries, {"VertexBlockOverlay.metal", "VertexBlockSelectMesh"}, true, true),
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
    const auto &variants = element == Element::Face ? MeshletFaces : element == Element::Vertex ? VertexBlocks :
                                                                                                  MeshletEdges;
    return variants[uint32_t(bitset_box) + 2u * uint32_t(xray)];
}

Pipelines::Pipelines(mtl::LibraryCache &libraries)
    : Main{libraries},
      SilhouetteSeed{mtl::MakeRenderPipeline(libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"VisibilitySelection.metal", "SilhouetteSeedFragment"}, PassFormats{{Format::Float2}, Format::Depth}, {NoBlend}, DepthState{.Test = false})},
      Silhouette{CreateMeshPipeline(libraries, FunctionRef{"VisibilitySelection.metal", "MeshletSilhouetteFragment"}, PassFormats{{Format::Float2}, Format::Depth}, {NoBlend}, DepthTestWrite, MeshletVisibilityVertex())},
      SelectionFragment{libraries},
      VisibilityObjectSelection{libraries, {"VisibilitySelection.metal", "VisibilityObjectSelectionKernel"}},
      OutlineOccluderSeed{libraries, {"VisibilitySelection.metal", "OutlineOccluderSeedKernel"}},
      CommitPosedGeometry{libraries, {"CommitPosedGeometry.metal", "CommitPosedGeometryKernel"}},
      FinalizeElementWork{libraries, {"CommitPosedGeometry.metal", "FinalizeElementWorkKernel"}},
      PosePrepass{libraries, {"PosePrepass.metal", "PosePrepassKernel"}},
      PosedMeshletBounds{libraries, {"PosedMeshletBounds.metal", "PosedMeshletBoundsKernel"}},
      BoundsReduce{libraries, {"BoundsReduce.metal", "BoundsReduceKernel"}},
      BoundsCombine{libraries, {"BoundsCombine.metal", "BoundsCombineKernel"}},
      WireRaster{libraries, {"WireRaster.metal", "WireRasterKernel"}},
      LodFrontierCount{libraries, {"MeshletCull.metal", "LodFrontierCount"}},
      LodFrontierPrefix{libraries, {"MeshletCull.metal", "LodFrontierPrefix"}},
      LodFrontierEmit{libraries, {"MeshletCull.metal", "LodFrontierEmit"}},
      MeshletCullBlockCount{libraries, {"MeshletCull.metal", "MeshletCullBlockCount"}},
      MeshletCullPrefix{libraries, {"MeshletCull.metal", "MeshletCullPrefix"}},
      MeshletCullEmit{libraries, {"MeshletCull.metal", "MeshletCullEmit"}},
      SilhouetteCullSize{libraries, {"MeshletCull.metal", "SilhouetteCullSize"}},
      OverlayJobBlockCount{libraries, {"OverlayJobCull.metal", "OverlayJobBlockCount"}},
      OverlayJobPrefix{libraries, {"OverlayJobCull.metal", "OverlayJobPrefix"}},
      OverlayJobEmit{libraries, {"OverlayJobCull.metal", "OverlayJobEmit"}},
      DepthPyramidReduce{libraries, {"DepthPyramidReduce.metal", "DepthPyramidReduceKernel"}},
      EquirectToCubemap{libraries, {"IblPrefilter.metal", "EquirectToCubemapKernel"}},
      DiffuseIrradiance{libraries, {"IblPrefilter.metal", "DiffuseIrradianceKernel"}},
      SpecularPrefilter{libraries, {"IblPrefilter.metal", "SpecularPrefilterKernel"}},
      MipDownsample{libraries, {"MipDownsample.metal", "MipDownsampleKernel"}} {}

Pipelines &GetPipelines(state::Registry &r) {
    if (auto *pipelines = r.Resources.find<Pipelines>()) return *pipelines;
    return r.Resources.emplace<Pipelines>(r.Resources.get<mtl::LibraryCache>());
}
