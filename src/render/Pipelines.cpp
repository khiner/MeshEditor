#include "render/Pipelines.h"
#include "Profile.h"
#include "gpu/BackgroundConstant.h"
#include "gpu/EditOverlayConstant.h"
#include "gpu/MeshVertexConstant.h"
#include "gpu/NormalIndicatorConstant.h"
#include "gpu/PbrConstant.h"
#include "metal/Bindless.h"
#include "metal/Buffer.h"

#include <array>
#include <bit>
#include <format>
#include <stdexcept>

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

mtl::MeshRenderPipeline CreateMeshPipeline(
    mtl::LibraryCache &libraries, std::optional<FunctionRef> fragment, PassFormats formats,
    std::vector<BlendState> blends = {}, std::optional<DepthState> depth = {}, FunctionRef mesh = MeshletVertex()
) {
    return {libraries, std::move(mesh), std::move(fragment), std::move(formats), std::move(blends), depth};
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

mtl::MeshRenderPipeline StrokePipeline(mtl::LibraryCache &libraries, FunctionRef mesh, bool include_outer = false) {
    return CreateMeshPipeline(
        libraries, FunctionRef{"EdgeQuad.metal", "EdgeQuadFragment", {BoolConstant(EditOverlayConstant::IncludeOuter, include_outer)}}, OverlayFormats(), {PremultipliedBlend}, DepthTestNoWriteLessEqual,
        std::move(mesh)
    );
}
} // namespace

void PipelineRenderer::CompileShaders(mtl::LibraryCache &libraries) {
    for (auto &pipeline : Pipelines) pipeline.second.Compile(libraries);
}

const RenderPipeline &PipelineRenderer::Bind(MTL::RenderCommandEncoder *encoder, SPT type) const {
    const auto it = Pipelines.find(type);
    if (it == Pipelines.end()) throw std::runtime_error(std::format("No pipeline for shader pipeline type {}", int(type)));
    const auto &pipeline = it->second;
    pipeline.Bind(encoder);
    encoder->setDepthBias(pipeline.DepthBias(), 0.f, 0.f);
    return pipeline;
}

// `transmission_prepass` skips exposure, which the main pass applies after sampling.
static RenderPipeline CreateBackgroundPipeline(mtl::LibraryCache &libraries, PassFormats formats, std::vector<mtl::BlendState> blends, bool transmission_prepass) {
    return {
        libraries, {"Background.metal", "BackgroundVertex"}, FunctionRef{"Background.metal", "BackgroundFragment", {BoolConstant(BackgroundConstant::TransmissionPrepass, transmission_prepass)}}, std::move(formats), std::move(blends), DepthOff
    };
}

static RenderPipeline CreateQuadPipeline(mtl::LibraryCache &libraries, PassFormats formats, const char *fragment_file, const char *fragment_name, mtl::BlendState blend) {
    return {
        libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{fragment_file, fragment_name}, std::move(formats), {blend}, DepthOff
    };
}

static PipelineRenderer CreateSceneRenderer(mtl::LibraryCache &libraries) {
    const auto formats = SceneFormats();
    std::unordered_map<SPT, RenderPipeline> pipelines;
    pipelines.emplace(SPT::Background, CreateBackgroundPipeline(libraries, formats, {Blend}, false));
    pipelines.emplace(SPT::TransmissionComposite, CreateQuadPipeline(libraries, formats, "TransmissionComposite.metal", "TransmissionCompositeFragment", PremultipliedBlend));
    pipelines.emplace(SPT::MotionBlurResolve, CreateQuadPipeline(libraries, formats, "MotionBlurResolve.metal", "MotionBlurResolveFragment", NoBlend));
    return {formats, std::move(pipelines)};
}

static PipelineRenderer CreateOverlayRenderer(mtl::LibraryCache &libraries) {
    const auto formats = OverlayFormats();
    std::unordered_map<SPT, RenderPipeline> pipelines;
    pipelines.emplace(SPT::Grid, RenderPipeline{libraries, {"GridLines.metal", "GridLinesVertex"}, FunctionRef{"GridLines.metal", "GridLinesFragment"}, formats, {Blend}, DepthState{.Write = false}});
    pipelines.emplace(SPT::SilhouetteEdgeColor, RenderPipeline{libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"SilhouetteEdgeColor.metal", "SilhouetteEdgeColorFragment"}, formats, {NoBlend}, DepthState{.Test = false}});
    return {formats, std::move(pipelines)};
}

PbrCompiler::PbrCompiler(PassFormats scene) : SceneFormats(std::move(scene)) {}

bool PbrCompiler::CompilePipelines(mtl::LibraryCache &libraries, PbrFeatureMask mask, bool non_triangle_topology) {
    if (mask == Mask && non_triangle_topology == NonTriangleTopology && Visibility) return false;
    const auto visibility = [&](bool prepass) {
        return std::make_unique<mtl::RenderPipeline>(libraries, FunctionRef{"TexQuad.metal", "TexQuadVertex"}, PbrFragment(mask, prepass, "PbrVisibilityFragment", non_triangle_topology), SceneFormats, std::vector<BlendState>{Blend}, DepthOff);
    };
    Visibility = visibility(false);
    Prepass = ::HasFeature(mask, PbrFeature::Transmission) ? visibility(true) : nullptr;
    Transparent = std::make_unique<mtl::MeshRenderPipeline>(libraries, MeshletVertex(non_triangle_topology), PbrFragment(mask, false, "PbrTransparentFragment", non_triangle_topology), SceneFormats, std::vector<BlendState>{NoWrite}, DepthTestNoWriteLessEqual);
    Mask = mask;
    NonTriangleTopology = non_triangle_topology;
    return true;
}

void PbrCompiler::BindMeshlets(MTL::RenderCommandEncoder *encoder) const { Transparent->Bind(encoder); }
void PbrCompiler::BindVisibility(MTL::RenderCommandEncoder *encoder, bool prepass) const { (prepass ? Prepass : Visibility)->Bind(encoder); }

void PbrCompiler::RecompileModules(mtl::LibraryCache &libraries) {
    if (Transparent) Transparent->Compile(libraries);
    if (Visibility) Visibility->Compile(libraries);
    if (Prepass) Prepass->Compile(libraries);
}

MainPipeline::MainPipeline(mtl::LibraryCache &libraries)
    : SceneRenderer{CreateSceneRenderer(libraries)},
      OverlayRenderer{CreateOverlayRenderer(libraries)},
      PrepassBackground{CreateBackgroundPipeline(libraries, SceneFormats(), {Blend}, true)},
      ViewportComposite{CreateQuadPipeline(libraries, {{Format::Color}, MTL::PixelFormatInvalid}, "ViewportComposite.metal", "ViewportCompositeFragment", NoBlend)},
      MotionBlurAccumulate{CreateQuadPipeline(libraries, {{MTL::PixelFormatRGBA32Float}, MTL::PixelFormatInvalid}, "MotionBlurAccumulate.metal", "MotionBlurAccumulateFragment", AdditiveBlend)},
      MotionBlurGather{CreateQuadPipeline(libraries, {{Format::HdrColor}, MTL::PixelFormatInvalid}, "MotionBlurGather.metal", "MotionBlurGatherFragment", NoBlend)},
      MotionBlurTilesFlatten{libraries, {"MotionBlurTilesFlatten.metal", "MotionBlurTilesFlattenKernel"}},
      MotionBlurTilesDilate{libraries, {"MotionBlurTilesDilate.metal", "MotionBlurTilesDilateKernel"}},
      WorkspaceVisibility{libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"WorkspaceLighting.metal", "WorkspaceVisibilityFragment"}, SceneFormats(), {Blend}, DepthOff},
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
      WireResolve{libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"WireResolve.metal", "WireResolveFragment"}, OverlayFormats(), {PremultipliedBlend}, DepthOff},
      Compiler{SceneFormats()} {}

MainPipeline::ResourcesT::ResourcesT(const mtl::Context &ctx, mtl::Extent2D extent, mtl::BindlessSet &slots)
    // Visibility depth remains paired with its IDs throughout shading and selection.
    : VisibilityDepth{mtl::CreateTexture2D(ctx, Format::Depth, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      ScratchDepth{mtl::CreateTexture2D(ctx, Format::Depth, extent, MTL::TextureUsageRenderTarget)},
      VisibilityImage{mtl::CreateTexture2D(ctx, Format::Uint, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      SilhouetteImage{mtl::CreateTexture2D(ctx, Format::Float2, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      SceneColorImage{mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      OverlayColorImage{mtl::CreateTexture2D(ctx, Format::Color, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      FinalColorImage{mtl::CreateTexture2D(ctx, Format::Color, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      DepthPyramidImage{[&] {
          const mtl::Extent2D padded{std::bit_ceil((extent.Width + 1) / 2), std::bit_ceil((extent.Height + 1) / 2)};
          return mtl::CreateTexture2D(ctx, Format::Float, padded, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite, mtl::MipLevelCount(padded.Width, padded.Height));
      }()},
      DepthPyramidMips{[&] {
          std::vector<PyramidMip> mips;
          mips.reserve(DepthPyramidImage.MipLevels);
          for (uint32_t mip = 0; mip < DepthPyramidImage.MipLevels; ++mip) {
              const mtl::Extent2D data_extent{((extent.Width - 1) >> (mip + 1)) + 1, ((extent.Height - 1) >> (mip + 1)) + 1};
              mips.push_back({mtl::CreateMipView(DepthPyramidImage, mip), slots.Allocate(SlotType::Image), data_extent});
          }
          return mips;
      }()},
      NearestSampler{mtl::CreateSampler(ctx, MTL::SamplerMinMagFilterNearest, MTL::SamplerMipFilterNearest, MTL::SamplerAddressModeClampToEdge)},
      Slots{slots} {
    for (const auto &mip : DepthPyramidMips) slots.SetTexture(mip.Slot, *mip.View);
}

MainPipeline::ResourcesT::~ResourcesT() {
    for (const auto &mip : DepthPyramidMips) Slots.Release({SlotType::Image, mip.Slot});
}

MainPipeline::TransmissionResourcesT::TransmissionResourcesT(const mtl::Context &ctx, mtl::Extent2D extent)
    : Image{mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead, mtl::MipLevelCount(extent.Width, extent.Height))},
      Mip0View{mtl::CreateMipView(Image, 0)},
      Sampler{mtl::CreateSampler(ctx, MTL::SamplerMinMagFilterLinear, MTL::SamplerMipFilterLinear, MTL::SamplerAddressModeClampToEdge)} {}

MainPipeline::MotionBlurResourcesT::MotionBlurResourcesT(const mtl::Context &ctx, mtl::Extent2D extent, bool fast)
    : OutputImage{mtl::CreateTexture2D(ctx, fast ? Format::HdrColor : MTL::PixelFormatRGBA32Float, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)} {
    if (!fast) return;
    const mtl::Extent2D tiles{(extent.Width + 31u) / 32u, (extent.Height + 31u) / 32u};
    VelocityImage = mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    TileImage = mtl::CreateTexture2D(ctx, Format::HdrColor, tiles, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    TileIndirection = mtl::NewBuffer(ctx, 2u * tiles.Width * tiles.Height * sizeof(uint32_t));
}

void MainPipeline::SetExtent(const mtl::Context &ctx, mtl::Extent2D extent, mtl::BindlessSet &slots) {
    Resources = std::make_unique<ResourcesT>(ctx, extent, slots);
    Transmission.reset();
    MotionBlur.reset();
}

bool MainPipeline::EnsureTransmissionResources(const mtl::Context &ctx, mtl::Extent2D extent, bool wanted) {
    if (!wanted) {
        if (!Transmission) return false;
        Transmission.reset();
        return true;
    }
    // Metal rejects zero-sized targets.
    if (extent.Width == 0 || extent.Height == 0) return false;
    if (Transmission && Transmission->Image.Extent == extent) return false;
    Transmission = std::make_unique<TransmissionResourcesT>(ctx, extent);
    return true;
}

bool MainPipeline::EnsureMotionBlurResources(const mtl::Context &ctx, bool fast) {
    if (!Resources || (MotionBlur && bool(MotionBlur->VelocityImage) == fast)) return false; // SetExtent drops it, so an allocated target is always at the color extent.
    MotionBlur = std::make_unique<MotionBlurResourcesT>(ctx, Resources->SceneColorImage.Extent, fast);
    return true;
}

SampledTexture MainPipeline::Nearest(const mtl::Texture *image) const {
    if (!Resources) return {};
    return {image ? **image : *Resources->SceneColorImage, Resources->NearestSampler.get()};
}
SampledTexture MainPipeline::SceneColorSampler() const { return Nearest(nullptr); }
SampledTexture MainPipeline::OverlayColorSampler() const { return Nearest(Resources ? &Resources->OverlayColorImage : nullptr); }
SampledTexture MainPipeline::SceneDepthSampler() const { return Nearest(Resources ? &Resources->VisibilityDepth : nullptr); }
SampledTexture MainPipeline::DepthPyramidSampler() const { return Nearest(Resources ? &Resources->DepthPyramidImage : nullptr); }
SampledTexture MainPipeline::MotionBlurOutputSampler() const { return Nearest(MotionBlur ? &MotionBlur->OutputImage : nullptr); }
SampledTexture MainPipeline::TransmissionSampler() const {
    if (!Resources) return {};
    if (!Transmission) return SceneColorSampler();
    return {*Transmission->Image, Transmission->Sampler.get()};
}

// The element rasters share these formats: depth only, and no color, since ids reach the fragment stage as a varying.
static PassFormats SelectionFormats() { return {{}, Format::Depth}; }

static mtl::MeshRenderPipeline MeshletElementRaster(
    mtl::LibraryCache &libraries, FunctionRef mesh, bool bitset_box, bool xray
) {
    const FunctionRef fragment = bitset_box ?
        FunctionRef{"SelectionElementBitsetBox.metal", "SelectionElementBitsetBoxFragment"} :
        FunctionRef{"SelectionElementPick.metal", "SelectionElementPickFragment"};
    return CreateMeshPipeline(libraries, fragment, SelectionFormats(), {}, xray ? DepthOff : DepthTestNoWriteLessEqual, std::move(mesh));
}

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
      OverlayJobLines{CreateMeshPipeline(libraries, FunctionRef{"SelectionFragment.metal", "SelectionStrokeFragment"}, SelectionFormats(), {}, DepthOff, {"OverlayJobLine.metal", "OverlayJobLineMesh"})},
      BoneSphere{CreateMeshPipeline(libraries, FunctionRef{"SelectionFragment.metal", "SelectionFragment"}, SelectionFormats(), {}, DepthOff, {"BoneSphere.metal", "BoneSphereMesh"})} {}

const mtl::MeshRenderPipeline &SelectionFragmentPipeline::ElementRaster(
    Element element, bool bitset_box, bool xray
) const {
    const auto &variants = element == Element::Face ? MeshletFaces : element == Element::Vertex ? MeshletVertices :
                                                                                                  MeshletEdges;
    return variants[uint32_t(bitset_box) + 2u * uint32_t(xray)];
}

Pipelines::Pipelines(mtl::LibraryCache &libraries)
    : Libraries(libraries),
      Main{libraries},
      Silhouette{libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"VisibilitySelection.metal", "VisibilitySilhouetteFragment"}, PassFormats{{Format::Float2}, Format::Depth}, {NoBlend}, DepthTestWrite},
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
      IblPrefilter{libraries},
      VertexAdjacency{libraries},
      VertexWeld{libraries},
      MeshConnectivity{libraries} {}

void Pipelines::CompileShaders() {
    Libraries.Clear();
    Main.SceneRenderer.CompileShaders(Libraries);
    Main.OverlayRenderer.CompileShaders(Libraries);
    Main.PrepassBackground.Compile(Libraries);
    Main.ViewportComposite.Compile(Libraries);
    Main.MotionBlurAccumulate.Compile(Libraries);
    Main.MotionBlurGather.Compile(Libraries);
    Main.MotionBlurTilesFlatten.Compile(Libraries);
    Main.MotionBlurTilesDilate.Compile(Libraries);
    Main.WorkspaceVisibility.Compile(Libraries);
    Main.TransparencyInit.Compile(Libraries);
    Main.TransparencyResolve.Compile(Libraries);
    Main.MeshletVisibilityOpaque.Compile(Libraries);
    Main.MeshletVisibilityCoverage.Compile(Libraries);
    Main.FaceNormalMesh.Compile(Libraries);
    Main.VertexNormalMesh.Compile(Libraries);
    Main.OverlayJobLines.Compile(Libraries);
    Main.MeshletEditEdges.Compile(Libraries);
    Main.MeshletEditSmoothEdges.Compile(Libraries);
    Main.MeshletEditPoint.Compile(Libraries);
    for (auto *bone : {&Main.BoneFillMesh, &Main.BoneWireMesh, &Main.BoneSphereFillMesh, &Main.BoneSphereWireMesh}) bone->Compile(Libraries);
    Main.WireResolve.Compile(Libraries);
    Main.Compiler.RecompileModules(Libraries);
    Silhouette.Compile(Libraries);
    SelectionFragment.OverlayJobLines.Compile(Libraries);
    SelectionFragment.BoneSphere.Compile(Libraries);
    for (auto *variants : {&SelectionFragment.MeshletFaces, &SelectionFragment.MeshletVertices, &SelectionFragment.MeshletEdges}) {
        for (auto &pipeline : *variants) pipeline.Compile(Libraries);
    }
    for (auto *pipeline : {&SelectionFragment.MeshletFaceXRayPointsBitsetBox, &SelectionFragment.MeshletEdgeXRayPointsBitsetBox}) {
        pipeline->Compile(Libraries);
    }
    PrepareEditSelection.Compile(Libraries);
    FillEditSelectionList.Compile(Libraries);
    EditSharpness.Compile(Libraries);
    CommitPosedGeometry.Compile(Libraries);
    GeometryWorkArgs.Compile(Libraries);
    for (auto *compute : {&VisibilityObjectSelection, &ResetEditSelectionSummary, &DeriveEditSelection, &SumEditSelectionPosition, &PosePrepass, &PosedMeshletBounds, &VertexNormalDerive, &BoundsReduce, &BoundsCombine, &BoundsTree, &WireRaster, &LodFrontierCount, &LodFrontierPrefix, &LodFrontierEmit, &MeshletCullBlockCount, &MeshletCullPrefix, &MeshletCullEmit, &OverlayJobBlockCount, &OverlayJobPrefix, &OverlayJobEmit, &DepthPyramidReduce, &IblPrefilter.EquirectToCubemap, &IblPrefilter.DiffuseIrradiance, &IblPrefilter.SpecularPrefilter, &VertexAdjacency.Zero, &VertexAdjacency.Count, &VertexAdjacency.BlockSum, &VertexAdjacency.BlockPrefix, &VertexAdjacency.Offsets, &VertexAdjacency.Scatter, &VertexAdjacency.Sort, &VertexWeld.TableInit, &VertexWeld.Insert, &VertexWeld.MarkReps, &VertexWeld.BlockSum, &VertexWeld.BlockPrefix, &VertexWeld.Scan, &VertexWeld.Emit, &VertexWeld.Compact, &VertexWeld.WriteBack, &VertexWeld.RemapCorners, &MeshConnectivity.Zero, &MeshConnectivity.Count, &MeshConnectivity.BlockSum, &MeshConnectivity.BlockPrefix, &MeshConnectivity.Offsets, &MeshConnectivity.Scatter, &MeshConnectivity.Pair, &MeshConnectivity.Bits, &MeshConnectivity.WordBlockSum, &MeshConnectivity.WordBlockPrefix, &MeshConnectivity.Ranks, &MeshConnectivity.Samples}) {
        compute->Compile(Libraries);
    }
}
