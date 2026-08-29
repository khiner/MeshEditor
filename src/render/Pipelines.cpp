#include "render/Pipelines.h"
#include "Profile.h"
#include "gpu/BackgroundConstant.h"
#include "gpu/EditOverlayConstant.h"
#include "gpu/MeshVertexConstant.h"
#include "gpu/NormalIndicatorConstant.h"
#include "gpu/PbrConstant.h"
#include "metal/Bindless.h"

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
std::vector<mtl::FunctionConstant> MeshVertexConstants(bool velocity, bool non_triangle_topology = false) {
    return {
        BoolConstant(MeshVertexConstant::VelocityOutput, velocity),
        BoolConstant(MeshVertexConstant::NonTriangleTopology, non_triangle_topology),
    };
}
FunctionRef NormalIndicatorMesh(bool faces) {
    return {"NormalIndicator.metal", "NormalIndicatorMesh", {BoolConstant(NormalIndicatorConstant::NormalIndicatorFaces, faces)}};
}

FunctionRef MeshletVertex(bool velocity = false, bool non_triangle_topology = false) {
    return {
        "MeshletTransform.metal", "MeshletForwardMesh",
        MeshVertexConstants(velocity, non_triangle_topology)
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

struct PbrPipelineSpec {
    bool VelocityPass, VelocityOutput;
    FunctionRef Fragment;
    std::vector<BlendState> Blends;
    DepthState Depth;
};

PbrPipelineSpec MakePbrPipelineSpec(
    PbrFeatureMask mask, bool prepass, PbrCompiler::Variant variant, const char *fragment,
    bool non_triangle_topology
) {
    const bool velocity_pass = variant == PbrCompiler::Variant::OpaqueVelocity || variant == PbrCompiler::Variant::BlendVelocity;
    const bool velocity_output = variant == PbrCompiler::Variant::OpaqueVelocity;
    std::vector<FunctionConstant> constants;
    constants.reserve(PbrSpecFeatures.size() + 2);
    for (const auto &[constant, feature] : PbrSpecFeatures) constants.push_back(BoolConstant(constant, HasFeature(mask, feature)));
    constants.push_back(BoolConstant(PbrConstant::TransmissionPrepass, prepass));
    constants.push_back(BoolConstant(PbrConstant::VelocityOutput, velocity_output));
    constants.push_back(BoolConstant(PbrConstant::NonTriangleTopology, non_triangle_topology));
    std::vector<BlendState> blends{Blend};
    if (velocity_pass) blends.push_back(velocity_output ? NoBlend : NoWrite);
    return {velocity_pass, velocity_output, {"pbr.metal", fragment, std::move(constants)}, std::move(blends), {.Write = variant != PbrCompiler::Variant::Blend && variant != PbrCompiler::Variant::BlendVelocity}};
}

PassFormats SceneFormats() { return {{Format::HdrColor}, Format::Depth}; }
PassFormats SceneVelocityFormats() { return {{Format::HdrColor, Format::Velocity}, Format::Depth}; }
PassFormats OverlayFormats() { return {{Format::Color, Format::LineData}, Format::Depth}; }

mtl::MeshRenderPipeline MeshletEditEdgePipeline(mtl::LibraryCache &libraries, bool include_outer) {
    return CreateMeshPipeline(
        libraries, FunctionRef{
            "EdgeQuad.metal", "EdgeQuadFragment",
            {BoolConstant(EditOverlayConstant::IncludeOuter, include_outer)}
        }, OverlayFormats(), {Blend, NoWrite}, DepthTestNoWriteLessEqual,
        {"MeshletEditOverlay.metal", "MeshletEditEdgeMesh"}
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
    // Non-edge texels fail the depth test, preserving prepass depth.
    pipelines.emplace(SPT::SilhouetteEdgeDepth, RenderPipeline{libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"SampleDepth.metal", "SampleDepthFragment"}, formats, {NoWrite}, DepthTestWrite});
    return {formats, std::move(pipelines)};
}

static PipelineRenderer CreateSceneVelocityRenderer(mtl::LibraryCache &libraries) {
    const auto formats = SceneVelocityFormats();
    std::unordered_map<SPT, RenderPipeline> pipelines;
    // Screen motion for every pixel geometry leaves uncovered. Drawn first, so geometry overwrites
    // it wherever it lands, and the scene color stays untouched through the write mask.
    pipelines.emplace(SPT::BackgroundVelocity, RenderPipeline{libraries, {"Background.metal", "BackgroundVertex"}, FunctionRef{"BackgroundVelocity.metal", "BackgroundVelocityFragment"}, formats, {NoWrite, NoBlend}, DepthOff});
    pipelines.emplace(SPT::Background, CreateBackgroundPipeline(libraries, formats, {Blend, NoWrite}, false));
    pipelines.emplace(SPT::SilhouetteEdgeDepth, RenderPipeline{libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"SampleDepth.metal", "SampleDepthFragment"}, formats, {NoWrite, NoWrite}, DepthTestWrite});
    return {formats, std::move(pipelines)};
}

static PipelineRenderer CreateOverlayRenderer(mtl::LibraryCache &libraries) {
    const auto formats = OverlayFormats();
    const FunctionRef vertex_color{"VertexColor.metal", "VertexColorFragment"};
    std::unordered_map<SPT, RenderPipeline> pipelines;
    pipelines.emplace(SPT::Grid, RenderPipeline{libraries, {"GridLines.metal", "GridLinesVertex"}, FunctionRef{"GridLines.metal", "GridLinesFragment"}, formats, {Blend, NoWrite}, DepthState{.Write = false}});
    pipelines.emplace(SPT::SilhouetteEdgeColor, RenderPipeline{libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"SilhouetteEdgeColor.metal", "SilhouetteEdgeColorFragment"}, formats, {Blend, NoWrite}, DepthOff});
    return {formats, std::move(pipelines)};
}

PbrCompiler::PbrCompiler(PassFormats scene, PassFormats scene_velocity)
    : SceneFormats(std::move(scene)), VelocityFormats(std::move(scene_velocity)) {}

std::unique_ptr<mtl::MeshRenderPipeline> PbrCompiler::CreateMeshletPipeline(
    mtl::LibraryCache &libraries, PbrFeatureMask mask, bool prepass, Variant variant,
    bool non_triangle_topology
) const {
    auto spec = MakePbrPipelineSpec(mask, prepass, variant, "PbrMeshletFragment", non_triangle_topology);
    return std::make_unique<mtl::MeshRenderPipeline>(
        libraries, MeshletVertex(spec.VelocityOutput, non_triangle_topology), std::move(spec.Fragment),
        spec.VelocityPass ? VelocityFormats : SceneFormats, std::move(spec.Blends), spec.Depth
    );
}

std::unique_ptr<mtl::RenderPipeline> PbrCompiler::CreateVisibilityPipeline(
    mtl::LibraryCache &libraries, PbrFeatureMask mask, bool prepass, Variant variant,
    bool non_triangle_topology
) const {
    auto spec = MakePbrPipelineSpec(mask, prepass, variant, "PbrVisibilityFragment", non_triangle_topology);
    return std::make_unique<mtl::RenderPipeline>(
        libraries, FunctionRef{"TexQuad.metal", "TexQuadVertex"}, std::move(spec.Fragment),
        spec.VelocityPass ? VelocityFormats : SceneFormats, std::move(spec.Blends), DepthOff
    );
}

bool PbrCompiler::CompilePipelines(
    mtl::LibraryCache &libraries, PbrFeatureMask mask, bool non_triangle_topology
) {
    if (mask == Mask && non_triangle_topology == NonTriangleTopology &&
        MeshletVariants[size_t(Variant::Opaque)] && MeshletVariants[size_t(Variant::Blend)]) return false;
    const profile::CpuScope scope{"CompilePbrPipelines"};

    const bool transmission = ::HasFeature(mask, PbrFeature::Transmission);
    for (size_t v = 0; v < VariantCount; ++v) {
        const auto variant = Variant(v);
        if (variant == Variant::OpaquePrepass && !transmission) {
            MeshletVariants[v].reset();
        } else {
            MeshletVariants[v] = CreateMeshletPipeline(
                libraries, mask, variant == Variant::OpaquePrepass, variant, non_triangle_topology
            );
        }
    }
    for (auto &pipeline : VisibilityVariants) pipeline.reset();
    for (const auto variant : {Variant::Opaque, Variant::OpaqueVelocity}) {
        VisibilityVariants[size_t(variant)] = CreateVisibilityPipeline(
            libraries, mask, false, variant, non_triangle_topology
        );
    }
    if (transmission) VisibilityVariants[size_t(Variant::OpaquePrepass)] = CreateVisibilityPipeline(
                          libraries, mask, true, Variant::OpaquePrepass, non_triangle_topology
                      );
    Mask = mask;
    NonTriangleTopology = non_triangle_topology;
    return true;
}

void PbrCompiler::BindMeshlets(MTL::RenderCommandEncoder *encoder, Variant variant) const {
    const auto &pipeline = MeshletVariants[size_t(variant)];
    if (!pipeline) throw std::runtime_error("PbrCompiler: binding a meshlet variant that was never compiled.");
    pipeline->Bind(encoder);
}

void PbrCompiler::BindVisibility(MTL::RenderCommandEncoder *encoder, Variant variant) const {
    const auto &pipeline = VisibilityVariants[size_t(variant)];
    if (!pipeline) throw std::runtime_error("PbrCompiler: binding a visibility variant that was never compiled.");
    pipeline->Bind(encoder);
}

void PbrCompiler::RecompileModules(mtl::LibraryCache &libraries) {
    for (auto &variant : MeshletVariants) {
        if (variant) variant->Compile(libraries);
    }
    for (auto &variant : VisibilityVariants) {
        if (variant) variant->Compile(libraries);
    }
}

MainPipeline::MainPipeline(mtl::LibraryCache &libraries)
    : SceneRenderer{CreateSceneRenderer(libraries)},
      OverlayRenderer{CreateOverlayRenderer(libraries)},
      SceneVelocityRenderer{CreateSceneVelocityRenderer(libraries)},
      PrepassBackground{CreateBackgroundPipeline(libraries, SceneFormats(), {Blend}, true)},
      CompositeFormats{{Format::Color}, MTL::PixelFormatInvalid},
      ViewportComposite{CreateQuadPipeline(libraries, CompositeFormats, "ViewportComposite.metal", "ViewportCompositeFragment", NoBlend)},
      MotionBlurAccumFormats{{Format::HdrColor}, MTL::PixelFormatInvalid},
      MotionBlurAccumulate{CreateQuadPipeline(libraries, MotionBlurAccumFormats, "MotionBlurAccumulate.metal", "MotionBlurAccumulateFragment", AdditiveBlend)},
      MotionBlurGatherFormats{{Format::HdrColor}, MTL::PixelFormatInvalid},
      MotionBlurGather{CreateQuadPipeline(libraries, MotionBlurGatherFormats, "MotionBlurGather.metal", "MotionBlurGatherFragment", NoBlend)},
      WorkspaceVisibility{libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"WorkspaceLighting.metal", "WorkspaceVisibilityFragment"}, SceneFormats(), {Blend}, DepthOff},
      MeshletVisibilityOpaque{CreateMeshPipeline(libraries, FunctionRef{"MeshletVisibility.metal", "MeshletVisibilityOpaqueFragment"}, {{Format::Uint}, Format::Depth}, {NoBlend}, DepthTestWrite, MeshletVisibilityVertex())},
      MeshletVisibilityCoverage{CreateMeshPipeline(libraries, FunctionRef{"MeshletVisibility.metal", "MeshletVisibilityPrimitiveFragment"}, {{Format::Uint}, Format::Depth}, {NoBlend}, DepthTestWrite, MeshletVisibilityVertex())},
      MeshletEditEdges{MeshletEditEdgePipeline(libraries, true)},
      MeshletEditSmoothEdges{MeshletEditEdgePipeline(libraries, false)},
      MeshletEditPoint{CreateMeshPipeline(libraries, FunctionRef{"VertexPoint.metal", "VertexPointFragment"}, OverlayFormats(), {Blend, NoWrite}, DepthTestLessEqual, {"MeshletEditOverlay.metal", "MeshletEditPointMesh"})},
      FaceNormalMesh{CreateMeshPipeline(libraries, FunctionRef{"VertexColor.metal", "VertexColorFragment"}, OverlayFormats(), {Blend, NoBlend}, DepthTestLessEqual, NormalIndicatorMesh(true))},
      VertexNormalMesh{CreateMeshPipeline(libraries, FunctionRef{"VertexColor.metal", "VertexColorFragment"}, OverlayFormats(), {Blend, NoBlend}, DepthTestLessEqual, NormalIndicatorMesh(false))},
      OverlayJobLines{CreateMeshPipeline(libraries, FunctionRef{"VertexColor.metal", "VertexColorFragment"}, OverlayFormats(), {Blend, NoBlend}, DepthTestLessEqual, {"OverlayJobLine.metal", "OverlayJobLineMesh"})},
      BoneFillMesh{CreateMeshPipeline(libraries, FunctionRef{"BoneSolid.metal", "BoneSolidFragment"}, OverlayFormats(), {Blend, NoWrite}, DepthTestWrite, {"BoneSolid.metal", "BoneSolidMesh"})},
      BoneWireMesh{CreateMeshPipeline(libraries, FunctionRef{"VertexColor.metal", "VertexColorFragment"}, OverlayFormats(), {Blend, NoBlend}, DepthTestNoWriteLessEqual, {"BoneWire.metal", "BoneWireMesh"})},
      BoneSphereFillMesh{CreateMeshPipeline(libraries, FunctionRef{"BoneSphere.metal", "BoneSphereFragment"}, OverlayFormats(), {Blend, NoWrite}, DepthTestLessEqual, {"BoneSphere.metal", "BoneSphereMesh"})},
      BoneSphereWireMesh{CreateMeshPipeline(libraries, FunctionRef{"VertexColor.metal", "VertexColorFragment"}, OverlayFormats(), {Blend, NoBlend}, DepthTestNoWriteLessEqual, {"BoneSphereWire.metal", "BoneSphereWireMesh"})},
      WireResolve{libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"WireResolve.metal", "WireResolveFragment"}, OverlayFormats(), {Blend, NoWrite}, DepthTestLessEqual},
      Compiler{SceneFormats(), SceneVelocityFormats()} {}

MainPipeline::ResourcesT::ResourcesT(const mtl::Context &ctx, mtl::Extent2D extent, mtl::BindlessSet &slots)
    // Depth is sampled as well as attached: the motion blur gather reads it to sort samples.
    : DepthImage{mtl::CreateTexture2D(ctx, Format::Depth, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      VisibilityImage{mtl::CreateTexture2D(ctx, Format::Uint, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      SceneColorImage{mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      OverlayColorImage{mtl::CreateTexture2D(ctx, Format::Color, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      LineDataImage{mtl::CreateTexture2D(ctx, Format::LineData, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
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
    for (const auto &mip : DepthPyramidMips) slots.SetTexture(mip.Slot, mip.View.get());
}

MainPipeline::ResourcesT::~ResourcesT() {
    for (const auto &mip : DepthPyramidMips) Slots.Release({SlotType::Image, mip.Slot});
}

MainPipeline::TransmissionResourcesT::TransmissionResourcesT(const mtl::Context &ctx, mtl::Extent2D extent)
    : Image{mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead, mtl::MipLevelCount(extent.Width, extent.Height))},
      Mip0View{mtl::CreateMipView(Image, 0)},
      Sampler{mtl::CreateSampler(ctx, MTL::SamplerMinMagFilterLinear, MTL::SamplerMipFilterLinear, MTL::SamplerAddressModeClampToEdge)} {}

MainPipeline::MotionBlurResourcesT::MotionBlurResourcesT(const mtl::Context &ctx, mtl::Extent2D extent)
    : AccumImage{mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      VelocityImage{mtl::CreateTexture2D(ctx, Format::Velocity, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      TileImage{mtl::CreateTexture2D(ctx, Format::HdrColor, {(extent.Width + 31) / 32, (extent.Height + 31) / 32}, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite)},
      GatherImage{mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)} {}

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

bool MainPipeline::EnsureMotionBlurResources(const mtl::Context &ctx) {
    if (MotionBlur || !Resources) return false; // SetExtent drops it, so an allocated target is always at the color extent.
    MotionBlur = std::make_unique<MotionBlurResourcesT>(ctx, Resources->SceneColorImage.Extent);
    return true;
}

SampledTexture MainPipeline::Nearest(const mtl::Texture *image) const {
    if (!Resources) return {};
    return {image ? **image : *Resources->SceneColorImage, Resources->NearestSampler.get()};
}
SampledTexture MainPipeline::SceneColorSampler() const { return Nearest(nullptr); }
SampledTexture MainPipeline::OverlayColorSampler() const { return Nearest(Resources ? &Resources->OverlayColorImage : nullptr); }
SampledTexture MainPipeline::SceneDepthSampler() const { return Nearest(Resources ? &Resources->DepthImage : nullptr); }
SampledTexture MainPipeline::DepthPyramidSampler() const { return Nearest(Resources ? &Resources->DepthPyramidImage : nullptr); }
SampledTexture MainPipeline::MotionBlurAccumSampler() const { return Nearest(MotionBlur ? &MotionBlur->AccumImage : nullptr); }
SampledTexture MainPipeline::VelocitySampler() const { return Nearest(MotionBlur ? &MotionBlur->VelocityImage : nullptr); }
SampledTexture MainPipeline::MotionBlurGatherSampler() const { return Nearest(MotionBlur ? &MotionBlur->GatherImage : nullptr); }
MTL::Texture *MainPipeline::MotionBlurTileImage() const { return MotionBlur ? *MotionBlur->TileImage : nullptr; }
SampledTexture MainPipeline::TransmissionSampler() const {
    if (!Resources) return {};
    if (!Transmission) return SceneColorSampler();
    return {*Transmission->Image, Transmission->Sampler.get()};
}

SilhouettePipeline::SilhouettePipeline(mtl::LibraryCache &libraries)
    : Visibility{
          libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"VisibilitySelection.metal", "VisibilitySilhouetteFragment"}, PassFormats{{Format::Float2}, Format::Depth}, {NoBlend}, DepthTestWrite
      } {}

SilhouettePipeline::ResourcesT::ResourcesT(const mtl::Context &ctx, mtl::Extent2D extent)
    : DepthImage{mtl::CreateTexture2D(ctx, Format::Depth, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      OffscreenImage{mtl::CreateTexture2D(ctx, Format::Float2, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      // Clamping keeps edge detection from wrapping around to the image's other side: it reads the
      // pixel value at the nearest edge instead.
      ImageSampler{mtl::CreateSampler(ctx, MTL::SamplerMinMagFilterNearest, MTL::SamplerMipFilterNearest, MTL::SamplerAddressModeClampToEdge)} {}

void SilhouettePipeline::SetExtent(const mtl::Context &ctx, mtl::Extent2D extent) {
    Resources = std::make_unique<ResourcesT>(ctx, extent);
}

static PipelineRenderer CreateSilhouetteEdgeRenderer(mtl::LibraryCache &libraries) {
    const PassFormats formats{{Format::Float}, Format::Depth};
    std::unordered_map<SPT, RenderPipeline> pipelines;
    pipelines.emplace(SPT::SilhouetteEdgeDepthObject, RenderPipeline{libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"SilhouetteEdgeDepthObject.metal", "SilhouetteEdgeDepthObjectFragment"}, formats, {NoBlend}, DepthTestWrite});
    return {formats, std::move(pipelines)};
}

SilhouetteEdgePipeline::SilhouetteEdgePipeline(mtl::LibraryCache &libraries) : Renderer{CreateSilhouetteEdgeRenderer(libraries)} {}

SilhouetteEdgePipeline::ResourcesT::ResourcesT(const mtl::Context &ctx, mtl::Extent2D extent)
    : DepthImage{mtl::CreateTexture2D(ctx, Format::Depth, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      OffscreenImage{mtl::CreateTexture2D(ctx, Format::Float, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      ImageSampler{mtl::CreateSampler(ctx, MTL::SamplerMinMagFilterNearest, MTL::SamplerMipFilterNearest, MTL::SamplerAddressModeClampToEdge)} {}

void SilhouetteEdgePipeline::SetExtent(const mtl::Context &ctx, mtl::Extent2D extent) {
    Resources = std::make_unique<ResourcesT>(ctx, extent);
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
      OverlayJobLines{CreateMeshPipeline(libraries, FunctionRef{"SelectionFragment.metal", "SelectionFragment"}, SelectionFormats(), {}, DepthOff, {"OverlayJobLine.metal", "OverlayJobLineMesh"})},
      BoneSphere{CreateMeshPipeline(libraries, FunctionRef{"SelectionFragment.metal", "SelectionFragment"}, SelectionFormats(), {}, DepthOff, {"BoneSphere.metal", "BoneSphereMesh"})} {}

const mtl::MeshRenderPipeline &SelectionFragmentPipeline::ElementRaster(
    Element element, bool bitset_box, bool xray
) const {
    const auto &variants = element == Element::Face ? MeshletFaces : element == Element::Vertex ? MeshletVertices : MeshletEdges;
    return variants[uint32_t(bitset_box) + 2u * uint32_t(xray)];
}

Pipelines::Pipelines(const mtl::Context &ctx, mtl::LibraryCache &libraries)
    : Ctx(ctx),
      Libraries(libraries),
      Main{libraries},
      Silhouette{libraries},
      SilhouetteEdge{libraries},
      SelectionFragment{libraries},
      VisibilityObjectSelection{libraries, {"VisibilitySelection.metal", "VisibilityObjectSelectionKernel"}},
      PrepareEditSelection{libraries, {"EditSelectionTransaction.metal", "PrepareEditSelectionKernel"}},
      FillEditSelectionList{libraries, {"EditSelectionTransaction.metal", "FillEditSelectionListKernel"}},
      ResetEditSelectionSummary{libraries, {"EditSelectionTransaction.metal", "ResetEditSelectionSummaryKernel"}},
      DeriveEditSelection{libraries, {"EditSelectionTransaction.metal", "DeriveEditSelectionKernel"}},
      EditSharpness{libraries, {"EditSharpness.metal", "EditSharpnessKernel"}},
      CommitPosedGeometry{libraries, {"CommitPosedGeometry.metal", "CommitPosedGeometryKernel"}},
      PosePrepass{libraries, {"PosePrepass.metal", "PosePrepassKernel"}},
      PosedMeshletBounds{libraries, {"PosedMeshletBounds.metal", "PosedMeshletBoundsKernel"}},
      VertexNormalDerive{libraries, {"VertexNormalDerive.metal", "VertexNormalDeriveKernel"}},
      BoundsReduce{libraries, {"BoundsReduce.metal", "BoundsReduceKernel"}},
      BoundsCombine{libraries, {"BoundsCombine.metal", "BoundsCombineKernel"}},
      WireRaster{libraries, {"WireRaster.metal", "WireRasterKernel"}},
      LodFrontierCount{libraries, {"MeshletCull.metal", "LodFrontierCount"}},
      LodFrontierPrefix{libraries, {"MeshletCull.metal", "LodFrontierPrefix"}},
      LodFrontierEmit{libraries, {"MeshletCull.metal", "LodFrontierEmit"}},
      MeshletCullBlockCount{libraries, {"MeshletCull.metal", "MeshletCullBlockCount"}},
      MeshletCullPrefix{libraries, {"MeshletCull.metal", "MeshletCullPrefix"}},
      MeshletCullEmit{libraries, {"MeshletCull.metal", "MeshletCullEmit"}},
      MeshletPhase2Cull{libraries, {"MeshletCull.metal", "MeshletPhase2Cull"}},
      MeshletPhase2RangeCull{libraries, {"MeshletCull.metal", "MeshletPhase2RangeCull"}},
      MeshletPhase2Prefix{libraries, {"MeshletCull.metal", "MeshletPhase2Prefix"}},
      OverlayJobBlockCount{libraries, {"OverlayJobCull.metal", "OverlayJobBlockCount"}},
      OverlayJobPrefix{libraries, {"OverlayJobCull.metal", "OverlayJobPrefix"}},
      OverlayJobEmit{libraries, {"OverlayJobCull.metal", "OverlayJobEmit"}},
      DepthPyramidReduce{libraries, {"DepthPyramidReduce.metal", "DepthPyramidReduceKernel"}},
      MotionBlurTilesFlatten{libraries, {"MotionBlurTilesFlatten.metal", "MotionBlurTilesFlattenKernel"}},
      MotionBlurTilesDilate{libraries, {"MotionBlurTilesDilate.metal", "MotionBlurTilesDilateKernel"}},
      IblPrefilter{libraries},
      VertexAdjacency{libraries},
      VertexWeld{libraries},
      MeshConnectivity{libraries} {}

void Pipelines::SetExtent(mtl::Extent2D extent, mtl::BindlessSet &slots) {
    Main.SetExtent(Ctx, extent, slots);
    Silhouette.SetExtent(Ctx, extent);
    SilhouetteEdge.SetExtent(Ctx, extent);
}

void Pipelines::CompileShaders() {
    Libraries.Clear();
    Main.SceneRenderer.CompileShaders(Libraries);
    Main.OverlayRenderer.CompileShaders(Libraries);
    Main.SceneVelocityRenderer.CompileShaders(Libraries);
    Main.PrepassBackground.Compile(Libraries);
    Main.ViewportComposite.Compile(Libraries);
    Main.MotionBlurAccumulate.Compile(Libraries);
    Main.MotionBlurGather.Compile(Libraries);
    Main.WorkspaceVisibility.Compile(Libraries);
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
    Silhouette.Visibility.Compile(Libraries);
    SilhouetteEdge.Renderer.CompileShaders(Libraries);
    SelectionFragment.OverlayJobLines.Compile(Libraries);
    SelectionFragment.BoneSphere.Compile(Libraries);
    for (auto *variants : {&SelectionFragment.MeshletFaces, &SelectionFragment.MeshletVertices,
             &SelectionFragment.MeshletEdges}) {
        for (auto &pipeline : *variants) pipeline.Compile(Libraries);
    }
    for (auto *pipeline : {&SelectionFragment.MeshletFaceXRayPointsBitsetBox,
             &SelectionFragment.MeshletEdgeXRayPointsBitsetBox}) {
        pipeline->Compile(Libraries);
    }
    PrepareEditSelection.Compile(Libraries);
    FillEditSelectionList.Compile(Libraries);
    EditSharpness.Compile(Libraries);
    CommitPosedGeometry.Compile(Libraries);
    for (auto *compute : {&VisibilityObjectSelection, &ResetEditSelectionSummary, &DeriveEditSelection, &PosePrepass, &PosedMeshletBounds, &VertexNormalDerive, &BoundsReduce, &BoundsCombine, &WireRaster, &LodFrontierCount, &LodFrontierPrefix, &LodFrontierEmit, &MeshletCullBlockCount, &MeshletCullPrefix, &MeshletCullEmit, &MeshletPhase2Cull, &MeshletPhase2RangeCull, &MeshletPhase2Prefix, &OverlayJobBlockCount, &OverlayJobPrefix, &OverlayJobEmit, &DepthPyramidReduce, &MotionBlurTilesFlatten, &MotionBlurTilesDilate, &IblPrefilter.EquirectToCubemap, &IblPrefilter.DiffuseIrradiance, &IblPrefilter.SpecularPrefilter, &VertexAdjacency.Zero, &VertexAdjacency.Count, &VertexAdjacency.BlockSum, &VertexAdjacency.BlockPrefix, &VertexAdjacency.Offsets, &VertexAdjacency.Scatter, &VertexAdjacency.Sort, &VertexWeld.TableInit, &VertexWeld.Insert, &VertexWeld.MarkReps, &VertexWeld.BlockSum, &VertexWeld.BlockPrefix, &VertexWeld.Scan, &VertexWeld.Emit, &VertexWeld.Compact, &VertexWeld.WriteBack, &VertexWeld.RemapCorners, &MeshConnectivity.Zero, &MeshConnectivity.Count, &MeshConnectivity.BlockSum, &MeshConnectivity.BlockPrefix, &MeshConnectivity.Offsets, &MeshConnectivity.Scatter, &MeshConnectivity.Pair, &MeshConnectivity.Bits, &MeshConnectivity.WordBlockSum, &MeshConnectivity.WordBlockPrefix, &MeshConnectivity.Ranks, &MeshConnectivity.Samples}) {
        compute->Compile(Libraries);
    }
}
