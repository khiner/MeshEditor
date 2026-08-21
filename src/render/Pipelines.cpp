#include "render/Pipelines.h"
#include "gpu/BackgroundConstant.h"
#include "gpu/MeshVertexConstant.h"
#include "gpu/PbrConstant.h"
#include "metal/Bindless.h"
#include "metal/Buffer.h"
#include "render/Profile.h"

#include <array>
#include <bit>
#include <format>
#include <stdexcept>

using mtl::AdditiveBlend, mtl::Blend, mtl::NoBlend, mtl::NoWrite, mtl::PremultipliedBlend;
using mtl::BlendState, mtl::DepthState, mtl::FunctionConstant, mtl::FunctionRef, mtl::PassFormats, mtl::RenderPipeline;

namespace {
enum class OverlayKind : uint32_t {
    FaceNormal = 1,
    VertexNormal = 2,
};
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
constexpr mtl::FunctionConstant UintConstant(auto index, uint32_t value) {
    return {uint32_t(index), MTL::DataTypeUInt, value};
}

std::vector<mtl::FunctionConstant> MeshVertexConstants(uint32_t overlay_kind, bool line_draw, bool line_quad, bool velocity) {
    return {
        UintConstant(MeshVertexConstant::OverlayKind, overlay_kind),
        UintConstant(MeshVertexConstant::IsLineDraw, line_draw ? 1u : 0u),
        BoolConstant(MeshVertexConstant::LineQuad, line_quad),
        BoolConstant(MeshVertexConstant::VelocityOutput, velocity),
    };
}

FunctionRef MeshVertex(uint32_t overlay_kind = 0, bool line_draw = false, bool line_quad = false, bool velocity = false) {
    return {"VertexTransform.metal", "VertexTransformVertex", MeshVertexConstants(overlay_kind, line_draw, line_quad, velocity)};
}

FunctionRef MeshletVertex(bool velocity = false) {
    return {"MeshletTransform.metal", "MeshletTransformMesh", MeshVertexConstants(0, false, false, velocity)};
}

mtl::MeshRenderPipeline CreateMeshPipeline(
    mtl::LibraryCache &libraries, std::optional<FunctionRef> fragment, PassFormats formats,
    std::vector<BlendState> blends = {}, std::optional<DepthState> depth = {}
) {
    return {libraries, MeshletVertex(), std::move(fragment), std::move(formats), std::move(blends), depth};
}

struct PbrPipelineSpec {
    bool VelocityPass, VelocityOutput;
    FunctionRef Fragment;
    std::vector<BlendState> Blends;
    DepthState Depth;
};

PbrPipelineSpec MakePbrPipelineSpec(PbrFeatureMask mask, bool prepass, PbrCompiler::Variant variant, PbrCompiler::Topology topology) {
    const bool velocity_pass = variant == PbrCompiler::Variant::OpaqueVelocity || variant == PbrCompiler::Variant::BlendVelocity;
    const bool velocity_output = variant == PbrCompiler::Variant::OpaqueVelocity;
    std::vector<FunctionConstant> constants;
    constants.reserve(PbrSpecFeatures.size() + 3);
    for (const auto &[constant, feature] : PbrSpecFeatures) constants.push_back(BoolConstant(constant, HasFeature(mask, feature)));
    constants.push_back(BoolConstant(PbrConstant::TransmissionPrepass, prepass));
    constants.push_back(UintConstant(PbrConstant::Topology, uint32_t(topology)));
    constants.push_back(BoolConstant(PbrConstant::VelocityOutput, velocity_output));
    std::vector<BlendState> blends{Blend};
    if (velocity_pass) blends.push_back(velocity_output ? NoBlend : NoWrite);
    return {velocity_pass, velocity_output, {"pbr.metal", "PbrFragment", std::move(constants)}, std::move(blends), {.Write = variant != PbrCompiler::Variant::Blend && variant != PbrCompiler::Variant::BlendVelocity}};
}

PassFormats SceneFormats() { return {{Format::HdrColor}, Format::Depth}; }
PassFormats SceneVelocityFormats() { return {{Format::HdrColor, Format::Velocity}, Format::Depth}; }
PassFormats OverlayFormats() { return {{Format::Color, Format::LineData}, Format::Depth}; }
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
    pipelines.emplace(SPT::Fill, RenderPipeline{libraries, MeshVertex(), FunctionRef{"WorkspaceLighting.metal", "WorkspaceLightingFragment"}, formats, {Blend}, DepthTestWrite});
    pipelines.emplace(SPT::FillDepth, RenderPipeline{libraries, MeshVertex(), std::nullopt, formats, {NoWrite}, DepthTestWrite});
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
    pipelines.emplace(SPT::EdgeQuad, RenderPipeline{libraries, {"EdgeQuad.metal", "EdgeQuadVertex"}, FunctionRef{"EdgeQuad.metal", "EdgeQuadFragment"}, formats, {Blend, NoWrite}, DepthTestNoWriteLessEqual});
    pipelines.emplace(SPT::Line, RenderPipeline{libraries, MeshVertex(0, true), vertex_color, formats, {Blend, NoBlend}, DepthTestLessEqual});
    pipelines.emplace(SPT::ObjectExtrasLine, RenderPipeline{libraries, {"ObjectExtras.metal", "ObjectExtrasVertex"}, vertex_color, formats, {Blend, NoBlend}, DepthTestLessEqual});
    pipelines.emplace(SPT::BoundsBox, RenderPipeline{libraries, {"BoundsBox.metal", "BoundsBoxVertex"}, vertex_color, formats, {Blend, NoBlend}, DepthTestLessEqual});
    const auto make_overlay_pipeline = [&](OverlayKind overlay_kind) {
        return RenderPipeline{libraries, MeshVertex(uint32_t(overlay_kind), true), vertex_color, formats, {Blend, NoBlend}, DepthTestLessEqual};
    };
    pipelines.emplace(SPT::LineOverlayFaceNormals, make_overlay_pipeline(OverlayKind::FaceNormal));
    pipelines.emplace(SPT::LineOverlayVertexNormals, make_overlay_pipeline(OverlayKind::VertexNormal));
    pipelines.emplace(SPT::Point, RenderPipeline{libraries, {"VertexPoint.metal", "VertexPointVertex"}, FunctionRef{"VertexPoint.metal", "VertexPointFragment"}, formats, {Blend, NoWrite}, DepthTestLessEqual});
    pipelines.emplace(SPT::Grid, RenderPipeline{libraries, {"GridLines.metal", "GridLinesVertex"}, FunctionRef{"GridLines.metal", "GridLinesFragment"}, formats, {Blend, NoWrite}, DepthState{.Write = false}});
    pipelines.emplace(SPT::SilhouetteEdgeColor, RenderPipeline{libraries, {"TexQuad.metal", "TexQuadVertex"}, FunctionRef{"SilhouetteEdgeColor.metal", "SilhouetteEdgeColorFragment"}, formats, {Blend, NoWrite}, DepthOff});
    pipelines.emplace(SPT::BoneFill, RenderPipeline{libraries, {"BoneSolid.metal", "BoneSolidVertex"}, FunctionRef{"BoneSolid.metal", "BoneSolidFragment"}, formats, {Blend, NoWrite}, DepthTestWrite, 2.f});
    pipelines.emplace(SPT::BoneWire, RenderPipeline{libraries, {"BoneWire.metal", "BoneWireVertex"}, vertex_color, formats, {Blend, NoBlend}, DepthTestNoWriteLessEqual});
    pipelines.emplace(SPT::BoneSphereFill, RenderPipeline{libraries, {"BoneSphere.metal", "BoneSphereVertex"}, FunctionRef{"BoneSphere.metal", "BoneSphereFragment"}, formats, {Blend, NoWrite}, DepthTestLessEqual});
    pipelines.emplace(SPT::BoneSphereWire, RenderPipeline{libraries, {"BoneSphereWire.metal", "BoneSphereWireVertex"}, vertex_color, formats, {Blend, NoBlend}, DepthTestNoWriteLessEqual});
    return {formats, std::move(pipelines)};
}

PbrCompiler::PbrCompiler(PassFormats scene, PassFormats scene_velocity)
    : SceneFormats(std::move(scene)), VelocityFormats(std::move(scene_velocity)) {}

std::unique_ptr<RenderPipeline> PbrCompiler::CreateTargetedPipeline(
    mtl::LibraryCache &libraries, PbrFeatureMask mask, bool prepass, Variant variant, Topology topology
) const {
    // Opaque geometry writes its screen motion into the velocity attachment. Blend geometry
    // writes neither depth nor velocity, so its velocity twin masks the attachment off.
    auto spec = MakePbrPipelineSpec(mask, prepass, variant, topology);
    return std::make_unique<RenderPipeline>(
        libraries, MeshVertex(0, false, topology == Topology::Line, spec.VelocityOutput), std::move(spec.Fragment),
        spec.VelocityPass ? VelocityFormats : SceneFormats, std::move(spec.Blends), spec.Depth
    );
}

std::unique_ptr<mtl::MeshRenderPipeline> PbrCompiler::CreateMeshletPipeline(
    mtl::LibraryCache &libraries, PbrFeatureMask mask, bool prepass, Variant variant
) const {
    auto spec = MakePbrPipelineSpec(mask, prepass, variant, Topology::Triangle);
    return std::make_unique<mtl::MeshRenderPipeline>(
        libraries, MeshletVertex(spec.VelocityOutput), std::move(spec.Fragment),
        spec.VelocityPass ? VelocityFormats : SceneFormats, std::move(spec.Blends), spec.Depth
    );
}

bool PbrCompiler::CompilePipelines(mtl::LibraryCache &libraries, PbrFeatureMask mask, bool non_triangle) {
    if (mask == Mask && non_triangle == NonTriangle && Variants[VariantIndex(Topology::Triangle, Variant::Opaque)] && Variants[VariantIndex(Topology::Triangle, Variant::Blend)]) return false;
    const profile::CpuScope scope{"CompilePbrPipelines"};

    const bool transmission = ::HasFeature(mask, PbrFeature::Transmission);
    for (size_t t = 0; t < TopologyCount; ++t) {
        const auto topology = Topology(t);
        for (size_t v = 0; v < VariantCount; ++v) Variants[VariantIndex(topology, Variant(v))].reset();
        const bool triangle = topology == Topology::Triangle;
        // Line and point topologies build only when the scene holds such meshes, and only in the variants their draws bind.
        if (!triangle && !non_triangle) continue;
        for (const auto v : {Variant::Opaque, Variant::OpaqueVelocity}) Variants[VariantIndex(topology, v)] = CreateTargetedPipeline(libraries, mask, false, v, topology);
        if (!triangle) continue;
        for (const auto v : {Variant::Blend, Variant::BlendVelocity}) Variants[VariantIndex(topology, v)] = CreateTargetedPipeline(libraries, mask, false, v, topology);
        if (transmission) Variants[VariantIndex(topology, Variant::OpaquePrepass)] = CreateTargetedPipeline(libraries, mask, true, Variant::OpaquePrepass, topology);
    }
    for (size_t v = 0; v < VariantCount; ++v) {
        const auto variant = Variant(v);
        if (variant == Variant::OpaquePrepass && !transmission) {
            MeshletVariants[v].reset();
        } else {
            MeshletVariants[v] = CreateMeshletPipeline(libraries, mask, variant == Variant::OpaquePrepass, variant);
        }
    }
    Mask = mask;
    NonTriangle = non_triangle;
    return true;
}

void PbrCompiler::Bind(MTL::RenderCommandEncoder *encoder, Variant v, Topology t) const {
    const auto &pipeline = Variants[VariantIndex(t, v)];
    if (!pipeline) throw std::runtime_error("PbrCompiler: binding a variant that was never compiled.");
    pipeline->Bind(encoder);
}

void PbrCompiler::BindMeshlets(MTL::RenderCommandEncoder *encoder, Variant variant) const {
    const auto &pipeline = MeshletVariants[size_t(variant)];
    if (!pipeline) throw std::runtime_error("PbrCompiler: binding a meshlet variant that was never compiled.");
    pipeline->Bind(encoder);
}

void PbrCompiler::RecompileModules(mtl::LibraryCache &libraries) {
    for (auto &variant : Variants) {
        if (variant) variant->Compile(libraries);
    }
    for (auto &variant : MeshletVariants) {
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
      MeshletFill{CreateMeshPipeline(libraries, FunctionRef{"WorkspaceLighting.metal", "WorkspaceLightingFragment"}, SceneFormats(), {Blend}, DepthTestWrite)},
      MeshletDepth{CreateMeshPipeline(libraries, FunctionRef{"DepthOnly.metal", "DepthOnlyFragment"}, SceneFormats(), {NoWrite}, DepthTestWrite)},
      Compiler{SceneFormats(), SceneVelocityFormats()} {}

MainPipeline::ResourcesT::ResourcesT(const mtl::Context &ctx, mtl::Extent2D extent, mtl::BindlessSet &slots)
    // Depth is sampled as well as attached: the motion blur gather reads it to sort samples.
    : DepthImage{mtl::CreateTexture2D(ctx, Format::Depth, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
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

static PipelineRenderer CreateSilhouetteRenderer(mtl::LibraryCache &libraries) {
    // Depth is stored for reuse by element selection (mutual occlusion between selected meshes).
    const PassFormats formats{{Format::Float2}, Format::Depth};
    std::unordered_map<SPT, RenderPipeline> pipelines;
    pipelines.emplace(SPT::SilhouetteDepthObject, RenderPipeline{libraries, {"PositionTransform.metal", "PositionTransformVertex"}, FunctionRef{"DepthObject.metal", "DepthObjectFragment"}, formats, {NoBlend}, DepthTestWrite});
    return {formats, std::move(pipelines)};
}

SilhouettePipeline::SilhouettePipeline(mtl::LibraryCache &libraries)
    : Renderer{CreateSilhouetteRenderer(libraries)},
      Meshlet{CreateMeshPipeline(libraries, FunctionRef{"DepthObject.metal", "DepthObjectFragment"}, Renderer.Formats, {NoBlend}, DepthTestWrite)} {}

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

static PipelineRenderer CreateSelectionFragmentRenderer(mtl::LibraryCache &libraries) {
    const PassFormats formats{{}, Format::Depth};
    struct Desc {
        SPT Type;
        const char *VertexName;
        const char *FragmentFile;
        const char *FragmentName;
        DepthState Depth;
    };
    static constexpr auto LinkedList = "SelectionElementLinkedList.metal";
    static constexpr auto LinkedListFn = "SelectionElementLinkedListFragment";
    static constexpr auto BitsetBox = "SelectionElementBitsetBox.metal";
    static constexpr auto BitsetBoxFn = "SelectionElementBitsetBoxFragment";
    constexpr DepthState DepthLess{.Compare = MTL::CompareFunctionLess};
    const std::array selection_element_pipelines{
        Desc{SPT::SelectionElementFace, "SelectionElementFaceVertex", LinkedList, LinkedListFn, DepthLess},
        Desc{SPT::SelectionElementFaceBitsetBox, "SelectionElementFaceVertex", BitsetBox, BitsetBoxFn, DepthLess},
        Desc{SPT::SelectionElementFaceXRay, "SelectionElementFaceVertex", LinkedList, LinkedListFn, DepthOff},
        Desc{SPT::SelectionElementFaceXRayBitsetBox, "SelectionElementFaceVertex", BitsetBox, BitsetBoxFn, DepthOff},
        Desc{SPT::SelectionElementEdge, "SelectionElementEdgeVertex", LinkedList, LinkedListFn, DepthTestNoWriteLessEqual},
        Desc{SPT::SelectionElementEdgeBitsetBox, "SelectionElementEdgeVertex", BitsetBox, BitsetBoxFn, DepthTestNoWriteLessEqual},
        Desc{SPT::SelectionElementEdgeXRay, "SelectionElementEdgeVertex", LinkedList, LinkedListFn, DepthOff},
        Desc{SPT::SelectionElementEdgeXRayBitsetBox, "SelectionElementEdgeVertex", BitsetBox, BitsetBoxFn, DepthOff},
        Desc{SPT::SelectionElementEdgeXRayVertsBitsetBox, "SelectionElementEdgeVertex", BitsetBox, BitsetBoxFn, DepthOff},
        Desc{SPT::SelectionElementVertex, "SelectionElementVertexVertex", LinkedList, LinkedListFn, DepthTestNoWriteLessEqual},
        Desc{SPT::SelectionElementVertexBitsetBox, "SelectionElementVertexVertex", BitsetBox, BitsetBoxFn, DepthTestNoWriteLessEqual},
        Desc{SPT::SelectionElementVertexXRay, "SelectionElementVertexVertex", LinkedList, LinkedListFn, DepthOff},
        Desc{SPT::SelectionElementVertexXRayBitsetBox, "SelectionElementVertexVertex", BitsetBox, BitsetBoxFn, DepthOff},
        Desc{SPT::SelectionElementFaceXRayVertsBitsetBox, "SelectionElementFaceVertex", BitsetBox, BitsetBoxFn, DepthOff},
    };
    std::unordered_map<SPT, RenderPipeline> pipelines;
    for (const auto &desc : selection_element_pipelines) {
        pipelines.emplace(desc.Type, RenderPipeline{libraries, {"SelectionElement.metal", desc.VertexName}, FunctionRef{desc.FragmentFile, desc.FragmentName}, formats, {}, desc.Depth});
    }
    const FunctionRef selection_fragment{"SelectionFragment.metal", "SelectionFragment"};
    pipelines.emplace(SPT::SelectionFragment, RenderPipeline{libraries, {"PositionTransform.metal", "PositionTransformVertex"}, selection_fragment, formats, {}, DepthOff});
    pipelines.emplace(SPT::SelectionFragmentBoneSphere, RenderPipeline{libraries, {"BoneSphere.metal", "BoneSphereVertex"}, selection_fragment, formats, {}, DepthOff});
    pipelines.emplace(SPT::SelectionObjectExtrasLines, RenderPipeline{libraries, {"ObjectExtras.metal", "ObjectExtrasSelectionVertex"}, selection_fragment, formats, {}, DepthOff});
    return {formats, std::move(pipelines)};
}

SelectionFragmentPipeline::SelectionFragmentPipeline(mtl::LibraryCache &libraries)
    : Renderer{CreateSelectionFragmentRenderer(libraries)},
      MeshletObject{CreateMeshPipeline(libraries, FunctionRef{"SelectionFragment.metal", "SelectionFragment"}, Renderer.Formats, {}, DepthOff)},
      MeshletFace{CreateMeshPipeline(libraries, FunctionRef{"SelectionElementLinkedList.metal", "SelectionElementLinkedListFragment"}, Renderer.Formats, {}, DepthState{.Compare = MTL::CompareFunctionLess})},
      MeshletFaceXRay{CreateMeshPipeline(libraries, FunctionRef{"SelectionElementLinkedList.metal", "SelectionElementLinkedListFragment"}, Renderer.Formats, {}, DepthOff)},
      MeshletFaceBitsetBox{CreateMeshPipeline(libraries, FunctionRef{"SelectionElementBitsetBox.metal", "SelectionElementBitsetBoxFragment"}, Renderer.Formats, {}, DepthState{.Compare = MTL::CompareFunctionLess})},
      MeshletFaceXRayBitsetBox{CreateMeshPipeline(libraries, FunctionRef{"SelectionElementBitsetBox.metal", "SelectionElementBitsetBoxFragment"}, Renderer.Formats, {}, DepthOff)} {}

SelectionFragmentPipeline::ResourcesT::ResourcesT(mtl::BufferContext &buffers, mtl::Extent2D extent)
    : HeadBuffer{buffers, uint64_t(extent.Width) * extent.Height * sizeof(uint32_t), SlotType::Buffer}, Extent{extent} {}

void SelectionFragmentPipeline::SetExtent(mtl::BufferContext &buffers, mtl::Extent2D extent) {
    Resources = std::make_unique<ResourcesT>(buffers, extent);
}

Pipelines::Pipelines(const mtl::Context &ctx, mtl::LibraryCache &libraries)
    : Ctx(ctx),
      Libraries(libraries),
      Main{libraries},
      Silhouette{libraries},
      SilhouetteEdge{libraries},
      SelectionFragment{libraries},
      ObjectPick{libraries, {"ObjectPick.metal", "ObjectPickKernel"}},
      ElementPick{libraries, {"ElementPick.metal", "ElementPickKernel"}},
      BoxSelect{libraries, {"BoxSelect.metal", "BoxSelectKernel"}},
      UpdateSelectionState{libraries, {"UpdateSelectionState.metal", "UpdateSelectionStateKernel"}},
      PosePrepass{libraries, {"PosePrepass.metal", "PosePrepassKernel"}},
      VertexNormalDerive{libraries, {"VertexNormalDerive.metal", "VertexNormalDeriveKernel"}},
      BoundsReduce{libraries, {"BoundsReduce.metal", "BoundsReduceKernel"}},
      BoundsCombine{libraries, {"BoundsCombine.metal", "BoundsCombineKernel"}},
      FrustumCull{libraries, {"FrustumCull.metal", "FrustumCullKernel"}},
      MeshletWorkBlockCount{libraries, {"MeshletCull.metal", "MeshletWorkBlockCount"}},
      MeshletWorkPrefix{libraries, {"MeshletCull.metal", "MeshletWorkPrefix"}},
      MeshletWorkEmit{libraries, {"MeshletCull.metal", "MeshletWorkEmit"}},
      MeshletCullBlockCount{libraries, {"MeshletCull.metal", "MeshletCullBlockCount"}},
      MeshletCullPrefix{libraries, {"MeshletCull.metal", "MeshletCullPrefix"}},
      MeshletCullEmit{libraries, {"MeshletCull.metal", "MeshletCullEmit"}},
      DepthPyramidReduce{libraries, {"DepthPyramidReduce.metal", "DepthPyramidReduceKernel"}},
      MotionBlurTilesFlatten{libraries, {"MotionBlurTilesFlatten.metal", "MotionBlurTilesFlattenKernel"}},
      MotionBlurTilesDilate{libraries, {"MotionBlurTilesDilate.metal", "MotionBlurTilesDilateKernel"}},
      IblPrefilter{libraries} {}

void Pipelines::SetExtent(mtl::Extent2D extent, mtl::BufferContext &buffers, mtl::BindlessSet &slots) {
    Main.SetExtent(Ctx, extent, slots);
    Silhouette.SetExtent(Ctx, extent);
    SilhouetteEdge.SetExtent(Ctx, extent);
    SelectionFragment.SetExtent(buffers, extent);
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
    Main.MeshletFill.Compile(Libraries);
    Main.MeshletDepth.Compile(Libraries);
    Main.Compiler.RecompileModules(Libraries);
    Silhouette.Renderer.CompileShaders(Libraries);
    Silhouette.Meshlet.Compile(Libraries);
    SilhouetteEdge.Renderer.CompileShaders(Libraries);
    SelectionFragment.Renderer.CompileShaders(Libraries);
    SelectionFragment.MeshletObject.Compile(Libraries);
    SelectionFragment.MeshletFace.Compile(Libraries);
    SelectionFragment.MeshletFaceXRay.Compile(Libraries);
    SelectionFragment.MeshletFaceBitsetBox.Compile(Libraries);
    SelectionFragment.MeshletFaceXRayBitsetBox.Compile(Libraries);
    for (auto *compute : {&ObjectPick, &ElementPick, &BoxSelect, &UpdateSelectionState, &PosePrepass, &VertexNormalDerive, &BoundsReduce, &BoundsCombine, &FrustumCull, &MeshletWorkBlockCount, &MeshletWorkPrefix, &MeshletWorkEmit, &MeshletCullBlockCount, &MeshletCullPrefix, &MeshletCullEmit, &DepthPyramidReduce, &MotionBlurTilesFlatten, &MotionBlurTilesDilate, &IblPrefilter.EquirectToCubemap, &IblPrefilter.DiffuseIrradiance, &IblPrefilter.SpecularPrefilter}) {
        compute->Compile(Libraries);
    }
}
