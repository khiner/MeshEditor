#include "render/Pipelines.h"
#include "gpu/BoxSelectPushConstants.h"
#include "gpu/DepthPyramidReducePushConstants.h"
#include "gpu/ElementPickPushConstants.h"
#include "gpu/FrustumCullPushConstants.h"
#include "gpu/MainDrawPushConstants.h"
#include "gpu/MotionBlurGatherPushConstants.h"
#include "gpu/MotionBlurTilesDilatePushConstants.h"
#include "gpu/MotionBlurTilesFlattenPushConstants.h"
#include "gpu/ObjectPickPushConstants.h"
#include "gpu/SelectionDrawPushConstants.h"
#include "gpu/SelectionElementPushConstants.h"
#include "gpu/UpdateSelectionStatePushConstants.h"
#include "render/Bindless.h"
#include "render/Profile.h"

#include <ranges>

namespace {
using enum vk::PrimitiveTopology;
using enum vk::ImageUsageFlagBits;
constexpr auto Vert = ShaderType::eVertex, Frag = ShaderType::eFragment;

enum class OverlayKind : uint32_t {
    FaceNormal = 1,
    VertexNormal = 2,
};
constexpr std::array PbrSpecFeatures{PbrFeature::Punctual, PbrFeature::Transmission, PbrFeature::DiffuseTrans, PbrFeature::Clearcoat, PbrFeature::Sheen, PbrFeature::Anisotropy, PbrFeature::Iridescence};

constexpr vk::PushConstantRange MainDrawPushConstantRange{Vert | Frag, 0, sizeof(MainDrawPushConstants)};
constexpr vk::PushConstantRange FragPc(uint32_t size) { return {Frag, 0, size}; }

// Straight-alpha source over an associated destination, leaving the target premultiplied.
constexpr auto Blend = CreateColorBlendAttachment(true);
// Write-through, no blending.
constexpr auto NoBlend = CreateColorBlendAttachment(false);
// Writes masked off, for attachments a pipeline leaves untouched.
constexpr vk::PipelineColorBlendAttachmentState NoWrite{};

// Motion blur reduces the frame to tiles of this many pixels a side. The flatten pass runs one
// workgroup per tile, so its local size must match.
constexpr uint32_t MotionBlurTileSize = 32;

constexpr vk::Extent2D DivideCeil(vk::Extent2D extent, uint32_t d) {
    return {(extent.width + d - 1) / d, (extent.height + d - 1) / d};
}

constexpr vk::SubpassDependency ExternalFragReadDependency() {
    return {
        0,
        vk::SubpassExternal,
        vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests,
        vk::PipelineStageFlagBits::eFragmentShader,
        vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits::eShaderRead,
        {},
    };
}

// Ordering for passes whose depth tests read a prior pass's depth writes, plus the shared frag-read dependency.
std::array<vk::SubpassDependency, 2> DepthLoadDependencies() {
    return {
        vk::SubpassDependency{
            vk::SubpassExternal,
            0,
            vk::PipelineStageFlagBits::eLateFragmentTests,
            vk::PipelineStageFlagBits::eEarlyFragmentTests,
            vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            {},
        },
        ExternalFragReadDependency(),
    };
}

// Render pass with a single graphics subpass. Depth (when `has_depth`) is attachment 0; colors follow in order.
vk::UniqueRenderPass CreateRenderPass(vk::Device d, std::span<const vk::AttachmentDescription> attachments, bool has_depth, std::span<const vk::SubpassDependency> dependencies = {}) {
    static constexpr vk::AttachmentReference DepthRef{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
    std::vector<vk::AttachmentReference> color_refs;
    for (uint32_t i = has_depth ? 1u : 0u; i < attachments.size(); ++i) color_refs.emplace_back(i, vk::ImageLayout::eColorAttachmentOptimal);
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, uint32_t(color_refs.size()), color_refs.data(), nullptr, has_depth ? &DepthRef : nullptr};
    return d.createRenderPassUnique({{}, uint32_t(attachments.size()), attachments.data(), 1, &subpass, uint32_t(dependencies.size()), dependencies.data()});
}

} // namespace

void PipelineRenderer::CompileShaders() {
    for (auto &shader_pipeline : std::views::values(ShaderPipelines)) shader_pipeline.Compile(*RenderPass);
}

const ShaderPipeline &PipelineRenderer::Bind(vk::CommandBuffer cb, SPT spt, uint32_t scene_ubo_offset) const {
    const auto &pipeline = ShaderPipelines.at(spt);
    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline.Pipeline);
    const std::array sets{pipeline.GetDescriptorSet(), pipeline.GetUboSet()};
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline.PipelineLayout, 0, uint32_t(sets.size()), sets.data(), 1, &scene_ubo_offset);
    return pipeline;
}

PbrCompiler::PbrCompiler(PipelineContext ctx, vk::RenderPass scene, vk::RenderPass scene_velocity)
    : Device(ctx.Device), Cache(Device.createPipelineCacheUnique({})), SetLayout(ctx.SharedLayout), Set(ctx.SharedSet), UboSetLayout(ctx.UboLayout), UboSet(ctx.UboSet), RenderPass(scene), VelocityRenderPass(scene_velocity) {
    CompileModules();
    const std::array set_layouts{SetLayout, UboSetLayout};
    Layout = Device.createPipelineLayoutUnique({{}, uint32_t(set_layouts.size()), set_layouts.data(), 1, &MainDrawPushConstantRange});
}

void PbrCompiler::CompileModules() {
    VertModule = CompileShaderModule(Device, Vert, "VertexTransform.vert");
    FragModule = CompileShaderModule(Device, Frag, "pbr.frag");
    VelocityVertModule = CompileShaderModule(Device, Vert, "VertexTransform.vert", {"VELOCITY_OUTPUT"});
    VelocityFragModule = CompileShaderModule(Device, Frag, "pbr.frag", {"VELOCITY_OUTPUT"});
}

vk::UniquePipeline PbrCompiler::CreateTargetedPipeline(const vk::SpecializationInfo &frag_spec, Variant variant) const {
    static constexpr vk::PipelineViewportStateCreateInfo viewport_state{{}, 1, nullptr, 1, nullptr};
    static constexpr std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    static const vk::PipelineDynamicStateCreateInfo dynamic_state{{}, dynamic_states};
    static constexpr vk::PipelineMultisampleStateCreateInfo multisample_state{{}, vk::SampleCountFlagBits::e1};
    static constexpr vk::PipelineVertexInputStateCreateInfo vertex_input{};
    static constexpr vk::PipelineInputAssemblyStateCreateInfo input_assembly{{}, eTriangleList};
    static const vk::PipelineRasterizationStateCreateInfo raster{{}, false, false, vk::PolygonMode::eFill, {}, vk::FrontFace::eClockwise, false, 0.f, {}, 0.f, 1.f};

    // Opaque geometry writes its screen motion into the velocity attachment. Blend geometry
    // writes neither depth nor velocity, so its velocity twin masks the attachment off.
    const bool velocity_pass = variant == Variant::OpaqueVelocity || variant == Variant::BlendVelocity;
    const bool velocity_modules = variant == Variant::OpaqueVelocity;
    const bool depth_write = variant != Variant::Blend && variant != Variant::BlendVelocity;
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{{}, Vert, velocity_modules ? *VelocityVertModule : *VertModule, "main"},
        vk::PipelineShaderStageCreateInfo{{}, Frag, velocity_modules ? *VelocityFragModule : *FragModule, "main", &frag_spec},
    };
    const auto depth_stencil = CreateDepthStencil(true, depth_write);
    const std::array color_blend_attachments{
        Blend,
        velocity_modules ? NoBlend : NoWrite,
    };
    const vk::PipelineColorBlendStateCreateInfo color_blending{{}, false, vk::LogicOp::eCopy, velocity_pass ? 2u : 1u, color_blend_attachments.data()};
    auto result = Device.createGraphicsPipelineUnique(
        *Cache,
        {{}, stages, &vertex_input, &input_assembly, nullptr, &viewport_state, &raster, &multisample_state, &depth_stencil, &color_blending, &dynamic_state, *Layout, velocity_pass ? VelocityRenderPass : RenderPass}
    );
    if (result.result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("PbrCompiler: failed to create targeted pipeline: {}", vk::to_string(result.result)));
    }
    return std::move(result.value);
}

bool PbrCompiler::CompilePipelines(PbrFeatureMask mask) {
    if (mask == Mask && Variants[size_t(Variant::Opaque)] && Variants[size_t(Variant::Blend)]) return false;
    const profile::CpuScope scope{"CompilePbrPipelines"};

    constexpr uint32_t N = PbrSpecFeatures.size();
    constexpr uint32_t TotalConstants = N + 1; // + TRANSMISSION_PREPASS
    std::array<uint32_t, TotalConstants> data{};
    std::array<vk::SpecializationMapEntry, TotalConstants> entries{};
    for (uint32_t i = 0; i < TotalConstants; ++i) entries[i] = vk::SpecializationMapEntry{i, i * uint32_t(sizeof(uint32_t)), uint32_t(sizeof(uint32_t))};
    for (uint32_t i = 0; i < N; ++i) data[i] = ::HasFeature(mask, PbrSpecFeatures[i]) ? 1u : 0u;
    data[N] = 0u; // main pipelines: exposed radiance, sampling the transmission framebuffer
    const vk::SpecializationInfo spec_main{TotalConstants, entries.data(), TotalConstants * sizeof(uint32_t), data.data()};
    for (const auto v : {Variant::Opaque, Variant::Blend, Variant::OpaqueVelocity, Variant::BlendVelocity}) Variants[size_t(v)] = CreateTargetedPipeline(spec_main, v);
    if (::HasFeature(mask, PbrFeature::Transmission)) {
        data[N] = 1u; // pre-pass pipeline: un-exposed radiance, no framebuffer self-sampling
        const vk::SpecializationInfo spec_prepass{TotalConstants, entries.data(), TotalConstants * sizeof(uint32_t), data.data()};
        Variants[size_t(Variant::OpaquePrepass)] = CreateTargetedPipeline(spec_prepass, Variant::OpaquePrepass);
    } else {
        Variants[size_t(Variant::OpaquePrepass)].reset();
    }
    Mask = mask;

    return true;
}

vk::PipelineLayout PbrCompiler::Bind(vk::CommandBuffer cb, Variant v, uint32_t scene_ubo_offset) const {
    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *Variants[size_t(v)]);
    const std::array sets{Set, UboSet};
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *Layout, 0, uint32_t(sets.size()), sets.data(), 1, &scene_ubo_offset);
    return *Layout;
}

void PbrCompiler::RecompileModules() {
    if (!Device) return;
    CompileModules();
    for (auto &pipeline : Variants) pipeline.reset();
    CompilePipelines(Mask);
}

// `transmission_prepass` selects the variant that skips exposure, which the main pass applies after sampling.
static ShaderPipeline CreateBackgroundPipeline(const PipelineContext &ctx, bool transmission_prepass) {
    return ctx.CreateGraphics({{{Vert, "Background.vert"}, {Frag, "Background.frag", {{0, transmission_prepass ? 1u : 0u}}}}}, eTriangleStrip, {Blend}, CreateDepthStencil(false, false));
}

// Fullscreen textured quad into a single color attachment: no depth, fragment-only push constants.
static ShaderPipeline CreateQuadPipeline(const PipelineContext &ctx, std::filesystem::path frag, vk::PipelineColorBlendAttachmentState blend, uint32_t push_constant_size) {
    return ctx.CreateGraphics({{{Vert, "TexQuad.vert"}, {Frag, std::move(frag)}}}, eTriangleStrip, {blend}, {}, FragPc(push_constant_size));
}

// Depth + a single scene-linear color attachment. Depth is stored for the overlay pass to load and
// occlude against. Also backs the transmission pre-pass, which renders into its own framebuffer.
// `load_depth` keeps the depth the transmission pre-pass wrote, for the composite path.
// Attachment compatibility ignores load ops, so the plain pass's pipelines and framebuffer serve both variants.
static vk::UniqueRenderPass CreateSceneRenderPass(vk::Device d, bool load_depth = false, bool load_color = false) {
    const std::vector<vk::AttachmentDescription> attachments{
        {{}, Format::Depth, vk::SampleCountFlagBits::e1, load_depth ? vk::AttachmentLoadOp::eLoad : vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, load_depth ? vk::ImageLayout::eDepthStencilAttachmentOptimal : vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
        // Color: cleared transparent, stored for sampling in the viewport composite. The load
        // variant resumes over the shader-readable color the prior scene pass part stored.
        {{}, Format::HdrColor, vk::SampleCountFlagBits::e1, load_color ? vk::AttachmentLoadOp::eLoad : vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, load_color ? vk::ImageLayout::eShaderReadOnlyOptimal : vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    // The depth dependency orders the load variant's tests after the pre-pass's depth writes.
    // Both variants carry it so they stay render-pass compatible (compatibility requires equal
    // dependency counts), and it is pure ordering for the clear variant.
    return CreateRenderPass(d, attachments, true, DepthLoadDependencies());
}

// The scene's depth loaded for occlusion, a display-referred overlay color target over transparent,
// and the line data driving overlay AA. Both color targets are sampled by the viewport composite.
static vk::UniqueRenderPass CreateOverlayRenderPass(vk::Device d) {
    const std::vector<vk::AttachmentDescription> attachments{
        {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal},
        // Overlay color: cleared transparent so the composite can merge it over the scene.
        {{}, Format::Color, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        // LineData: cleared to zero (alpha=0 means "no line here").
        {{}, Format::LineData, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    // The scene pass's depth writes must land before this pass tests against them.
    return CreateRenderPass(d, attachments, true, DepthLoadDependencies());
}

// The scene render pass plus a velocity attachment the geometry writes its screen motion into.
// Motion blur steps render through this instead of the plain scene pass.
static vk::UniqueRenderPass CreateSceneVelocityRenderPass(vk::Device d) {
    const std::vector<vk::AttachmentDescription> attachments{
        {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
        // Color: cleared transparent, stored for sampling in the gather and the viewport composite.
        {{}, Format::HdrColor, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        // Velocity: cleared to zero, which reads as static wherever nothing draws.
        {{}, Format::Velocity, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const std::array dependencies{ExternalFragReadDependency()};
    return CreateRenderPass(d, attachments, true, dependencies);
}

// Fullscreen-quad twins for the scene+velocity render pass, keyed by the same SPTs the plain scene
// pass binds. PBR geometry goes through PbrCompiler's velocity variants instead.
static PipelineRenderer CreateSceneVelocityRenderer(const PipelineContext &ctx) {
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    // Screen motion for every pixel geometry leaves uncovered. Drawn first, so geometry overwrites
    // it wherever it lands, and the scene color stays untouched through the write mask.
    pipelines.emplace(SPT::BackgroundVelocity, ctx.CreateGraphics({{{Vert, "Background.vert"}, {Frag, "BackgroundVelocity.frag"}}}, eTriangleStrip, {NoWrite, NoBlend}, CreateDepthStencil(false, false)));
    // The environment background writes color only: its motion comes from the quad above.
    pipelines.emplace(SPT::Background, ctx.CreateGraphics({{{Vert, "Background.vert"}, {Frag, "Background.frag", {{0, 0u}}}}}, eTriangleStrip, {Blend, NoWrite}, CreateDepthStencil(false, false)));
    // Depth-only silhouette edge quad (see the scene renderer's twin for the depth-test rationale).
    pipelines.emplace(SPT::SilhouetteEdgeDepth, ctx.CreateGraphics({{{Vert, "TexQuad.vert"}, {Frag, "SampleDepth.frag"}}}, eTriangleStrip, {NoWrite, NoWrite}, CreateDepthStencil(true, true), FragPc(sizeof(uint32_t))));
    return {CreateSceneVelocityRenderPass(ctx.Device), std::move(pipelines)};
}

// One color attachment, no depth, always stored. Backs each of the fullscreen-quad passes.
static vk::UniqueRenderPass CreateColorOnlyRenderPass(vk::Device d, vk::Format format, vk::AttachmentLoadOp load, vk::ImageLayout initial, vk::ImageLayout final) {
    const vk::AttachmentDescription attachment{{}, format, vk::SampleCountFlagBits::e1, load, vk::AttachmentStoreOp::eStore, {}, {}, initial, final};
    const std::array dependencies{ExternalFragReadDependency()};
    return CreateRenderPass(d, {&attachment, 1}, false, dependencies);
}

static PipelineRenderer CreateSceneRenderer(const PipelineContext &ctx) {
    // Can't construct this map in-place with pairs because `ShaderPipeline` doesn't have a copy constructor.
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    pipelines.emplace(SPT::Fill, ctx.CreateGraphics({{{Vert, "VertexTransform.vert"}, {Frag, "WorkspaceLighting.frag"}}}, eTriangleList, {Blend}, CreateDepthStencil(), MainDrawPushConstantRange));
    pipelines.emplace(SPT::FillDepth, ctx.CreateGraphics({{{Vert, "PositionTransform.vert"}}}, eTriangleList, {NoWrite}, CreateDepthStencil(), MainDrawPushConstantRange));
    pipelines.emplace(SPT::Background, CreateBackgroundPipeline(ctx, false));
    // Exposes the transmission prepass into the scene target as the background and plain-opaque
    // pixels. Premultiplied: the prepass blended straight alpha over a transparent clear.
    static constexpr vk::PipelineColorBlendAttachmentState PremultipliedBlend{
        true,
        vk::BlendFactor::eOne,
        vk::BlendFactor::eOneMinusSrcAlpha,
        vk::BlendOp::eAdd,
        vk::BlendFactor::eOne,
        vk::BlendFactor::eOneMinusSrcAlpha,
        vk::BlendOp::eAdd,
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    pipelines.emplace(SPT::TransmissionComposite, ctx.CreateGraphics({{{Vert, "TexQuad.vert"}, {Frag, "TransmissionComposite.frag"}}}, eTriangleStrip, {PremultipliedBlend}, CreateDepthStencil(false, false)));
    // Fills the scene target with the averaged motion blur accumulation.
    pipelines.emplace(SPT::MotionBlurResolve, ctx.CreateGraphics({{{Vert, "TexQuad.vert"}, {Frag, "MotionBlurResolve.frag"}}}, eTriangleStrip, {NoBlend}, CreateDepthStencil(false, false), FragPc(sizeof(uint32_t) * 2)));
    // Write the silhouette edge depths where they are nearest, so overlays occlude against the
    // outline. Non-edge texels hold the far plane and fail the test, which keeps the depth the
    // transmission composite path loads from its prepass.
    pipelines.emplace(SPT::SilhouetteEdgeDepth, ctx.CreateGraphics({{{Vert, "TexQuad.vert"}, {Frag, "SampleDepth.frag"}}}, eTriangleStrip, {NoWrite}, CreateDepthStencil(true, true), FragPc(sizeof(uint32_t))));
    return {CreateSceneRenderPass(ctx.Device), std::move(pipelines)};
}

static PipelineRenderer CreateOverlayRenderer(const PipelineContext &ctx) {
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    pipelines.emplace(SPT::EdgeQuad, ctx.CreateGraphics({{{Vert, "EdgeQuad.vert"}, {Frag, "EdgeQuad.frag"}}}, eTriangleList, {Blend, NoWrite}, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange));
    pipelines.emplace(SPT::Line, ctx.CreateGraphics({{{Vert, "VertexTransform.vert", {{1, 1u}}}, {Frag, "VertexColor.frag"}}}, eLineList, {Blend, NoBlend}, CreateDepthStencil(true, true, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange));
    pipelines.emplace(SPT::ObjectExtrasLine, ctx.CreateGraphics({{{Vert, "ObjectExtras.vert"}, {Frag, "VertexColor.frag"}}}, eLineList, {Blend, NoBlend}, CreateDepthStencil(true, true, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange));
    const auto make_overlay_pipeline = [&](OverlayKind overlay_kind) {
        return ctx.CreateGraphics({{{Vert, "VertexTransform.vert", {{0, uint32_t(overlay_kind)}, {1, 1u}}}, {Frag, "VertexColor.frag"}}}, eLineList, {Blend, NoBlend}, CreateDepthStencil(true, true, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange);
    };
    pipelines.emplace(SPT::LineOverlayFaceNormals, make_overlay_pipeline(OverlayKind::FaceNormal));
    pipelines.emplace(SPT::LineOverlayVertexNormals, make_overlay_pipeline(OverlayKind::VertexNormal));
    pipelines.emplace(SPT::Point, ctx.CreateGraphics({{{Vert, "VertexPoint.vert"}, {Frag, "VertexPoint.frag"}}}, ePointList, {Blend, NoWrite}, CreateDepthStencil(true, true, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange));
    pipelines.emplace(SPT::Grid, ctx.CreateGraphics({{{Vert, "GridLines.vert"}, {Frag, "GridLines.frag"}}}, eTriangleList, {Blend, NoWrite}, CreateDepthStencil(true, false)));
    // Render silhouette edge color regardless of the tested depth value.
    // Push constants: manipulating flag + sampler index + active object id.
    pipelines.emplace(SPT::SilhouetteEdgeColor, ctx.CreateGraphics({{{Vert, "TexQuad.vert"}, {Frag, "SilhouetteEdgeColor.frag"}}}, eTriangleStrip, {Blend, NoWrite}, CreateDepthStencil(false, false), FragPc(sizeof(uint32_t) * 3)));
    pipelines.emplace(SPT::BoneFill, ctx.CreateGraphics({{{Vert, "BoneSolid.vert"}, {Frag, "BoneSolid.frag"}}}, eTriangleList, {Blend, NoWrite}, CreateDepthStencil(), MainDrawPushConstantRange, 2.0f));
    pipelines.emplace(SPT::BoneWire, ctx.CreateGraphics({{{Vert, "BoneWire.vert"}, {Frag, "VertexColor.frag"}}}, eLineList, {Blend, NoBlend}, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange));
    pipelines.emplace(SPT::BoneSphereFill, ctx.CreateGraphics({{{Vert, "BoneSphere.vert"}, {Frag, "BoneSphere.frag"}}}, eTriangleList, {Blend, NoWrite}, CreateDepthStencil(true, true, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange));
    pipelines.emplace(SPT::BoneSphereWire, ctx.CreateGraphics({{{Vert, "BoneSphereWire.vert"}, {Frag, "VertexColor.frag"}}}, eLineList, {Blend, NoBlend}, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange));
    return {CreateOverlayRenderPass(ctx.Device), std::move(pipelines)};
}

// Additive blend so successive motion blur sub-frames sum into the accumulation target.
static constexpr vk::PipelineColorBlendAttachmentState AdditiveBlend{
    true,
    vk::BlendFactor::eOne,
    vk::BlendFactor::eOne,
    vk::BlendOp::eAdd,
    vk::BlendFactor::eOne,
    vk::BlendFactor::eOne,
    vk::BlendOp::eAdd,
    vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
};

MainPipeline::MainPipeline(const PipelineContext &ctx)
    : SceneRenderer{CreateSceneRenderer(ctx)},
      OverlayRenderer{CreateOverlayRenderer(ctx)},
      SceneDepthLoadRenderPass{CreateSceneRenderPass(ctx.Device, /*load_depth=*/true)},
      SceneResumeRenderPass{CreateSceneRenderPass(ctx.Device, /*load_depth=*/true, /*load_color=*/true)},
      SceneVelocityRenderer{CreateSceneVelocityRenderer(ctx)},
      PrepassBackground{CreateBackgroundPipeline(ctx, true)},
      CompositeRenderPass{CreateColorOnlyRenderPass(ctx.Device, Format::Color, vk::AttachmentLoadOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal)},
      ViewportComposite{CreateQuadPipeline(ctx, "ViewportComposite.frag", NoBlend, sizeof(uint32_t) * 5 + sizeof(vec4))},
      MotionBlurAccumClearRenderPass{CreateColorOnlyRenderPass(ctx.Device, Format::HdrColor, vk::AttachmentLoadOp::eClear, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal)},
      MotionBlurAccumRenderPass{CreateColorOnlyRenderPass(ctx.Device, Format::HdrColor, vk::AttachmentLoadOp::eLoad, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal)},
      MotionBlurAccumulate{CreateQuadPipeline(ctx, "MotionBlurAccumulate.frag", AdditiveBlend, sizeof(uint32_t))},
      MotionBlurGatherRenderPass{CreateColorOnlyRenderPass(ctx.Device, Format::HdrColor, vk::AttachmentLoadOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal)},
      MotionBlurGather{CreateQuadPipeline(ctx, "MotionBlurGather.frag", NoBlend, sizeof(MotionBlurGatherPushConstants))},
      Compiler{ctx, *SceneRenderer.RenderPass, *SceneVelocityRenderer.RenderPass} {}

MainPipeline::ResourcesT::ResourcesT(
    vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass scene_render_pass, vk::RenderPass overlay_render_pass, vk::RenderPass composite_render_pass, DescriptorSlots &slots
    // Depth eSampled: the motion blur gather reads depth to sort samples in front of or behind each pixel.
) : DepthImage{mvk::CreateImage2D(d, pd, Format::Depth, extent, eDepthStencilAttachment | eSampled)},
    SceneColorImage{mvk::CreateImage2D(d, pd, Format::HdrColor, extent, eSampled | eColorAttachment)},
    OverlayColorImage{mvk::CreateImage2D(d, pd, Format::Color, extent, eSampled | eColorAttachment)},
    LineDataImage{mvk::CreateImage2D(d, pd, Format::LineData, extent, eSampled | eColorAttachment)},
    // eTransferSrc enables video-recording readback via vkCmdCopyImageToBuffer.
    FinalColorImage{mvk::CreateImage2D(d, pd, Format::Color, extent, eSampled | eColorAttachment | eTransferSrc)},
    // Power-of-two dimensions keep every level an exact halving, so the multi-level reduce needs no
    // odd-dimension handling. Texels beyond each level's data extent are padding the cull never reads.
    DepthPyramidImage{[&] {
        const vk::Extent2D padded{std::bit_ceil((extent.width + 1) / 2), std::bit_ceil((extent.height + 1) / 2)};
        return mvk::CreateImage2D(d, pd, Format::Float, padded, eSampled | eStorage, mvk::MipLevelCount(padded.width, padded.height));
    }()},
    DepthPyramidMips{[&] {
        const uint32_t mip_count = mvk::MipLevelCount(DepthPyramidImage.Extent.width, DepthPyramidImage.Extent.height);
        std::vector<PyramidMip> mips;
        mips.reserve(mip_count);
        for (uint32_t mip = 0; mip < mip_count; ++mip) {
            auto view = d.createImageViewUnique({{}, *DepthPyramidImage.Image, vk::ImageViewType::e2D, Format::Float, {}, {vk::ImageAspectFlagBits::eColor, mip, 1, 0, 1}});
            // Valid data bounds: scene texel s lands at level index s >> (mip + 1).
            const vk::Extent2D data_extent{((extent.width - 1) >> (mip + 1)) + 1, ((extent.height - 1) >> (mip + 1)) + 1};
            mips.push_back({std::move(view), slots.Allocate(SlotType::Image), data_extent});
        }
        return mips;
    }()},
    NearestSampler{d.createSamplerUnique(mvk::SamplerInfo(vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge))},
    SceneFramebuffer{mvk::CreateFramebuffer(d, scene_render_pass, {*DepthImage.View, *SceneColorImage.View}, extent)},
    OverlayFramebuffer{mvk::CreateFramebuffer(d, overlay_render_pass, {*DepthImage.View, *OverlayColorImage.View, *LineDataImage.View}, extent)},
    CompositeFramebuffer{mvk::CreateFramebuffer(d, composite_render_pass, {*FinalColorImage.View}, extent)},
    Slots{slots} {}

MainPipeline::ResourcesT::~ResourcesT() {
    for (const auto &mip : DepthPyramidMips) Slots.Release({SlotType::Image, mip.Slot});
}

void MainPipeline::SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, DescriptorSlots &slots) {
    Resources = std::make_unique<ResourcesT>(extent, d, pd, *SceneRenderer.RenderPass, *OverlayRenderer.RenderPass, *CompositeRenderPass, slots);
    // Resize invalidates any existing transmission framebuffer (it shares this struct's depth view).
    Transmission.reset();
    // The accumulation target is extent-sized. Drop it so it is reallocated at the new extent on the next MB frame.
    MotionBlur.reset();
}

MainPipeline::MotionBlurResourcesT::MotionBlurResourcesT(
    vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass accum_render_pass, vk::RenderPass scene_velocity_render_pass, vk::RenderPass gather_render_pass, vk::ImageView depth_view, vk::ImageView scene_color_view
) : AccumImage{mvk::CreateImage2D(d, pd, Format::HdrColor, extent, eSampled | eColorAttachment)},
    VelocityImage{mvk::CreateImage2D(d, pd, Format::Velocity, extent, eSampled | eColorAttachment)},
    TileImage{mvk::CreateImage2D(d, pd, Format::HdrColor, DivideCeil(extent, MotionBlurTileSize), eStorage)},
    GatherImage{mvk::CreateImage2D(d, pd, Format::HdrColor, extent, eColorAttachment | eSampled)},
    TileExtent{DivideCeil(extent, MotionBlurTileSize)},
    Framebuffer{mvk::CreateFramebuffer(d, accum_render_pass, {*AccumImage.View}, extent)},
    SceneVelocityFramebuffer{mvk::CreateFramebuffer(d, scene_velocity_render_pass, {depth_view, scene_color_view, *VelocityImage.View}, extent)},
    GatherFramebuffer{mvk::CreateFramebuffer(d, gather_render_pass, {*GatherImage.View}, extent)} {}

bool MainPipeline::EnsureMotionBlurResources(vk::Device d, vk::PhysicalDevice pd) {
    if (MotionBlur || !Resources) return false; // SetExtent drops it, so an allocated target is always at the color extent.
    const auto extent = Resources->SceneColorImage.Extent;
    MotionBlur = std::make_unique<MotionBlurResourcesT>(vk::Extent2D{extent.width, extent.height}, d, pd, *MotionBlurAccumRenderPass, *SceneVelocityRenderer.RenderPass, *MotionBlurGatherRenderPass, *Resources->DepthImage.View, *Resources->SceneColorImage.View);
    return true;
}

vk::DescriptorImageInfo MainPipeline::SceneColorSamplerInfo() const {
    return {*Resources->NearestSampler, *Resources->SceneColorImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
}
vk::DescriptorImageInfo MainPipeline::OverlayColorSamplerInfo() const {
    return {*Resources->NearestSampler, *Resources->OverlayColorImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
}
vk::DescriptorImageInfo MainPipeline::TransmissionSamplerInfo() const {
    if (!Transmission) return SceneColorSamplerInfo();
    return {*Transmission->Sampler, *Transmission->Image.View, vk::ImageLayout::eShaderReadOnlyOptimal};
}
vk::DescriptorImageInfo MainPipeline::MotionBlurAccumSamplerInfo() const {
    if (!MotionBlur) return SceneColorSamplerInfo();
    return {*Resources->NearestSampler, *MotionBlur->AccumImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
}
vk::DescriptorImageInfo MainPipeline::VelocitySamplerInfo() const {
    if (!MotionBlur) return SceneColorSamplerInfo();
    return {*Resources->NearestSampler, *MotionBlur->VelocityImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
}
vk::DescriptorImageInfo MainPipeline::SceneDepthSamplerInfo() const {
    return {*Resources->NearestSampler, *Resources->DepthImage.View, vk::ImageLayout::eDepthStencilReadOnlyOptimal};
}
vk::DescriptorImageInfo MainPipeline::DepthPyramidSamplerInfo() const {
    return {*Resources->NearestSampler, *Resources->DepthPyramidImage.View, vk::ImageLayout::eGeneral};
}
vk::DescriptorImageInfo MainPipeline::MotionBlurTileImageInfo() const {
    return {{}, *MotionBlur->TileImage.View, vk::ImageLayout::eGeneral};
}
vk::DescriptorImageInfo MainPipeline::MotionBlurGatherSamplerInfo() const {
    if (!MotionBlur) return SceneColorSamplerInfo();
    return {*Resources->NearestSampler, *MotionBlur->GatherImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
}

MainPipeline::TransmissionResourcesT::TransmissionResourcesT(
    vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass, vk::ImageView depth_view
) : Image{mvk::CreateImage2D(d, pd, Format::HdrColor, extent, eSampled | eColorAttachment | eTransferSrc | eTransferDst, mvk::MipLevelCount(extent.width, extent.height))},
    MipCount{mvk::MipLevelCount(extent.width, extent.height)},
    Extent{extent},
    Sampler{d.createSamplerUnique(mvk::SamplerInfo(vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eClampToEdge, float(MipCount)))} {
    Mip0View = d.createImageViewUnique({{}, *Image.Image, vk::ImageViewType::e2D, Format::HdrColor, {}, ColorSubresourceRange});
    // The pre-pass renders through the scene render pass into the transmission image's mip 0.
    // The depth view is owned by ResourcesT. Both passes loadOp=Clear so prior contents don't matter.
    Framebuffer = mvk::CreateFramebuffer(d, render_pass, {depth_view, *Mip0View}, extent);
}

bool MainPipeline::EnsureTransmissionResources(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, bool wanted) {
    if (!wanted || !Resources) {
        if (!Transmission) return false;
        Transmission.reset();
        return true;
    }
    if (Transmission && Transmission->Extent.width == extent.width && Transmission->Extent.height == extent.height) return false;
    Transmission = std::make_unique<TransmissionResourcesT>(extent, d, pd, *SceneRenderer.RenderPass, *Resources->DepthImage.View);
    return true;
}

static PipelineRenderer CreateSilhouetteRenderer(const PipelineContext &ctx) {
    const std::vector<vk::AttachmentDescription> attachments{
        // Store depth for reuse by element selection (mutual occlusion between selected meshes).
        {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
        // Single-sampled offscreen "image" of two channels: depth and object ID.
        {{}, Format::Float2, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    pipelines.emplace(SPT::SilhouetteDepthObject, ctx.CreateGraphics({{{Vert, "PositionTransform.vert"}, {Frag, "DepthObject.frag"}}}, eTriangleList, {NoBlend}, CreateDepthStencil(), MainDrawPushConstantRange));
    return {CreateRenderPass(ctx.Device, attachments, true), std::move(pipelines)};
}

SilhouettePipeline::SilhouettePipeline(const PipelineContext &ctx) : Renderer{CreateSilhouetteRenderer(ctx)} {}

SilhouettePipeline::ResourcesT::ResourcesT(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass)
    : DepthImage{mvk::CreateImage2D(d, pd, Format::Depth, extent, eDepthStencilAttachment)},
      OffscreenImage{mvk::CreateImage2D(d, pd, Format::Float2, extent, eSampled | eColorAttachment)},
      // ClampToEdge keeps edge detection from wrapping around to the image's other side: it reads
      // the pixel value at the nearest edge instead.
      ImageSampler{d.createSamplerUnique(mvk::SamplerInfo(vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge))},
      Framebuffer{mvk::CreateFramebuffer(d, render_pass, {*DepthImage.View, *OffscreenImage.View}, extent)} {}

void SilhouettePipeline::SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd) {
    Resources = std::make_unique<ResourcesT>(extent, d, pd, *Renderer.RenderPass);
}

static PipelineRenderer CreateSilhouetteEdgeRenderer(const PipelineContext &ctx) {
    const std::vector<vk::AttachmentDescription> attachments{
        {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilReadOnlyOptimal},
        {{}, Format::Float, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    pipelines.emplace(SPT::SilhouetteEdgeDepthObject, ctx.CreateGraphics({{{Vert, "TexQuad.vert"}, {Frag, "SilhouetteEdgeDepthObject.frag"}}}, eTriangleStrip, {NoBlend}, CreateDepthStencil(), FragPc(sizeof(uint32_t) * 2)));
    return {CreateRenderPass(ctx.Device, attachments, true), std::move(pipelines)};
}

SilhouetteEdgePipeline::SilhouetteEdgePipeline(const PipelineContext &ctx) : Renderer{CreateSilhouetteEdgeRenderer(ctx)} {}

SilhouetteEdgePipeline::ResourcesT::ResourcesT(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass)
    : DepthImage{mvk::CreateImage2D(d, pd, Format::Depth, extent, eSampled | eDepthStencilAttachment)},
      OffscreenImage{mvk::CreateImage2D(d, pd, Format::Float, extent, eSampled | eColorAttachment)},
      ImageSampler{d.createSamplerUnique(vk::SamplerCreateInfo{})},
      DepthSampler{d.createSamplerUnique(vk::SamplerCreateInfo{})},
      Framebuffer{mvk::CreateFramebuffer(d, render_pass, {*DepthImage.View, *OffscreenImage.View}, extent)} {}

void SilhouetteEdgePipeline::SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd) {
    Resources = std::make_unique<ResourcesT>(extent, d, pd, *Renderer.RenderPass);
}

static PipelineRenderer CreateSelectionFragmentRenderer(const PipelineContext &ctx) {
    const std::vector<vk::AttachmentDescription> attachments{
        {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal},
    };
    constexpr vk::PushConstantRange draw_pc{Vert | Frag, 0, sizeof(SelectionDrawPushConstants)};
    constexpr vk::PushConstantRange element_pc{Vert | Frag, 0, sizeof(SelectionElementPushConstants)};
    struct Desc {
        SPT Type;
        const char *Vert;
        const char *Frag;
        vk::PrimitiveTopology Topology;
        vk::PipelineDepthStencilStateCreateInfo DepthStencil;
    };
    constexpr std::array selection_element_pipelines{
        Desc{SPT::SelectionElementFace, "SelectionElementFace.vert", "SelectionElementLinkedList.frag", eTriangleList, CreateDepthStencil(true, true, vk::CompareOp::eLess)},
        Desc{SPT::SelectionElementFaceBitsetBox, "SelectionElementFace.vert", "SelectionElementBitsetBox.frag", eTriangleList, CreateDepthStencil(true, true, vk::CompareOp::eLess)},
        Desc{SPT::SelectionElementFaceXRay, "SelectionElementFace.vert", "SelectionElementLinkedList.frag", eTriangleList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementFaceXRayBitsetBox, "SelectionElementFace.vert", "SelectionElementBitsetBox.frag", eTriangleList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementEdge, "SelectionElementEdge.vert", "SelectionElementLinkedList.frag", eLineList, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual)},
        Desc{SPT::SelectionElementEdgeBitsetBox, "SelectionElementEdge.vert", "SelectionElementBitsetBox.frag", eLineList, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual)},
        Desc{SPT::SelectionElementEdgeXRay, "SelectionElementEdge.vert", "SelectionElementLinkedList.frag", eLineList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementEdgeXRayBitsetBox, "SelectionElementEdge.vert", "SelectionElementBitsetBox.frag", eLineList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementEdgeXRayVerts, "SelectionElementEdge.vert", "SelectionElementLinkedList.frag", ePointList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementEdgeXRayVertsBitsetBox, "SelectionElementEdge.vert", "SelectionElementBitsetBox.frag", ePointList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementVertex, "SelectionElementVertex.vert", "SelectionElementLinkedList.frag", ePointList, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual)},
        Desc{SPT::SelectionElementVertexBitsetBox, "SelectionElementVertex.vert", "SelectionElementBitsetBox.frag", ePointList, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual)},
        Desc{SPT::SelectionElementVertexXRay, "SelectionElementVertex.vert", "SelectionElementLinkedList.frag", ePointList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementVertexXRayBitsetBox, "SelectionElementVertex.vert", "SelectionElementBitsetBox.frag", ePointList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementFaceXRayVerts, "SelectionElementFace.vert", "SelectionElementLinkedList.frag", ePointList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementFaceXRayVertsBitsetBox, "SelectionElementFace.vert", "SelectionElementBitsetBox.frag", ePointList, CreateDepthStencil(false, false)},
    };
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    for (const auto &desc : selection_element_pipelines) {
        pipelines.emplace(desc.Type, ctx.CreateGraphics({{{Vert, desc.Vert}, {Frag, desc.Frag}}}, desc.Topology, {}, desc.DepthStencil, element_pc));
    }
    pipelines.emplace(SPT::SelectionFragmentTriangles, ctx.CreateGraphics({{{Vert, "PositionTransform.vert"}, {Frag, "SelectionFragment.frag"}}}, eTriangleList, {}, CreateDepthStencil(false, false), draw_pc));
    pipelines.emplace(SPT::SelectionFragmentBoneSphere, ctx.CreateGraphics({{{Vert, "BoneSphere.vert"}, {Frag, "SelectionFragment.frag"}}}, eTriangleList, {}, CreateDepthStencil(false, false), draw_pc));
    pipelines.emplace(SPT::SelectionFragmentLines, ctx.CreateGraphics({{{Vert, "PositionTransform.vert"}, {Frag, "SelectionFragment.frag"}}}, eLineList, {}, CreateDepthStencil(false, false), draw_pc));
    pipelines.emplace(SPT::SelectionObjectExtrasLines, ctx.CreateGraphics({{{Vert, "ObjectExtrasSelection.vert"}, {Frag, "SelectionFragment.frag"}}}, eLineList, {}, CreateDepthStencil(false, false), draw_pc));
    pipelines.emplace(SPT::SelectionFragmentPoints, ctx.CreateGraphics({{{Vert, "PositionTransform.vert"}, {Frag, "SelectionFragment.frag"}}}, ePointList, {}, CreateDepthStencil(false, false), draw_pc));
    return {CreateRenderPass(ctx.Device, attachments, true), std::move(pipelines)};
}

SelectionFragmentPipeline::SelectionFragmentPipeline(const PipelineContext &ctx) : Renderer{CreateSelectionFragmentRenderer(ctx)} {}

SelectionFragmentPipeline::ResourcesT::ResourcesT(
    vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass, vk::ImageView silhouette_depth_view
) : HeadImage{mvk::CreateImage2D(d, pd, Format::Uint, extent, eStorage | eTransferDst)},
    Framebuffer{mvk::CreateFramebuffer(d, render_pass, {silhouette_depth_view}, extent)} {}

void SelectionFragmentPipeline::SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::ImageView silhouette_depth_view) {
    Resources = std::make_unique<ResourcesT>(extent, d, pd, *Renderer.RenderPass, silhouette_depth_view);
}

static ComputePipeline CreateCompute(const PipelineContext &ctx, const char *comp, uint32_t pc_size) {
    return ctx.CreateCompute(Shaders{{{ShaderType::eCompute, comp}}}, vk::PushConstantRange{ShaderType::eCompute, 0, pc_size});
}

Pipelines::Pipelines(vk::PhysicalDevice pd, PipelineContext ctx)
    : PhysicalDevice(pd),
      Ctx(ctx),
      Main{Ctx},
      Silhouette{Ctx},
      SilhouetteEdge{Ctx},
      SelectionFragment{Ctx},
      ObjectPick{CreateCompute(Ctx, "ObjectPick.comp", sizeof(ObjectPickPushConstants))},
      ElementPick{CreateCompute(Ctx, "ElementPick.comp", sizeof(ElementPickPushConstants))},
      BoxSelect{CreateCompute(Ctx, "BoxSelect.comp", sizeof(BoxSelectPushConstants))},
      UpdateSelectionState{CreateCompute(Ctx, "UpdateSelectionState.comp", sizeof(UpdateSelectionStatePushConstants))},
      FrustumCull{CreateCompute(Ctx, "FrustumCull.comp", sizeof(FrustumCullPushConstants))},
      DepthPyramidReduce{CreateCompute(Ctx, "DepthPyramidReduce.comp", sizeof(DepthPyramidReducePushConstants))},
      MotionBlurTilesFlatten{CreateCompute(Ctx, "MotionBlurTilesFlatten.comp", sizeof(MotionBlurTilesFlattenPushConstants))},
      MotionBlurTilesDilate{CreateCompute(Ctx, "MotionBlurTilesDilate.comp", sizeof(MotionBlurTilesDilatePushConstants))},
      IblPrefilter{Ctx.Device} {}

void Pipelines::SetExtent(vk::Extent2D extent, DescriptorSlots &slots) {
    Main.SetExtent(extent, Ctx.Device, PhysicalDevice, slots);
    Silhouette.SetExtent(extent, Ctx.Device, PhysicalDevice);
    SilhouetteEdge.SetExtent(extent, Ctx.Device, PhysicalDevice);
    SelectionFragment.SetExtent(extent, Ctx.Device, PhysicalDevice, *Silhouette.Resources->DepthImage.View);
}

void Pipelines::CompileShaders() {
    Main.SceneRenderer.CompileShaders();
    Main.OverlayRenderer.CompileShaders();
    Main.SceneVelocityRenderer.CompileShaders();
    Main.PrepassBackground.Compile(*Main.SceneRenderer.RenderPass);
    Main.ViewportComposite.Compile(*Main.CompositeRenderPass);
    Main.MotionBlurAccumulate.Compile(*Main.MotionBlurAccumRenderPass);
    Main.MotionBlurGather.Compile(*Main.MotionBlurGatherRenderPass);
    Main.Compiler.RecompileModules();
    Silhouette.Renderer.CompileShaders();
    SilhouetteEdge.Renderer.CompileShaders();
    SelectionFragment.Renderer.CompileShaders();
    ObjectPick.Compile();
    ElementPick.Compile();
    BoxSelect.Compile();
    UpdateSelectionState.Compile();
    FrustumCull.Compile();
    DepthPyramidReduce.Compile();
    MotionBlurTilesFlatten.Compile();
    MotionBlurTilesDilate.Compile();
}
