#include "render/Pipelines.h"
#include "Timer.h"
#include "gpu/BoxSelectPushConstants.h"
#include "gpu/ElementPickPushConstants.h"
#include "gpu/MainDrawPushConstants.h"
#include "gpu/MotionBlurGatherPushConstants.h"
#include "gpu/MotionBlurTilesDilatePushConstants.h"
#include "gpu/MotionBlurTilesFlattenPushConstants.h"
#include "gpu/ObjectPickPushConstants.h"
#include "gpu/SelectionDrawPushConstants.h"
#include "gpu/SelectionElementPushConstants.h"
#include "gpu/UpdateSelectionStatePushConstants.h"

#include <ranges>

namespace {
enum class OverlayKind : uint32_t {
    FaceNormal = 1,
    VertexNormal = 2,
};
constexpr std::array PbrSpecFeatures{PbrFeature::Punctual, PbrFeature::Transmission, PbrFeature::DiffuseTrans, PbrFeature::Clearcoat, PbrFeature::Sheen, PbrFeature::Anisotropy, PbrFeature::Iridescence};

constexpr vk::PushConstantRange MainDrawPushConstantRange{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(MainDrawPushConstants)};

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

} // namespace

void PipelineRenderer::CompileShaders() {
    for (auto &shader_pipeline : std::views::values(ShaderPipelines)) shader_pipeline.Compile(*RenderPass);
}

const ShaderPipeline &PipelineRenderer::Bind(vk::CommandBuffer cb, SPT spt) const {
    const auto &pipeline = ShaderPipelines.at(spt);
    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline.Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline.PipelineLayout, 0, pipeline.GetDescriptorSet(), {});
    return pipeline;
}

// No-write: the LineData attachment for overlay pipelines that draw no lines, and the color
// attachment for depth-only scene pipelines.
static constexpr vk::PipelineColorBlendAttachmentState NoWriteBlend{};

PbrCompiler::PbrCompiler(PipelineContext ctx, vk::RenderPass scene, vk::RenderPass scene_velocity)
    : Device(ctx.Device), Cache(Device.createPipelineCacheUnique({})), SetLayout(ctx.SharedLayout), Set(ctx.SharedSet), RenderPass(scene), VelocityRenderPass(scene_velocity) {
    CompileModules();
    Layout = Device.createPipelineLayoutUnique({{}, 1, &SetLayout, 1, &MainDrawPushConstantRange});
}

void PbrCompiler::CompileModules() {
    VertModule = CompileShaderModule(Device, ShaderType::eVertex, "VertexTransform.vert");
    FragModule = CompileShaderModule(Device, ShaderType::eFragment, "pbr.frag");
    VelocityVertModule = CompileShaderModule(Device, ShaderType::eVertex, "VertexTransform.vert", {"VELOCITY_OUTPUT"});
    VelocityFragModule = CompileShaderModule(Device, ShaderType::eFragment, "pbr.frag", {"VELOCITY_OUTPUT"});
}

vk::UniquePipeline PbrCompiler::CreateTargetedPipeline(const vk::SpecializationInfo &frag_spec, bool depth_write, Variant variant) const {
    static constexpr vk::PipelineViewportStateCreateInfo viewport_state{{}, 1, nullptr, 1, nullptr};
    static constexpr std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    static const vk::PipelineDynamicStateCreateInfo dynamic_state{{}, dynamic_states};
    static constexpr vk::PipelineMultisampleStateCreateInfo multisample_state{{}, vk::SampleCountFlagBits::e1};
    static constexpr vk::PipelineVertexInputStateCreateInfo vertex_input{};
    static constexpr vk::PipelineInputAssemblyStateCreateInfo input_assembly{{}, vk::PrimitiveTopology::eTriangleList};
    static const vk::PipelineRasterizationStateCreateInfo raster{{}, false, false, vk::PolygonMode::eFill, {}, vk::FrontFace::eClockwise, false, 0.f, {}, 0.f, 1.f};

    // Opaque geometry writes its screen motion into the velocity attachment. Blend geometry
    // writes neither depth nor velocity, so its velocity twin masks the attachment off.
    const bool velocity_pass = variant == Variant::OpaqueVelocity || variant == Variant::BlendVelocity;
    const bool velocity_modules = variant == Variant::OpaqueVelocity;
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{{}, ShaderType::eVertex, velocity_modules ? *VelocityVertModule : *VertModule, "main"},
        vk::PipelineShaderStageCreateInfo{{}, ShaderType::eFragment, velocity_modules ? *VelocityFragModule : *FragModule, "main", &frag_spec},
    };
    const auto depth_stencil = CreateDepthStencil(true, depth_write);
    const std::array color_blend_attachments{
        CreateColorBlendAttachment(true),
        velocity_modules ? CreateColorBlendAttachment(false) : NoWriteBlend,
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
    if (mask == Mask && OpaqueTargeted && BlendTargeted) return false;

    constexpr uint32_t N = PbrSpecFeatures.size();
    constexpr uint32_t TotalConstants = N + 1; // + TRANSMISSION_PREPASS
    std::array<uint32_t, TotalConstants> data{};
    std::array<vk::SpecializationMapEntry, TotalConstants> entries{};
    for (uint32_t i = 0; i < TotalConstants; ++i) entries[i] = vk::SpecializationMapEntry{i, i * uint32_t(sizeof(uint32_t)), uint32_t(sizeof(uint32_t))};
    for (uint32_t i = 0; i < N; ++i) data[i] = ::HasFeature(mask, PbrSpecFeatures[i]) ? 1u : 0u;
    data[N] = 0u; // main pipelines: exposed radiance, sampling the transmission framebuffer
    const vk::SpecializationInfo spec_main{TotalConstants, entries.data(), TotalConstants * sizeof(uint32_t), data.data()};
    const Timer timer{"PBR pipeline compile"};
    OpaqueTargeted = CreateTargetedPipeline(spec_main, true, Variant::Opaque);
    BlendTargeted = CreateTargetedPipeline(spec_main, false, Variant::Blend);
    OpaqueVelocityTargeted = CreateTargetedPipeline(spec_main, true, Variant::OpaqueVelocity);
    BlendVelocityTargeted = CreateTargetedPipeline(spec_main, false, Variant::BlendVelocity);
    if (::HasFeature(mask, PbrFeature::Transmission)) {
        data[N] = 1u; // pre-pass pipeline: un-exposed radiance, no framebuffer self-sampling
        const vk::SpecializationInfo spec_prepass{TotalConstants, entries.data(), TotalConstants * sizeof(uint32_t), data.data()};
        OpaquePrepass = CreateTargetedPipeline(spec_prepass, true, Variant::OpaquePrepass);
    } else {
        OpaquePrepass.reset();
    }
    Mask = mask;

    return true;
}

vk::PipelineLayout PbrCompiler::Bind(vk::CommandBuffer cb, Variant v) const {
    const auto &pipeline = v == Variant::Opaque ? OpaqueTargeted :
        v == Variant::Blend                     ? BlendTargeted :
        v == Variant::OpaqueVelocity            ? OpaqueVelocityTargeted :
        v == Variant::BlendVelocity             ? BlendVelocityTargeted :
                                                  OpaquePrepass;
    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *Layout, 0, Set, {});
    return *Layout;
}

void PbrCompiler::RecompileModules() {
    if (!Device) return;
    CompileModules();
    OpaqueTargeted.reset();
    BlendTargeted.reset();
    OpaqueVelocityTargeted.reset();
    BlendVelocityTargeted.reset();
    OpaquePrepass.reset();
    CompilePipelines(Mask);
}

// Write-only for LineData attachment (used by line pipelines as their 2nd color blend state).
static const vk::PipelineColorBlendAttachmentState LineDataBlend = CreateColorBlendAttachment(false);

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

// `transmission_prepass` selects the variant that skips exposure, which the main pass applies after sampling.
static ShaderPipeline CreateBackgroundPipeline(const PipelineContext &ctx, bool transmission_prepass) {
    return ctx.CreateGraphics(
        {{{ShaderType::eVertex, "Background.vert"}, {ShaderType::eFragment, "Background.frag", {{0, transmission_prepass ? 1u : 0u}}}}},
        {},
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
        {CreateColorBlendAttachment(true)}, CreateDepthStencil(false, false)
    );
}

// Fullscreen textured quad into a single color attachment: no depth, fragment-only push constants.
static ShaderPipeline CreateQuadPipeline(const PipelineContext &ctx, std::filesystem::path frag, vk::PipelineColorBlendAttachmentState blend, uint32_t push_constant_size) {
    return ctx.CreateGraphics(
        {{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, std::move(frag)}}},
        {},
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
        {blend}, {}, vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, push_constant_size}
    );
}

// Depth + a single scene-linear color attachment. Depth is stored for the overlay pass to load and
// occlude against. Also backs the transmission pre-pass, which renders into its own framebuffer.
static vk::UniqueRenderPass CreateSceneRenderPass(vk::Device d) {
    const std::vector<vk::AttachmentDescription> attachments{
        {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
        // Color: cleared transparent, stored for sampling in the viewport composite.
        {{}, Format::HdrColor, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
    const vk::AttachmentReference color_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, &depth_attachment_ref};
    const std::array dependencies{ExternalFragReadDependency()};
    return d.createRenderPassUnique({{}, attachments, subpass, dependencies});
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
    const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
    const std::array color_attachment_refs{
        vk::AttachmentReference{1, vk::ImageLayout::eColorAttachmentOptimal},
        vk::AttachmentReference{2, vk::ImageLayout::eColorAttachmentOptimal},
    };
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, uint32_t(color_attachment_refs.size()), color_attachment_refs.data(), nullptr, &depth_attachment_ref};
    const std::array dependencies{
        // The scene pass's depth writes must land before this pass tests against them.
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
    return d.createRenderPassUnique({{}, attachments, subpass, dependencies});
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
    const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
    const std::array color_attachment_refs{
        vk::AttachmentReference{1, vk::ImageLayout::eColorAttachmentOptimal},
        vk::AttachmentReference{2, vk::ImageLayout::eColorAttachmentOptimal},
    };
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, uint32_t(color_attachment_refs.size()), color_attachment_refs.data(), nullptr, &depth_attachment_ref};
    const std::array dependencies{ExternalFragReadDependency()};
    return d.createRenderPassUnique({{}, attachments, subpass, dependencies});
}

// Fullscreen-quad twins for the scene+velocity render pass, keyed by the same SPTs the plain scene
// pass binds. PBR geometry goes through PbrCompiler's velocity variants instead.
static PipelineRenderer CreateSceneVelocityRenderer(vk::Device d, vk::DescriptorSetLayout shared_layout, vk::DescriptorSet shared_set) {
    const PipelineContext ctx{d, shared_layout, shared_set};
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    // Screen motion for every pixel geometry leaves uncovered. Drawn first, so geometry overwrites
    // it wherever it lands, and the scene color stays untouched through the write mask.
    pipelines.emplace(
        SPT::BackgroundVelocity,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "Background.vert"}, {ShaderType::eFragment, "BackgroundVelocity.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
            {NoWriteBlend, CreateColorBlendAttachment(false)}, CreateDepthStencil(false, false)
        )
    );
    // The environment background writes color only: its motion comes from the quad above.
    pipelines.emplace(
        SPT::Background,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "Background.vert"}, {ShaderType::eFragment, "Background.frag", {{0, 0u}}}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
            {CreateColorBlendAttachment(true), NoWriteBlend}, CreateDepthStencil(false, false)
        )
    );
    // Depth-only silhouette edge quad (see the scene renderer's twin for the depth-test rationale).
    pipelines.emplace(
        SPT::SilhouetteEdgeDepth,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "SampleDepth.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
            {NoWriteBlend, NoWriteBlend}, CreateDepthStencil(true, true, vk::CompareOp::eAlways),
            vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t)}
        )
    );
    return {CreateSceneVelocityRenderPass(d), std::move(pipelines)};
}

// One color attachment, no depth, always stored. Backs each of the fullscreen-quad passes.
static vk::UniqueRenderPass CreateColorOnlyRenderPass(vk::Device d, vk::Format format, vk::AttachmentLoadOp load, vk::ImageLayout initial, vk::ImageLayout final) {
    const vk::AttachmentDescription attachment{{}, format, vk::SampleCountFlagBits::e1, load, vk::AttachmentStoreOp::eStore, {}, {}, initial, final};
    const vk::AttachmentReference color_ref{0, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_ref};
    const std::array dependencies{ExternalFragReadDependency()};
    return d.createRenderPassUnique({{}, attachment, subpass, dependencies});
}

static PipelineRenderer CreateSceneRenderer(
    vk::Device d,
    vk::DescriptorSetLayout shared_layout, vk::DescriptorSet shared_set
) {
    const PipelineContext ctx{d, shared_layout, shared_set};

    // Can't construct this map in-place with pairs because `ShaderPipeline` doesn't have a copy constructor.
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    pipelines.emplace(
        SPT::Fill,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "WorkspaceLighting.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            {CreateColorBlendAttachment(true)}, CreateDepthStencil(), MainDrawPushConstantRange
        )
    );
    pipelines.emplace(
        SPT::FillDepth,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "PositionTransform.vert"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            {NoWriteBlend}, CreateDepthStencil(), MainDrawPushConstantRange
        )
    );
    pipelines.emplace(SPT::Background, CreateBackgroundPipeline(ctx, false));
    // Fills the scene target with the averaged motion blur accumulation.
    pipelines.emplace(
        SPT::MotionBlurResolve,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "MotionBlurResolve.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
            {CreateColorBlendAttachment(false)}, CreateDepthStencil(false, false),
            vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t) * 2}
        )
    );
    // Render the silhouette edge depth regardless of the tested depth value.
    // We should be able to just disable depth tests and enable depth writes, but it seems that some GPUs or drivers
    // optimize out depth writes when depth testing is disabled, so instead we configure a depth test that always passes.
    pipelines.emplace(
        SPT::SilhouetteEdgeDepth,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "SampleDepth.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
            {NoWriteBlend}, CreateDepthStencil(true, true, vk::CompareOp::eAlways),
            vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t)}
        )
    );
    return {CreateSceneRenderPass(d), std::move(pipelines)};
}

static PipelineRenderer CreateOverlayRenderer(
    vk::Device d,
    vk::DescriptorSetLayout shared_layout, vk::DescriptorSet shared_set
) {
    const PipelineContext ctx{d, shared_layout, shared_set};

    std::unordered_map<SPT, ShaderPipeline> pipelines;
    pipelines.emplace(
        SPT::EdgeQuad,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "EdgeQuad.vert"}, {ShaderType::eFragment, "EdgeQuad.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            {CreateColorBlendAttachment(true), NoWriteBlend}, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange
        )
    );
    pipelines.emplace(
        SPT::Line,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "VertexTransform.vert", {{1, 1u}}}, {ShaderType::eFragment, "VertexColor.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eLineList,
            {CreateColorBlendAttachment(true), LineDataBlend}, CreateDepthStencil(true, true, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange
        )
    );
    pipelines.emplace(
        SPT::ObjectExtrasLine,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "ObjectExtras.vert"}, {ShaderType::eFragment, "VertexColor.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eLineList,
            {CreateColorBlendAttachment(true), LineDataBlend}, CreateDepthStencil(true, true, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange
        )
    );
    const auto make_overlay_pipeline = [&](OverlayKind overlay_kind) {
        return ctx.CreateGraphics(
            {{{ShaderType::eVertex, "VertexTransform.vert", {{0, uint32_t(overlay_kind)}, {1, 1u}}},
              {ShaderType::eFragment, "VertexColor.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eLineList,
            {CreateColorBlendAttachment(true), LineDataBlend}, CreateDepthStencil(true, true, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange
        );
    };
    pipelines.emplace(SPT::LineOverlayFaceNormals, make_overlay_pipeline(OverlayKind::FaceNormal));
    pipelines.emplace(SPT::LineOverlayVertexNormals, make_overlay_pipeline(OverlayKind::VertexNormal));
    pipelines.emplace(
        SPT::Point,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "VertexPoint.vert"}, {ShaderType::eFragment, "VertexPoint.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::ePointList,
            {CreateColorBlendAttachment(true), NoWriteBlend}, CreateDepthStencil(true, true, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange
        )
    );
    pipelines.emplace(
        SPT::Grid,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "GridLines.vert"}, {ShaderType::eFragment, "GridLines.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            {CreateColorBlendAttachment(true), NoWriteBlend}, CreateDepthStencil(true, false)
        )
    );
    // Render silhouette edge color regardless of the tested depth value.
    pipelines.emplace(
        SPT::SilhouetteEdgeColor,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "SilhouetteEdgeColor.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
            {CreateColorBlendAttachment(true), NoWriteBlend}, CreateDepthStencil(false, false),
            vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t) * 3} // Manipulating flag + sampler index + active object id
        )
    );
    pipelines.emplace(
        SPT::BoneFill,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "BoneSolid.vert"}, {ShaderType::eFragment, "BoneSolid.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            {CreateColorBlendAttachment(true), NoWriteBlend}, CreateDepthStencil(), MainDrawPushConstantRange, 2.0f
        )
    );
    pipelines.emplace(
        SPT::BoneWire,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "BoneWire.vert"}, {ShaderType::eFragment, "VertexColor.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eLineList,
            {CreateColorBlendAttachment(true), LineDataBlend}, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange
        )
    );
    pipelines.emplace(
        SPT::BoneSphereFill,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "BoneSphere.vert"}, {ShaderType::eFragment, "BoneSphere.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            {CreateColorBlendAttachment(true), NoWriteBlend}, CreateDepthStencil(true, true, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange
        )
    );
    pipelines.emplace(
        SPT::BoneSphereWire,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "BoneSphereWire.vert"}, {ShaderType::eFragment, "VertexColor.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eLineList,
            {CreateColorBlendAttachment(true), LineDataBlend}, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual), MainDrawPushConstantRange
        )
    );
    return {CreateOverlayRenderPass(d), std::move(pipelines)};
}

MainPipeline::MainPipeline(
    vk::Device d,
    vk::DescriptorSetLayout shared_layout, vk::DescriptorSet shared_set
) : SceneRenderer{CreateSceneRenderer(d, shared_layout, shared_set)},
    OverlayRenderer{CreateOverlayRenderer(d, shared_layout, shared_set)},
    SceneVelocityRenderer{CreateSceneVelocityRenderer(d, shared_layout, shared_set)},
    PrepassBackground{CreateBackgroundPipeline({d, shared_layout, shared_set}, true)},
    CompositeRenderPass{CreateColorOnlyRenderPass(d, Format::Color, vk::AttachmentLoadOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal)},
    ViewportComposite{CreateQuadPipeline({d, shared_layout, shared_set}, "ViewportComposite.frag", CreateColorBlendAttachment(false), sizeof(uint32_t) * 5 + sizeof(vec4))},
    MotionBlurAccumClearRenderPass{CreateColorOnlyRenderPass(d, Format::HdrColor, vk::AttachmentLoadOp::eClear, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal)},
    MotionBlurAccumRenderPass{CreateColorOnlyRenderPass(d, Format::HdrColor, vk::AttachmentLoadOp::eLoad, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal)},
    MotionBlurAccumulate{CreateQuadPipeline({d, shared_layout, shared_set}, "MotionBlurAccumulate.frag", AdditiveBlend, sizeof(uint32_t))},
    MotionBlurGatherRenderPass{CreateColorOnlyRenderPass(d, Format::HdrColor, vk::AttachmentLoadOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal)},
    MotionBlurGather{CreateQuadPipeline({d, shared_layout, shared_set}, "MotionBlurGather.frag", CreateColorBlendAttachment(false), sizeof(MotionBlurGatherPushConstants))},
    Compiler{{d, shared_layout, shared_set}, *SceneRenderer.RenderPass, *SceneVelocityRenderer.RenderPass} {}

static uint32_t MipCount(vk::Extent2D extent) {
    uint32_t levels = 1;
    uint32_t side = std::max(extent.width, extent.height);
    while (side > 1) {
        side >>= 1;
        ++levels;
    }
    return levels;
}

MainPipeline::ResourcesT::ResourcesT(
    vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass scene_render_pass, vk::RenderPass overlay_render_pass, vk::RenderPass composite_render_pass
    // eSampled: the motion blur gather reads depth to sort samples in front of or behind each pixel.
) : DepthImage{mvk::CreateImage(d, pd, {{}, vk::ImageType::e2D, Format::Depth, vk::Extent3D{extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled, vk::SharingMode::eExclusive}, {{}, {}, vk::ImageViewType::e2D, Format::Depth, {}, DepthSubresourceRange})},
    SceneColorImage{mvk::CreateImage(d, pd, {{}, vk::ImageType::e2D, Format::HdrColor, vk::Extent3D{extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive}, {{}, {}, vk::ImageViewType::e2D, Format::HdrColor, {}, ColorSubresourceRange})},
    OverlayColorImage{mvk::CreateImage(d, pd, {{}, vk::ImageType::e2D, Format::Color, vk::Extent3D{extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive}, {{}, {}, vk::ImageViewType::e2D, Format::Color, {}, ColorSubresourceRange})},
    LineDataImage{mvk::CreateImage(d, pd, {{}, vk::ImageType::e2D, Format::LineData, vk::Extent3D{extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive}, {{}, {}, vk::ImageViewType::e2D, Format::LineData, {}, ColorSubresourceRange})},
    // eTransferSrc enables video-recording readback via vkCmdCopyImageToBuffer.
    FinalColorImage{mvk::CreateImage(d, pd, {{}, vk::ImageType::e2D, Format::Color, vk::Extent3D{extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive}, {{}, {}, vk::ImageViewType::e2D, Format::Color, {}, ColorSubresourceRange})},
    NearestSampler{d.createSamplerUnique({
        {},
        vk::Filter::eNearest,
        vk::Filter::eNearest,
        vk::SamplerMipmapMode::eNearest,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
    })} {
    {
        const std::array image_views{*DepthImage.View, *SceneColorImage.View};
        SceneFramebuffer = d.createFramebufferUnique({{}, scene_render_pass, image_views, extent.width, extent.height, 1});
    }
    {
        const std::array image_views{*DepthImage.View, *OverlayColorImage.View, *LineDataImage.View};
        OverlayFramebuffer = d.createFramebufferUnique({{}, overlay_render_pass, image_views, extent.width, extent.height, 1});
    }
    {
        const std::array image_views{*FinalColorImage.View};
        CompositeFramebuffer = d.createFramebufferUnique({{}, composite_render_pass, image_views, extent.width, extent.height, 1});
    }
}

void MainPipeline::SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd) {
    Resources = std::make_unique<ResourcesT>(extent, d, pd, *SceneRenderer.RenderPass, *OverlayRenderer.RenderPass, *CompositeRenderPass);
    // Resize invalidates any existing transmission framebuffer (it shares this struct's depth view).
    Transmission.reset();
    // The accumulation target is extent-sized. Drop it so it is reallocated at the new extent on the next MB frame.
    MotionBlur.reset();
}

MainPipeline::MotionBlurResourcesT::MotionBlurResourcesT(
    vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass accum_render_pass, vk::RenderPass scene_velocity_render_pass, vk::RenderPass gather_render_pass, vk::ImageView depth_view, vk::ImageView scene_color_view
) : AccumImage{mvk::CreateImage(d, pd, {{}, vk::ImageType::e2D, Format::HdrColor, vk::Extent3D{extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive}, {{}, {}, vk::ImageViewType::e2D, Format::HdrColor, {}, ColorSubresourceRange})},
    VelocityImage{mvk::CreateImage(d, pd, {{}, vk::ImageType::e2D, Format::Velocity, vk::Extent3D{extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive}, {{}, {}, vk::ImageViewType::e2D, Format::Velocity, {}, ColorSubresourceRange})},
    TileImage{mvk::CreateImage(d, pd, {{}, vk::ImageType::e2D, Format::HdrColor, vk::Extent3D{DivideCeil(extent, MotionBlurTileSize), 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eStorage, vk::SharingMode::eExclusive}, {{}, {}, vk::ImageViewType::e2D, Format::HdrColor, {}, ColorSubresourceRange})},
    GatherImage{mvk::CreateImage(d, pd, {{}, vk::ImageType::e2D, Format::HdrColor, vk::Extent3D{extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::SharingMode::eExclusive}, {{}, {}, vk::ImageViewType::e2D, Format::HdrColor, {}, ColorSubresourceRange})},
    TileExtent{DivideCeil(extent, MotionBlurTileSize)} {
    {
        const std::array image_views{*AccumImage.View};
        Framebuffer = d.createFramebufferUnique({{}, accum_render_pass, image_views, extent.width, extent.height, 1});
    }
    {
        const std::array image_views{depth_view, scene_color_view, *VelocityImage.View};
        SceneVelocityFramebuffer = d.createFramebufferUnique({{}, scene_velocity_render_pass, image_views, extent.width, extent.height, 1});
    }
    {
        const std::array image_views{*GatherImage.View};
        GatherFramebuffer = d.createFramebufferUnique({{}, gather_render_pass, image_views, extent.width, extent.height, 1});
    }
}

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
vk::DescriptorImageInfo MainPipeline::MotionBlurTileImageInfo() const {
    return {{}, *MotionBlur->TileImage.View, vk::ImageLayout::eGeneral};
}
vk::DescriptorImageInfo MainPipeline::MotionBlurGatherSamplerInfo() const {
    if (!MotionBlur) return SceneColorSamplerInfo();
    return {*Resources->NearestSampler, *MotionBlur->GatherImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
}

MainPipeline::TransmissionResourcesT::TransmissionResourcesT(
    vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass, vk::ImageView depth_view
) : Image{mvk::CreateImage(d, pd, {{}, vk::ImageType::e2D, Format::HdrColor, vk::Extent3D{extent, 1}, ::MipCount(extent), 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive}, {{}, {}, vk::ImageViewType::e2D, Format::HdrColor, {}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, ::MipCount(extent), 0, 1}})},
    MipCount{::MipCount(extent)},
    Extent{extent},
    Sampler{d.createSamplerUnique({
        {},
        vk::Filter::eLinear,
        vk::Filter::eLinear,
        vk::SamplerMipmapMode::eLinear,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        0.f,
        VK_FALSE,
        1.f,
        VK_FALSE,
        vk::CompareOp::eNever,
        0.f,
        float(::MipCount(extent)),
    })} {
    Mip0View = d.createImageViewUnique({{}, *Image.Image, vk::ImageViewType::e2D, Format::HdrColor, {}, ColorSubresourceRange});
    // The pre-pass renders through the scene render pass into the transmission image's mip 0.
    // The depth view is owned by ResourcesT. Both passes loadOp=Clear so prior contents don't matter.
    const std::array image_views{depth_view, *Mip0View};
    Framebuffer = d.createFramebufferUnique({{}, render_pass, image_views, extent.width, extent.height, 1});
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

static PipelineRenderer CreateSilhouetteRenderer(vk::Device d, vk::DescriptorSetLayout shared_layout, vk::DescriptorSet shared_set) {
    const std::vector<vk::AttachmentDescription> attachments{
        // Store depth for reuse by element selection (mutual occlusion between selected meshes).
        {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
        // Single-sampled offscreen "image" of two channels: depth and object ID.
        {{}, Format::Float2, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
    const vk::AttachmentReference color_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, &depth_attachment_ref};

    const PipelineContext ctx{d, shared_layout, shared_set};
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    pipelines.emplace(
        SPT::SilhouetteDepthObject,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "DepthObject.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            {CreateColorBlendAttachment(false)}, CreateDepthStencil(), MainDrawPushConstantRange
        )
    );
    return {d.createRenderPassUnique({{}, attachments, subpass}), std::move(pipelines)};
}

SilhouettePipeline::SilhouettePipeline(vk::Device d, vk::DescriptorSetLayout shared_layout, vk::DescriptorSet shared_set)
    : Renderer{CreateSilhouetteRenderer(d, shared_layout, shared_set)} {}

SilhouettePipeline::ResourcesT::ResourcesT(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass)
    : DepthImage{mvk::CreateImage(
          d, pd,
          {{},
           vk::ImageType::e2D,
           Format::Depth,
           vk::Extent3D{extent, 1},
           1,
           1,
           vk::SampleCountFlagBits::e1,
           vk::ImageTiling::eOptimal,
           vk::ImageUsageFlagBits::eDepthStencilAttachment,
           vk::SharingMode::eExclusive},
          {{}, {}, vk::ImageViewType::e2D, Format::Depth, {}, DepthSubresourceRange}
      )},
      OffscreenImage{mvk::CreateImage(
          d, pd,
          {{},
           vk::ImageType::e2D,
           Format::Float2,
           vk::Extent3D{extent, 1},
           1,
           1,
           vk::SampleCountFlagBits::e1,
           vk::ImageTiling::eOptimal,
           vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
           vk::SharingMode::eExclusive},
          {{}, {}, vk::ImageViewType::e2D, Format::Float2, {}, ColorSubresourceRange}
      )} {
    const std::array image_views{*DepthImage.View, *OffscreenImage.View};
    Framebuffer = d.createFramebufferUnique({{}, render_pass, image_views, extent.width, extent.height, 1});
    ImageSampler = d.createSamplerUnique({
        {},
        vk::Filter::eNearest,
        vk::Filter::eNearest,
        vk::SamplerMipmapMode::eNearest,
        // Prevent edge detection from wrapping around to the other side of the image.
        // Instead, use the pixel value at the nearest edge.
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
    });
}

void SilhouettePipeline::SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd) {
    Resources = std::make_unique<ResourcesT>(extent, d, pd, *Renderer.RenderPass);
}

static PipelineRenderer CreateSilhouetteEdgeRenderer(vk::Device d, vk::DescriptorSetLayout shared_layout, vk::DescriptorSet shared_set) {
    const std::vector<vk::AttachmentDescription> attachments{
        {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilReadOnlyOptimal},
        {{}, Format::Float, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
    const vk::AttachmentReference color_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, &depth_attachment_ref};

    const PipelineContext ctx{d, shared_layout, shared_set};
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    pipelines.emplace(
        SPT::SilhouetteEdgeDepthObject,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "SilhouetteEdgeDepthObject.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
            {CreateColorBlendAttachment(false)}, CreateDepthStencil(),
            vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t) * 2}
        )
    );
    return {d.createRenderPassUnique({{}, attachments, subpass}), std::move(pipelines)};
}

SilhouetteEdgePipeline::SilhouetteEdgePipeline(vk::Device d, vk::DescriptorSetLayout shared_layout, vk::DescriptorSet shared_set)
    : Renderer{CreateSilhouetteEdgeRenderer(d, shared_layout, shared_set)} {}

SilhouetteEdgePipeline::ResourcesT::ResourcesT(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass)
    : DepthImage{mvk::CreateImage(
          d, pd,
          {{},
           vk::ImageType::e2D,
           Format::Depth,
           vk::Extent3D{extent, 1},
           1,
           1,
           vk::SampleCountFlagBits::e1,
           vk::ImageTiling::eOptimal,
           vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eDepthStencilAttachment,
           vk::SharingMode::eExclusive},
          {{}, {}, vk::ImageViewType::e2D, Format::Depth, {}, DepthSubresourceRange}
      )},
      OffscreenImage{mvk::CreateImage(
          d, pd,
          {{},
           vk::ImageType::e2D,
           Format::Float,
           vk::Extent3D{extent, 1},
           1,
           1,
           vk::SampleCountFlagBits::e1,
           vk::ImageTiling::eOptimal,
           vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
           vk::SharingMode::eExclusive},
          {{}, {}, vk::ImageViewType::e2D, Format::Float, {}, ColorSubresourceRange}
      )} {
    const std::array image_views{*DepthImage.View, *OffscreenImage.View};
    Framebuffer = d.createFramebufferUnique({{}, render_pass, image_views, extent.width, extent.height, 1});
    ImageSampler = d.createSamplerUnique(vk::SamplerCreateInfo{});
    DepthSampler = d.createSamplerUnique(vk::SamplerCreateInfo{});
}

void SilhouetteEdgePipeline::SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd) {
    Resources = std::make_unique<ResourcesT>(extent, d, pd, *Renderer.RenderPass);
}

static PipelineRenderer CreateSelectionFragmentRenderer(vk::Device d, vk::DescriptorSetLayout shared_layout, vk::DescriptorSet shared_set) {
    const std::vector<vk::AttachmentDescription> attachments{
        {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal},
    };
    const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 0, nullptr, nullptr, &depth_attachment_ref};

    const PipelineContext ctx{d, shared_layout, shared_set};
    const vk::PushConstantRange draw_pc{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(SelectionDrawPushConstants)};
    const vk::PushConstantRange element_pc{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(SelectionElementPushConstants)};
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    const auto make_selection_element = [&](const char *vert, const char *frag, vk::PrimitiveTopology topology, vk::PipelineDepthStencilStateCreateInfo depth_stencil) {
        return ctx.CreateGraphics({{{ShaderType::eVertex, vert}, {ShaderType::eFragment, frag}}}, {}, vk::PolygonMode::eFill, topology, {}, depth_stencil, element_pc);
    };
    struct Desc {
        SPT Type;
        const char *Vert;
        const char *Frag;
        vk::PrimitiveTopology Topology;
        vk::PipelineDepthStencilStateCreateInfo DepthStencil;
    };
    const std::array selection_element_pipelines{
        Desc{SPT::SelectionElementFace, "SelectionElementFace.vert", "SelectionElementLinkedList.frag", vk::PrimitiveTopology::eTriangleList, CreateDepthStencil(true, true, vk::CompareOp::eLess)},
        Desc{SPT::SelectionElementFaceBitsetBox, "SelectionElementFace.vert", "SelectionElementBitsetBox.frag", vk::PrimitiveTopology::eTriangleList, CreateDepthStencil(true, true, vk::CompareOp::eLess)},
        Desc{SPT::SelectionElementFaceXRay, "SelectionElementFace.vert", "SelectionElementLinkedList.frag", vk::PrimitiveTopology::eTriangleList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementFaceXRayBitsetBox, "SelectionElementFace.vert", "SelectionElementBitsetBox.frag", vk::PrimitiveTopology::eTriangleList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementEdge, "SelectionElementEdge.vert", "SelectionElementLinkedList.frag", vk::PrimitiveTopology::eLineList, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual)},
        Desc{SPT::SelectionElementEdgeBitsetBox, "SelectionElementEdge.vert", "SelectionElementBitsetBox.frag", vk::PrimitiveTopology::eLineList, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual)},
        Desc{SPT::SelectionElementEdgeXRay, "SelectionElementEdge.vert", "SelectionElementLinkedList.frag", vk::PrimitiveTopology::eLineList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementEdgeXRayBitsetBox, "SelectionElementEdge.vert", "SelectionElementBitsetBox.frag", vk::PrimitiveTopology::eLineList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementEdgeXRayVerts, "SelectionElementEdge.vert", "SelectionElementLinkedList.frag", vk::PrimitiveTopology::ePointList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementEdgeXRayVertsBitsetBox, "SelectionElementEdge.vert", "SelectionElementBitsetBox.frag", vk::PrimitiveTopology::ePointList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementVertex, "SelectionElementVertex.vert", "SelectionElementLinkedList.frag", vk::PrimitiveTopology::ePointList, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual)},
        Desc{SPT::SelectionElementVertexBitsetBox, "SelectionElementVertex.vert", "SelectionElementBitsetBox.frag", vk::PrimitiveTopology::ePointList, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual)},
        Desc{SPT::SelectionElementVertexXRay, "SelectionElementVertex.vert", "SelectionElementLinkedList.frag", vk::PrimitiveTopology::ePointList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementVertexXRayBitsetBox, "SelectionElementVertex.vert", "SelectionElementBitsetBox.frag", vk::PrimitiveTopology::ePointList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementFaceXRayVerts, "SelectionElementFace.vert", "SelectionElementLinkedList.frag", vk::PrimitiveTopology::ePointList, CreateDepthStencil(false, false)},
        Desc{SPT::SelectionElementFaceXRayVertsBitsetBox, "SelectionElementFace.vert", "SelectionElementBitsetBox.frag", vk::PrimitiveTopology::ePointList, CreateDepthStencil(false, false)},
    };
    for (const auto &desc : selection_element_pipelines) {
        pipelines.emplace(desc.Type, make_selection_element(desc.Vert, desc.Frag, desc.Topology, desc.DepthStencil));
    }
    pipelines.emplace(
        SPT::SelectionFragmentTriangles,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "SelectionFragment.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            {}, CreateDepthStencil(false, false), draw_pc
        )
    );
    pipelines.emplace(
        SPT::SelectionFragmentBoneSphere,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "BoneSphere.vert"}, {ShaderType::eFragment, "SelectionFragment.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            {}, CreateDepthStencil(false, false), draw_pc
        )
    );
    pipelines.emplace(
        SPT::SelectionFragmentLines,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "SelectionFragment.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eLineList,
            {}, CreateDepthStencil(false, false), draw_pc
        )
    );
    pipelines.emplace(
        SPT::SelectionObjectExtrasLines,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "ObjectExtrasSelection.vert"}, {ShaderType::eFragment, "SelectionFragment.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eLineList,
            {}, CreateDepthStencil(false, false), draw_pc
        )
    );
    pipelines.emplace(
        SPT::SelectionFragmentPoints,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "SelectionFragment.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::ePointList,
            {}, CreateDepthStencil(false, false), draw_pc
        )
    );
    return {d.createRenderPassUnique({{}, attachments, subpass}), std::move(pipelines)};
}

SelectionFragmentPipeline::SelectionFragmentPipeline(vk::Device d, vk::DescriptorSetLayout shared_layout, vk::DescriptorSet shared_set)
    : Renderer{CreateSelectionFragmentRenderer(d, shared_layout, shared_set)} {}

SelectionFragmentPipeline::ResourcesT::ResourcesT(
    vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass, vk::ImageView silhouette_depth_view
) : HeadImage{mvk::CreateImage(d, pd, {{}, vk::ImageType::e2D, Format::Uint, vk::Extent3D{extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive}, {{}, {}, vk::ImageViewType::e2D, Format::Uint, {}, ColorSubresourceRange})},
    Framebuffer{d.createFramebufferUnique({{}, render_pass, silhouette_depth_view, extent.width, extent.height, 1})} {}

void SelectionFragmentPipeline::SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::ImageView silhouette_depth_view) {
    Resources = std::make_unique<ResourcesT>(extent, d, pd, *Renderer.RenderPass, silhouette_depth_view);
}

Pipelines::Pipelines(
    vk::Device d, vk::PhysicalDevice pd,
    vk::DescriptorSetLayout selection_layout, vk::DescriptorSet selection_set
) : Device(d),
    PhysicalDevice(pd),
    Main{d, selection_layout, selection_set},
    Silhouette{d, selection_layout, selection_set},
    SilhouetteEdge{d, selection_layout, selection_set},
    SelectionFragment{d, selection_layout, selection_set},
    ObjectPick{
        d, Shaders{{{ShaderType::eCompute, "ObjectPick.comp"}}},
        vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ObjectPickPushConstants)},
        selection_layout,
        selection_set
    },
    ElementPick{
        d, Shaders{{{ShaderType::eCompute, "ElementPick.comp"}}},
        vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ElementPickPushConstants)},
        selection_layout,
        selection_set
    },
    BoxSelect{
        d, Shaders{{{ShaderType::eCompute, "BoxSelect.comp"}}},
        vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(BoxSelectPushConstants)},
        selection_layout,
        selection_set
    },
    UpdateSelectionState{
        d, Shaders{{{ShaderType::eCompute, "UpdateSelectionState.comp"}}},
        vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(UpdateSelectionStatePushConstants)},
        selection_layout,
        selection_set
    },
    MotionBlurTilesFlatten{
        d, Shaders{{{ShaderType::eCompute, "MotionBlurTilesFlatten.comp"}}},
        vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(MotionBlurTilesFlattenPushConstants)},
        selection_layout,
        selection_set
    },
    MotionBlurTilesDilate{
        d, Shaders{{{ShaderType::eCompute, "MotionBlurTilesDilate.comp"}}},
        vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(MotionBlurTilesDilatePushConstants)},
        selection_layout,
        selection_set
    },
    IblPrefilter{d} {}

void Pipelines::SetExtent(vk::Extent2D extent) {
    Main.SetExtent(extent, Device, PhysicalDevice);
    Silhouette.SetExtent(extent, Device, PhysicalDevice);
    SilhouetteEdge.SetExtent(extent, Device, PhysicalDevice);
    SelectionFragment.SetExtent(extent, Device, PhysicalDevice, *Silhouette.Resources->DepthImage.View);
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
    MotionBlurTilesFlatten.Compile();
    MotionBlurTilesDilate.Compile();
}
