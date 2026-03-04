#include "ScenePipelines.h"
#include "Shader.h"
#include "gpu/BoxSelectPushConstants.h"
#include "gpu/DrawPassPushConstants.h"
#include "gpu/ElementPickPushConstants.h"
#include "gpu/ObjectPickPushConstants.h"
#include "gpu/SelectionElementPushConstants.h"
#include "gpu/UpdateSelectionStatePushConstants.h"

#include <ranges>

namespace {
enum class OverlayKind : uint32_t {
    Edge = 0,
    FaceNormal = 1,
    VertexNormal = 2,
};
constexpr std::array PbrSpecFeatures{PbrFeature::Punctual, PbrFeature::Transmission, PbrFeature::DiffuseTrans, PbrFeature::Clearcoat, PbrFeature::Sheen, PbrFeature::Anisotropy, PbrFeature::Iridescence};

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

// No-write for LineData attachment (non-line pipelines) - also used by PbrCompiler.
static constexpr vk::PipelineColorBlendAttachmentState NoWriteBlend{};

PbrCompiler::PbrCompiler(PipelineContext ctx, vk::RenderPass rp)
    : Device(ctx.Device), SetLayout(ctx.SharedLayout), Set(ctx.SharedSet), RenderPass(rp) {
    CompileModules();
    const vk::PushConstantRange draw_pc{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPassPushConstants)};
    Layout = Device.createPipelineLayoutUnique({{}, 1, &SetLayout, 1, &draw_pc});
}

void PbrCompiler::CompileModules() {
    VertModule = CompileShaderModule(Device, ShaderType::eVertex, "VertexTransform.vert");
    FragModule = CompileShaderModule(Device, ShaderType::eFragment, "pbr.frag");
}

vk::UniquePipeline PbrCompiler::CreateTargetedPipeline(const vk::SpecializationInfo &frag_spec, bool depth_write) const {
    static constexpr vk::PipelineViewportStateCreateInfo viewport_state{{}, 1, nullptr, 1, nullptr};
    static constexpr std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    static const vk::PipelineDynamicStateCreateInfo dynamic_state{{}, dynamic_states};
    static constexpr vk::PipelineMultisampleStateCreateInfo multisample_state{{}, vk::SampleCountFlagBits::e1};
    static constexpr vk::PipelineVertexInputStateCreateInfo vertex_input{};
    static constexpr vk::PipelineInputAssemblyStateCreateInfo input_assembly{{}, vk::PrimitiveTopology::eTriangleList};
    static const vk::PipelineRasterizationStateCreateInfo raster{{}, false, false, vk::PolygonMode::eFill, {}, vk::FrontFace::eClockwise, false, 0.f, {}, {}, 1.f};

    const std::array stages{
        vk::PipelineShaderStageCreateInfo{{}, ShaderType::eVertex, *VertModule, "main"},
        vk::PipelineShaderStageCreateInfo{{}, ShaderType::eFragment, *FragModule, "main", &frag_spec},
    };
    const auto depth_stencil = CreateDepthStencil(true, depth_write);
    const std::array color_blend_attachments{CreateColorBlendAttachment(true), NoWriteBlend};
    const vk::PipelineColorBlendStateCreateInfo color_blending{{}, false, vk::LogicOp::eCopy, color_blend_attachments.size(), color_blend_attachments.data()};
    auto result = Device.createGraphicsPipelineUnique(
        {},
        {{}, stages, &vertex_input, &input_assembly, nullptr, &viewport_state, &raster, &multisample_state, &depth_stencil, &color_blending, &dynamic_state, *Layout, RenderPass}
    );
    if (result.result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("PbrCompiler: failed to create targeted pipeline: {}", vk::to_string(result.result)));
    }
    return std::move(result.value);
}

bool PbrCompiler::CompilePipelines(PbrFeatureMask mask) {
    if (mask == Mask && OpaqueTargeted && BlendTargeted) return false;

    constexpr auto N = uint32_t(PbrSpecFeatures.size());
    std::array<uint32_t, PbrSpecFeatures.size()> data{};
    std::array<vk::SpecializationMapEntry, PbrSpecFeatures.size()> entries{};
    for (uint32_t i = 0; i < N; ++i) {
        data[i] = HasFeature(mask, PbrSpecFeatures[i]) ? 1u : 0u;
        entries[i] = vk::SpecializationMapEntry{i, i * uint32_t(sizeof(uint32_t)), uint32_t(sizeof(uint32_t))};
    }
    const vk::SpecializationInfo spec{N, entries.data(), N * sizeof(uint32_t), data.data()};
    OpaqueTargeted = CreateTargetedPipeline(spec, true);
    BlendTargeted = CreateTargetedPipeline(spec, false);
    Mask = mask;

    return true;
}

vk::PipelineLayout PbrCompiler::BindTargeted(vk::CommandBuffer cb, bool opaque) const {
    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, opaque ? *OpaqueTargeted : *BlendTargeted);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *Layout, 0, Set, {});
    return *Layout;
}

void PbrCompiler::RecompileModules() {
    if (!Device) return;
    CompileModules();
    OpaqueTargeted.reset();
    BlendTargeted.reset();
    CompilePipelines(Mask);
}

// Write-only for LineData attachment (used by line pipelines as their 2nd color blend state).
static const vk::PipelineColorBlendAttachmentState LineDataBlend = CreateColorBlendAttachment(false);

static PipelineRenderer CreateMainRenderer(
    vk::Device d,
    vk::DescriptorSetLayout shared_layout, vk::DescriptorSet shared_set
) {
    const std::vector<vk::AttachmentDescription> attachments{
        // Depth: cleared each frame, eDontCare store (no sampling after main pass).
        {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
        // Color: cleared to background, stored for sampling in line AA composite pass.
        {{}, Format::Color, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        // LineData: cleared to zero (alpha=0 means "no line here"), stored for sampling in line AA composite pass.
        {{}, Format::LineData, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
    const std::array color_attachment_refs{
        vk::AttachmentReference{1, vk::ImageLayout::eColorAttachmentOptimal}, // Color (location 0)
        vk::AttachmentReference{2, vk::ImageLayout::eColorAttachmentOptimal}, // LineData (location 1)
    };
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, uint32_t(color_attachment_refs.size()), color_attachment_refs.data(), nullptr, &depth_attachment_ref};

    const PipelineContext ctx{d, shared_layout, shared_set};
    const vk::PushConstantRange draw_pc{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPassPushConstants)};

    // Can't construct this map in-place with pairs because `ShaderPipeline` doesn't have a copy constructor.
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    pipelines.emplace(
        SPT::Fill,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "WorkspaceLighting.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            {CreateColorBlendAttachment(true), NoWriteBlend}, CreateDepthStencil(), draw_pc
        )
    );
    pipelines.emplace(
        SPT::Line,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "VertexTransform.vert", {{1, 1u}}}, {ShaderType::eFragment, "VertexColor.frag"}}},
            {},
            vk::PolygonMode::eLine, vk::PrimitiveTopology::eLineList,
            {CreateColorBlendAttachment(true), LineDataBlend}, CreateDepthStencil(), draw_pc
        )
    );
    pipelines.emplace(
        SPT::ObjectExtrasLine,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "ObjectExtras.vert"}, {ShaderType::eFragment, "VertexColor.frag"}}},
            {},
            vk::PolygonMode::eLine, vk::PrimitiveTopology::eLineList,
            {CreateColorBlendAttachment(true), LineDataBlend}, CreateDepthStencil(), draw_pc
        )
    );
    const auto make_overlay_pipeline = [&](OverlayKind overlay_kind) {
        return ctx.CreateGraphics(
            {{{ShaderType::eVertex, "VertexTransform.vert", {{0, uint32_t(overlay_kind)}, {1, 1u}}},
              {ShaderType::eFragment, "VertexColor.frag"}}},
            {},
            vk::PolygonMode::eLine, vk::PrimitiveTopology::eLineList,
            {CreateColorBlendAttachment(true), LineDataBlend}, CreateDepthStencil(), draw_pc
        );
    };
    pipelines.emplace(SPT::LineOverlayFaceNormals, make_overlay_pipeline(OverlayKind::FaceNormal));
    pipelines.emplace(SPT::LineOverlayVertexNormals, make_overlay_pipeline(OverlayKind::VertexNormal));
    pipelines.emplace(SPT::LineOverlayBBox, make_overlay_pipeline(OverlayKind::Edge));
    pipelines.emplace(
        SPT::Point,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "VertexPoint.vert"}, {ShaderType::eFragment, "VertexPoint.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::ePointList,
            {CreateColorBlendAttachment(true), NoWriteBlend}, CreateDepthStencil(), draw_pc, -1.0f
        )
    );
    pipelines.emplace(
        SPT::Grid,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "GridLines.vert"}, {ShaderType::eFragment, "GridLines.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
            {CreateColorBlendAttachment(true), NoWriteBlend}, CreateDepthStencil(true, false)
        )
    );
    pipelines.emplace(
        SPT::Background,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "Background.vert"}, {ShaderType::eFragment, "Background.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
            {CreateColorBlendAttachment(true), NoWriteBlend}, CreateDepthStencil(false, false)
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
            {CreateColorBlendAttachment(true), NoWriteBlend}, CreateDepthStencil(true, true, vk::CompareOp::eAlways),
            vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t)}
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
        SPT::DebugNormals,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "Normals.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            {CreateColorBlendAttachment(true), NoWriteBlend}, CreateDepthStencil(), draw_pc
        )
    );
    return {d.createRenderPassUnique({{}, attachments, subpass}), std::move(pipelines)};
}

static vk::UniqueRenderPass CreateLineAARenderPass(vk::Device d) {
    const vk::AttachmentDescription attachment{
        {},
        Format::Color,
        vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eStore,
        {},
        {},
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    const vk::AttachmentReference color_ref{0, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_ref};
    return d.createRenderPassUnique({{}, attachment, subpass});
}

MainPipeline::MainPipeline(
    vk::Device d,
    vk::DescriptorSetLayout shared_layout, vk::DescriptorSet shared_set
) : Renderer{CreateMainRenderer(d, shared_layout, shared_set)},
    LineAARenderPass{CreateLineAARenderPass(d)},
    LineAAComposite{
        d,
        Shaders{{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "LineAA.frag"}}},
        {},
        vk::PolygonMode::eFill,
        vk::PrimitiveTopology::eTriangleStrip,
        {CreateColorBlendAttachment(false)},
        {},
        vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t) * 2},
        0.f,
        shared_layout,
        shared_set
    },
    Compiler{{d, shared_layout, shared_set}, *Renderer.RenderPass} {}

MainPipeline::ResourcesT::ResourcesT(
    vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass, vk::RenderPass line_aa_render_pass
) : DepthImage{mvk::CreateImage(d, pd, {{}, vk::ImageType::e2D, Format::Depth, vk::Extent3D{extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::SharingMode::eExclusive}, {{}, {}, vk::ImageViewType::e2D, Format::Depth, {}, DepthSubresourceRange})},
    ColorImage{mvk::CreateImage(d, pd, {{}, vk::ImageType::e2D, Format::Color, vk::Extent3D{extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive}, {{}, {}, vk::ImageViewType::e2D, Format::Color, {}, ColorSubresourceRange})},
    LineDataImage{mvk::CreateImage(d, pd, {{}, vk::ImageType::e2D, Format::LineData, vk::Extent3D{extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive}, {{}, {}, vk::ImageViewType::e2D, Format::LineData, {}, ColorSubresourceRange})},
    FinalColorImage{mvk::CreateImage(d, pd, {{}, vk::ImageType::e2D, Format::Color, vk::Extent3D{extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive}, {{}, {}, vk::ImageViewType::e2D, Format::Color, {}, ColorSubresourceRange})},
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
        const std::array image_views{*DepthImage.View, *ColorImage.View, *LineDataImage.View};
        Framebuffer = d.createFramebufferUnique({{}, render_pass, image_views, extent.width, extent.height, 1});
    }
    {
        const std::array image_views{*FinalColorImage.View};
        LineAAFramebuffer = d.createFramebufferUnique({{}, line_aa_render_pass, image_views, extent.width, extent.height, 1});
    }
}

void MainPipeline::SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd) {
    Resources = std::make_unique<ResourcesT>(extent, d, pd, *Renderer.RenderPass, *LineAARenderPass);
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
    const vk::PushConstantRange draw_pc{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPassPushConstants)};
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    pipelines.emplace(
        SPT::SilhouetteDepthObject,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "DepthObject.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            {CreateColorBlendAttachment(false)}, CreateDepthStencil(), draw_pc
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
    const vk::PushConstantRange draw_pc{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPassPushConstants)};
    const vk::PushConstantRange element_pc{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(SelectionElementPushConstants)};
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    pipelines.emplace(
        SPT::SelectionElementFace,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "SelectionElementFace.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            {}, CreateDepthStencil(true, true, vk::CompareOp::eLess), element_pc
        )
    );
    pipelines.emplace(
        SPT::SelectionElementFaceXRay,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "SelectionElementFace.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            {}, CreateDepthStencil(false, false), element_pc
        )
    );
    pipelines.emplace(
        SPT::SelectionElementEdge,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "SelectionElementEdge.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eLineList,
            {}, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual), element_pc
        )
    );
    pipelines.emplace(
        SPT::SelectionElementEdgeXRay,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "SelectionElementEdge.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eLineList,
            {}, CreateDepthStencil(false, false), element_pc
        )
    );
    pipelines.emplace(
        SPT::SelectionElementEdgeXRayVerts,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "SelectionElementEdge.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::ePointList,
            {}, CreateDepthStencil(false, false), element_pc
        )
    );
    pipelines.emplace(
        SPT::SelectionElementVertex,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "SelectionElementVertex.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::ePointList,
            {}, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual), element_pc
        )
    );
    pipelines.emplace(
        SPT::SelectionElementVertexXRay,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "SelectionElementVertex.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::ePointList,
            {}, CreateDepthStencil(false, false), element_pc
        )
    );
    pipelines.emplace(
        SPT::SelectionElementFaceXRayVerts,
        ctx.CreateGraphics(
            {{{ShaderType::eVertex, "SelectionElementFace.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
            {},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::ePointList,
            {}, CreateDepthStencil(false, false), element_pc
        )
    );
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

ScenePipelines::ScenePipelines(
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
    IblPrefilter{d} {}

void ScenePipelines::SetExtent(vk::Extent2D extent) {
    Main.SetExtent(extent, Device, PhysicalDevice);
    Silhouette.SetExtent(extent, Device, PhysicalDevice);
    SilhouetteEdge.SetExtent(extent, Device, PhysicalDevice);
    SelectionFragment.SetExtent(extent, Device, PhysicalDevice, *Silhouette.Resources->DepthImage.View);
}

void ScenePipelines::CompileShaders() {
    Main.Renderer.CompileShaders();
    Main.LineAAComposite.Compile(*Main.LineAARenderPass);
    Main.Compiler.RecompileModules();
    Silhouette.Renderer.CompileShaders();
    SilhouetteEdge.Renderer.CompileShaders();
    SelectionFragment.Renderer.CompileShaders();
    ObjectPick.Compile();
    ElementPick.Compile();
    BoxSelect.Compile();
    UpdateSelectionState.Compile();
}
