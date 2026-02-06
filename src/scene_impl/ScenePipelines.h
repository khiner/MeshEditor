#pragma once

using SPT = ShaderPipelineType;
enum class OverlayKind : uint32_t {
    Edge = 0,
    FaceNormal = 1,
    VertexNormal = 2,
};

namespace {
constexpr vk::ImageSubresourceRange DepthSubresourceRange{vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
constexpr vk::ImageSubresourceRange ColorSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

struct PipelineRenderer {
    vk::UniqueRenderPass RenderPass;
    std::unordered_map<SPT, ShaderPipeline> ShaderPipelines;

    void CompileShaders() {
        for (auto &shader_pipeline : std::views::values(ShaderPipelines)) shader_pipeline.Compile(*RenderPass);
    }

    const ShaderPipeline &Bind(vk::CommandBuffer cb, SPT spt) const {
        const auto &pipeline = ShaderPipelines.at(spt);
        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline.Pipeline);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline.PipelineLayout, 0, pipeline.GetDescriptorSet(), {});
        return pipeline;
    }
};

namespace Format {
constexpr auto Color = vk::Format::eB8G8R8A8Unorm;
constexpr auto Depth = vk::Format::eD32Sfloat;
constexpr auto Float = vk::Format::eR32Sfloat;
constexpr auto Float2 = vk::Format::eR32G32Sfloat;
constexpr auto Uint = vk::Format::eR32Uint;
} // namespace Format

struct MainPipeline {
    static PipelineRenderer CreateRenderer(
        vk::Device d, vk::SampleCountFlagBits msaa_samples,
        vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {}
    ) {
        const std::vector<vk::AttachmentDescription> attachments{
            // Depth attachment.
            {{}, Format::Depth, msaa_samples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
            // Multisampled offscreen image.
            {{}, Format::Color, msaa_samples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
            // Single-sampled resolve target. UNDEFINED + DONT_CARE = discard previous contents, let render pass handle transition.
            {{}, Format::Color, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        };
        const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
        const vk::AttachmentReference color_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
        const vk::AttachmentReference resolve_attachment_ref{2, vk::ImageLayout::eColorAttachmentOptimal};
        const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, &resolve_attachment_ref, &depth_attachment_ref};

        const PipelineContext ctx{d, shared_layout, shared_set, msaa_samples};
        const vk::PushConstantRange draw_pc{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPassPushConstants)};

        // Can't construct this map in-place with pairs because `ShaderPipeline` doesn't have a copy constructor.
        std::unordered_map<SPT, ShaderPipeline> pipelines;
        pipelines.emplace(
            SPT::Fill,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "Lighting.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
                CreateColorBlendAttachment(true), CreateDepthStencil(), draw_pc
            )
        );
        pipelines.emplace(
            SPT::Line,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "VertexColor.frag"}}},
                {},
                vk::PolygonMode::eLine, vk::PrimitiveTopology::eLineList,
                CreateColorBlendAttachment(true), CreateDepthStencil(), draw_pc
            )
        );
        const auto make_overlay_pipeline = [&](OverlayKind overlay_kind) {
            return ctx.CreateGraphics(
                {{{ShaderType::eVertex, "VertexTransform.vert", {{0, uint32_t(overlay_kind)}}},
                  {ShaderType::eFragment, "VertexColor.frag"}}},
                {},
                vk::PolygonMode::eLine, vk::PrimitiveTopology::eLineList,
                CreateColorBlendAttachment(true), CreateDepthStencil(), draw_pc
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
                CreateColorBlendAttachment(true), CreateDepthStencil(), draw_pc, -1.0f
            )
        );
        pipelines.emplace(
            SPT::Grid,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "GridLines.vert"}, {ShaderType::eFragment, "GridLines.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
                CreateColorBlendAttachment(true), CreateDepthStencil(true, false)
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
                CreateColorBlendAttachment(true), CreateDepthStencil(true, true, vk::CompareOp::eAlways),
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
                CreateColorBlendAttachment(true), CreateDepthStencil(false, false),
                vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t) * 3} // Manipulating flag + sampler index + active object id
            )
        );
        pipelines.emplace(
            SPT::DebugNormals,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "Normals.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
                CreateColorBlendAttachment(true), CreateDepthStencil(), draw_pc
            )
        );
        return {d.createRenderPassUnique({{}, attachments, subpass}), std::move(pipelines)};
    }

    struct ResourcesT {
        ResourcesT(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::SampleCountFlagBits msaa_samples, vk::RenderPass render_pass)
            : DepthImage{mvk::CreateImage(
                  d, pd,
                  {{},
                   vk::ImageType::e2D,
                   Format::Depth,
                   vk::Extent3D{extent, 1},
                   1,
                   1,
                   msaa_samples,
                   vk::ImageTiling::eOptimal,
                   vk::ImageUsageFlagBits::eDepthStencilAttachment,
                   vk::SharingMode::eExclusive},
                  {{}, {}, vk::ImageViewType::e2D, Format::Depth, {}, DepthSubresourceRange}
              )},
              OffscreenImage{mvk::CreateImage(
                  d, pd,
                  {{},
                   vk::ImageType::e2D,
                   Format::Color,
                   vk::Extent3D{extent, 1},
                   1,
                   1,
                   msaa_samples,
                   vk::ImageTiling::eOptimal,
                   vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
                   vk::SharingMode::eExclusive},
                  {{}, {}, vk::ImageViewType::e2D, Format::Color, {}, ColorSubresourceRange}
              )},
              ResolveImage{mvk::CreateImage(
                  d, pd,
                  {
                      {},
                      vk::ImageType::e2D,
                      Format::Color,
                      vk::Extent3D{extent, 1},
                      1,
                      1,
                      vk::SampleCountFlagBits::e1,
                      vk::ImageTiling::eOptimal,
                      vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
                      vk::SharingMode::eExclusive,
                  },
                  {{}, {}, vk::ImageViewType::e2D, Format::Color, {}, ColorSubresourceRange}
              )} {
            const std::array image_views{*DepthImage.View, *OffscreenImage.View, *ResolveImage.View};
            Framebuffer = d.createFramebufferUnique({{}, render_pass, image_views, extent.width, extent.height, 1});
        }

        // Perform depth testing, render into a multisampled offscreen image, and resolve into a single-sampled image.
        mvk::ImageResource DepthImage, OffscreenImage, ResolveImage;
        vk::UniqueFramebuffer Framebuffer;
    };

    void SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::SampleCountFlagBits msaa_samples) {
        Resources = std::make_unique<ResourcesT>(extent, d, pd, msaa_samples, *Renderer.RenderPass);
    }
    PipelineRenderer Renderer;
    std::unique_ptr<ResourcesT> Resources;
};

struct SilhouettePipeline {
    static PipelineRenderer CreateRenderer(vk::Device d, vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {}) {
        const std::vector<vk::AttachmentDescription> attachments{
            // Store depth for reuse by element selection (mutual occlusion between selected meshes).
            {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
            // Single-sampled offscreen "image" of two channels: depth and object ID.
            {{}, Format::Float2, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        };
        const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
        const vk::AttachmentReference color_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
        const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, &depth_attachment_ref};

        const PipelineContext ctx{d, shared_layout, shared_set, vk::SampleCountFlagBits::e1};
        const vk::PushConstantRange draw_pc{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPassPushConstants)};
        std::unordered_map<SPT, ShaderPipeline> pipelines;
        pipelines.emplace(
            SPT::SilhouetteDepthObject,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "DepthObject.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
                CreateColorBlendAttachment(false), CreateDepthStencil(), draw_pc
            )
        );
        return {d.createRenderPassUnique({{}, attachments, subpass}), std::move(pipelines)};
    }

    struct ResourcesT {
        ResourcesT(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass)
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

        mvk::ImageResource DepthImage, OffscreenImage;
        vk::UniqueSampler ImageSampler;
        vk::UniqueFramebuffer Framebuffer;
    };

    void SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd) {
        Resources = std::make_unique<ResourcesT>(extent, d, pd, *Renderer.RenderPass);
    }

    PipelineRenderer Renderer;
    std::unique_ptr<ResourcesT> Resources;
};

struct SilhouetteEdgePipeline {
    static PipelineRenderer CreateRenderer(vk::Device d, vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {}) {
        const std::vector<vk::AttachmentDescription> attachments{
            {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilReadOnlyOptimal},
            {{}, Format::Float, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        };
        const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
        const vk::AttachmentReference color_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
        const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, &depth_attachment_ref};

        const PipelineContext ctx{d, shared_layout, shared_set, vk::SampleCountFlagBits::e1};
        std::unordered_map<SPT, ShaderPipeline> pipelines;
        pipelines.emplace(
            SPT::SilhouetteEdgeDepthObject,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "SilhouetteEdgeDepthObject.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
                CreateColorBlendAttachment(false), CreateDepthStencil(),
                vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t) * 2}
            )
        );
        return {d.createRenderPassUnique({{}, attachments, subpass}), std::move(pipelines)};
    }

    struct ResourcesT {
        ResourcesT(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass)
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

        mvk::ImageResource DepthImage, OffscreenImage;
        vk::UniqueSampler ImageSampler, DepthSampler;
        vk::UniqueFramebuffer Framebuffer;
    };

    void SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd) {
        Resources = std::make_unique<ResourcesT>(extent, d, pd, *Renderer.RenderPass);
    }

    PipelineRenderer Renderer;
    std::unique_ptr<ResourcesT> Resources;
};

struct SelectionFragmentPipeline {
    // Render pass that loads depth from silhouette pass for element occlusion
    static PipelineRenderer CreateRenderer(vk::Device d, vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {}) {
        const std::vector<vk::AttachmentDescription> attachments{
            {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal},
        };
        const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
        const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 0, nullptr, nullptr, &depth_attachment_ref};

        const PipelineContext ctx{d, shared_layout, shared_set, vk::SampleCountFlagBits::e1};
        const vk::PushConstantRange draw_pc{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPassPushConstants)};
        std::unordered_map<SPT, ShaderPipeline> pipelines;
        pipelines.emplace(
            SPT::SelectionElementFace,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "SelectionElementFace.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
                {}, CreateDepthStencil(true, true, vk::CompareOp::eLess), draw_pc
            )
        );
        pipelines.emplace(
            SPT::SelectionElementFaceXRay,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "SelectionElementFace.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
                {}, CreateDepthStencil(false, false), draw_pc
            )
        );
        pipelines.emplace(
            SPT::SelectionElementEdge,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "SelectionElementEdge.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eLineList,
                {}, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual), draw_pc
            )
        );
        pipelines.emplace(
            SPT::SelectionElementEdgeXRay,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "SelectionElementEdge.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eLineList,
                {}, CreateDepthStencil(false, false), draw_pc
            )
        );
        pipelines.emplace(
            SPT::SelectionElementVertex,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "SelectionElementVertex.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::ePointList,
                {}, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual), draw_pc
            )
        );
        pipelines.emplace(
            SPT::SelectionElementVertexXRay,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "SelectionElementVertex.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::ePointList,
                {}, CreateDepthStencil(false, false), draw_pc
            )
        );
        pipelines.emplace(
            SPT::SelectionFragment,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "SelectionFragment.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
                {}, CreateDepthStencil(), draw_pc
            )
        );
        pipelines.emplace(
            SPT::SelectionFragmentXRay,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "SelectionFragment.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
                {}, CreateDepthStencil(false, false), draw_pc
            )
        );
        return {d.createRenderPassUnique({{}, attachments, subpass}), std::move(pipelines)};
    }

    struct ResourcesT {
        ResourcesT(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass, vk::ImageView silhouette_depth_view)
            : HeadImage{mvk::CreateImage(
                  d, pd,
                  {{},
                   vk::ImageType::e2D,
                   Format::Uint,
                   vk::Extent3D{extent, 1},
                   1,
                   1,
                   vk::SampleCountFlagBits::e1,
                   vk::ImageTiling::eOptimal,
                   vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst,
                   vk::SharingMode::eExclusive},
                  {{}, {}, vk::ImageViewType::e2D, Format::Uint, {}, ColorSubresourceRange}
              )},
              Framebuffer{d.createFramebufferUnique({{}, render_pass, silhouette_depth_view, extent.width, extent.height, 1})} {}

        mvk::ImageResource HeadImage;
        vk::UniqueFramebuffer Framebuffer;
    };

    void SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::ImageView silhouette_depth_view) {
        Resources = std::make_unique<ResourcesT>(extent, d, pd, *Renderer.RenderPass, silhouette_depth_view);
    }

    PipelineRenderer Renderer;
    std::unique_ptr<ResourcesT> Resources;
};

vk::SampleCountFlagBits GetMaxUsableSampleCount(vk::PhysicalDevice pd) {
    const auto props = pd.getProperties();
    const auto counts = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
    if (counts & vk::SampleCountFlagBits::e64) return vk::SampleCountFlagBits::e64;
    if (counts & vk::SampleCountFlagBits::e32) return vk::SampleCountFlagBits::e32;
    if (counts & vk::SampleCountFlagBits::e16) return vk::SampleCountFlagBits::e16;
    if (counts & vk::SampleCountFlagBits::e8) return vk::SampleCountFlagBits::e8;
    if (counts & vk::SampleCountFlagBits::e4) return vk::SampleCountFlagBits::e4;
    if (counts & vk::SampleCountFlagBits::e2) return vk::SampleCountFlagBits::e2;

    return vk::SampleCountFlagBits::e1;
}
} // namespace

struct ScenePipelines {
    ScenePipelines(
        vk::Device d, vk::PhysicalDevice pd,
        vk::DescriptorSetLayout selection_layout = {}, vk::DescriptorSet selection_set = {}
    ) : Device(d), PhysicalDevice(pd), Samples{GetMaxUsableSampleCount(pd)},
        Main{MainPipeline::CreateRenderer(d, Samples, selection_layout, selection_set), nullptr},
        Silhouette{SilhouettePipeline::CreateRenderer(d, selection_layout, selection_set), nullptr},
        SilhouetteEdge{SilhouetteEdgePipeline::CreateRenderer(d, selection_layout, selection_set), nullptr},
        SelectionFragment{SelectionFragmentPipeline::CreateRenderer(d, selection_layout, selection_set), nullptr},
        ClickSelect{
            d, Shaders{{{ShaderType::eCompute, "ClickSelect.comp"}}},
            vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ClickSelectPushConstants)},
            selection_layout,
            selection_set
        },
        ClickSelectElement{
            d, Shaders{{{ShaderType::eCompute, "ClickSelectElement.comp"}}},
            vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ClickSelectElementPushConstants)},
            selection_layout,
            selection_set
        },
        BoxSelect{
            d, Shaders{{{ShaderType::eCompute, "BoxSelect.comp"}}},
            vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(BoxSelectPushConstants)},
            selection_layout,
            selection_set
        } {}

    vk::Device Device;
    vk::PhysicalDevice PhysicalDevice;
    vk::SampleCountFlagBits Samples;

    MainPipeline Main;
    SilhouettePipeline Silhouette;
    SilhouetteEdgePipeline SilhouetteEdge;
    SelectionFragmentPipeline SelectionFragment;
    ComputePipeline ClickSelect;
    ComputePipeline ClickSelectElement;
    ComputePipeline BoxSelect;

    void SetExtent(vk::Extent2D);
    void CompileShaders() {
        Main.Renderer.CompileShaders();
        Silhouette.Renderer.CompileShaders();
        SilhouetteEdge.Renderer.CompileShaders();
        SelectionFragment.Renderer.CompileShaders();
        ClickSelect.Compile();
        ClickSelectElement.Compile();
        BoxSelect.Compile();
    }
};
