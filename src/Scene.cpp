#include "Scene.h"

#include <format>
#include <ranges>

#include "imgui.h"

#include "ImGuizmo.h"

#include "Geometry/GeometryInstance.h"
#include "Geometry/Primitive/Cuboid.h"

using glm::vec3, glm::vec4, glm::mat4;

static const vk::ClearColorValue transparent(0.f, 0.f, 0.f, 0.f);

static vk::SampleCountFlagBits GetMaxUsableSampleCount(const vk::PhysicalDevice physical_device) {
    const auto props = physical_device.getProperties();
    const auto counts = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
    if (counts & vk::SampleCountFlagBits::e64) return vk::SampleCountFlagBits::e64;
    if (counts & vk::SampleCountFlagBits::e32) return vk::SampleCountFlagBits::e32;
    if (counts & vk::SampleCountFlagBits::e16) return vk::SampleCountFlagBits::e16;
    if (counts & vk::SampleCountFlagBits::e8) return vk::SampleCountFlagBits::e8;
    if (counts & vk::SampleCountFlagBits::e4) return vk::SampleCountFlagBits::e4;
    if (counts & vk::SampleCountFlagBits::e2) return vk::SampleCountFlagBits::e2;

    return vk::SampleCountFlagBits::e1;
}

static const auto ImageFormat = vk::Format::eB8G8R8A8Unorm;

struct Gizmo {
    void Begin() const {
        using namespace ImGui;

        const auto content_region = GetContentRegionAvail();
        const auto &window_pos = GetWindowPos();
        ImGuizmo::BeginFrame();
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetRect(window_pos.x, window_pos.y + GetTextLineHeightWithSpacing(), content_region.x, content_region.y);
    }

    bool Render(Camera &camera, mat4 &model, float aspect_ratio = 1) const {
        using namespace ImGui;

        static const float ViewManipulateSize = 128;
        const auto &window_pos = GetWindowPos();
        const auto view_manipulate_pos = window_pos + ImVec2{GetWindowContentRegionMax().x - ViewManipulateSize, GetWindowContentRegionMin().y};
        auto camera_view = camera.GetViewMatrix();
        const float camera_distance = camera.GetDistance();
        const bool view_changed = ImGuizmo::ViewManipulate(&camera_view[0][0], camera_distance, view_manipulate_pos, {ViewManipulateSize, ViewManipulateSize}, 0);
        if (view_changed) camera.SetPositionFromView(camera_view);

        auto camera_projection = camera.GetProjectionMatrix(aspect_ratio);
        const bool model_changed = ShowModelGizmo && ImGuizmo::Manipulate(&camera_view[0][0], &camera_projection[0][0], ActiveOp, ImGuizmo::LOCAL, &model[0][0]);
        return view_changed || model_changed;
    }

    void RenderDebug() {
        using namespace ImGui;
        using namespace ImGuizmo;

        SeparatorText("Gizmo");
        Checkbox("Show gizmo", &ShowModelGizmo);
        if (!ShowModelGizmo) return;

        const char *interaction_text =
            IsUsing()         ? "Using Gizmo" :
            IsOver(TRANSLATE) ? "Translate hovered" :
            IsOver(ROTATE)    ? "Rotate hovered" :
            IsOver(SCALE)     ? "Scale hovered" :
            IsOver()          ? "Hovered" :
                                "Not interacting";
        Text("Interaction: %s", interaction_text);

        if (IsKeyPressed(ImGuiKey_T)) ActiveOp = TRANSLATE;
        if (IsKeyPressed(ImGuiKey_R)) ActiveOp = ROTATE;
        if (IsKeyPressed(ImGuiKey_S)) ActiveOp = SCALE;
        if (RadioButton("Translate (T)", ActiveOp == TRANSLATE)) ActiveOp = TRANSLATE;
        if (RadioButton("Rotate (R)", ActiveOp == ROTATE)) ActiveOp = ROTATE;
        if (RadioButton("Scale (S)", ActiveOp == SCALE)) ActiveOp = SCALE;
        if (RadioButton("Universal", ActiveOp == UNIVERSAL)) ActiveOp = UNIVERSAL;
        // Checkbox("Bound sizing", &ShowBounds);
    }

    ImGuizmo::OPERATION ActiveOp{ImGuizmo::TRANSLATE};
    bool ShowBounds{false};
    bool ShowModelGizmo{false};
};

void ImageResource::Create(const VulkanContext &vc, vk::ImageCreateInfo image_info, vk::ImageViewCreateInfo view_info, vk::MemoryPropertyFlags mem_props) {
    const auto &device = vc.Device;
    Image = device->createImageUnique(image_info);
    const auto mem_reqs = device->getImageMemoryRequirements(*Image);
    Memory = device->allocateMemoryUnique({mem_reqs.size, vc.FindMemoryType(mem_reqs.memoryTypeBits, mem_props)});
    device->bindImageMemory(*Image, *Memory, 0);
    view_info.image = *Image;
    View = device->createImageViewUnique(view_info);
}

RenderPipeline::RenderPipeline(const VulkanContext &vc) : VC(vc) {}
RenderPipeline::~RenderPipeline() = default;

void RenderPipeline::CompileShaders() {
    for (auto &shader_pipeline : std::views::values(ShaderPipelines)) shader_pipeline->Compile(*RenderPass);
}

void RenderPipeline::RenderGeometryBuffers(vk::CommandBuffer command_buffer, const GeometryInstance &geometry_instance, SPT spt, GeometryMode mode) const {
    const auto &buffers = geometry_instance.GetBuffers(mode);
    const auto &shader_pipeline = *ShaderPipelines.at(spt);
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shader_pipeline.Pipeline);
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shader_pipeline.PipelineLayout, 0, *shader_pipeline.DescriptorSet, {});
    static const vk::DeviceSize vertex_buffer_offsets[] = {0};
    command_buffer.bindVertexBuffers(0, *buffers.VertexBuffer.Buffer, vertex_buffer_offsets);
    command_buffer.bindIndexBuffer(*buffers.IndexBuffer.Buffer, 0, vk::IndexType::eUint32);
    command_buffer.drawIndexed(buffers.IndexBuffer.Size / sizeof(uint), 1, 0, 0, 0);
}

MainRenderPipeline::MainRenderPipeline(const VulkanContext &vc)
    : RenderPipeline(vc), MsaaSamples(GetMaxUsableSampleCount(VC.PhysicalDevice)) {
    const std::vector<vk::AttachmentDescription> attachments{
        // Depth attachment.
        {{}, vk::Format::eD32Sfloat, MsaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
        // Multisampled offscreen image.
        {{}, ImageFormat, MsaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        // Single-sampled resolve.
        {{}, ImageFormat, vk::SampleCountFlagBits::e1, {}, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
    const vk::AttachmentReference color_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::AttachmentReference resolve_attachment_ref{2, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, &resolve_attachment_ref, &depth_attachment_ref};
    RenderPass = VC.Device->createRenderPassUnique({{}, attachments, subpass});

    ShaderPipelines[SPT::Fill] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "Lighting.frag"}}},
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList, GenerateColorBlendAttachment(true), GenerateDepthStencil(), MsaaSamples
    );
    ShaderPipelines[SPT::Line] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "VertexColor.frag"}}},
        vk::PolygonMode::eLine, vk::PrimitiveTopology::eLineList, GenerateColorBlendAttachment(true), GenerateDepthStencil(), MsaaSamples
    );
    ShaderPipelines[SPT::Grid] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "GridLines.vert"}, {ShaderType::eFragment, "GridLines.frag"}}},
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip, GenerateColorBlendAttachment(true), GenerateDepthStencil(true, false), MsaaSamples
    );
    ShaderPipelines[SPT::Texture] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "SilhouetteEdgeTexture.frag"}}},
        // For the silhouette edge texture, we want to render all its pixels, but also explicitly override the depth buffer to make edge pixels "stick" to the geometry they are derived from.
        // We should be able to just set depth testing to false and depth writing to true, but it seems that some GPUs or drivers optimize out depth writes when depth testing is disabled,
        // so instead we configure a depth test that always passes.
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip, GenerateColorBlendAttachment(true), GenerateDepthStencil(true, true, vk::CompareOp::eAlways), MsaaSamples
    );
}

void MainRenderPipeline::UpdateDescriptors(
    vk::DescriptorBufferInfo transform,
    vk::DescriptorBufferInfo light,
    vk::DescriptorBufferInfo view_proj,
    vk::DescriptorBufferInfo view_proj_near_far,
    vk::DescriptorBufferInfo silhouette_display
) const {
    const auto &fill_sp = ShaderPipelines.at(SPT::Fill);
    const auto &line_sp = ShaderPipelines.at(SPT::Line);
    const auto &grid_sp = ShaderPipelines.at(SPT::Grid);
    const auto &texture_sp = ShaderPipelines.at(SPT::Texture);
    const std::vector<vk::WriteDescriptorSet> write_descriptor_sets{
        {*fill_sp->DescriptorSet, fill_sp->GetBinding("TransformUBO"), 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &transform},
        {*fill_sp->DescriptorSet, fill_sp->GetBinding("LightUBO"), 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &light},
        {*line_sp->DescriptorSet, line_sp->GetBinding("TransformUBO"), 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &transform},
        {*grid_sp->DescriptorSet, grid_sp->GetBinding("ViewProjectionUBO"), 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &view_proj},
        {*grid_sp->DescriptorSet, grid_sp->GetBinding("ViewProjNearFarUBO"), 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &view_proj_near_far},
        {*texture_sp->DescriptorSet, texture_sp->GetBinding("SilhouetteDisplayUBO"), 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &silhouette_display},
    };
    VC.Device->updateDescriptorSets(write_descriptor_sets, {});
}

void MainRenderPipeline::UpdateImageDescriptors(vk::DescriptorImageInfo silhouette_edge_image) const {
    const auto &sp = ShaderPipelines.at(SPT::Texture);
    const std::vector<vk::WriteDescriptorSet> write_descriptor_sets{
        {*sp->DescriptorSet, sp->GetBinding("SilhouetteEdgeTexture"), 0, 1, vk::DescriptorType::eCombinedImageSampler, &silhouette_edge_image},
    };
    VC.Device->updateDescriptorSets(write_descriptor_sets, {});
}

void MainRenderPipeline::SetExtent(vk::Extent2D extent) {
    Extent = extent;
    const vk::Extent3D e3d{Extent, 1};
    DepthImage.Create(
        VC,
        {{}, vk::ImageType::e2D, vk::Format::eD32Sfloat, e3d, 1, 1, MsaaSamples, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, vk::Format::eD32Sfloat, {}, {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}}
    );
    OffscreenImage.Create(
        VC,
        {{}, vk::ImageType::e2D, ImageFormat, e3d, 1, 1, MsaaSamples, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, ImageFormat, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
    );
    ResolveImage.Create(
        VC,
        {{}, vk::ImageType::e2D, ImageFormat, e3d, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, ImageFormat, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
    );
    const std::array image_views{*DepthImage.View, *OffscreenImage.View, *ResolveImage.View};
    Framebuffer = VC.Device->createFramebufferUnique({{}, *RenderPass, image_views, Extent.width, Extent.height, 1});
}

void MainRenderPipeline::Begin(vk::CommandBuffer command_buffer, const vk::ClearColorValue &background_color) const {
    const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {background_color}};
    command_buffer.beginRenderPass({*RenderPass, *Framebuffer, vk::Rect2D{{0, 0}, Extent}, clear_values}, vk::SubpassContents::eInline);
}

SilhouetteRenderPipeline::SilhouetteRenderPipeline(const VulkanContext &vc) : RenderPipeline(vc) {
    const std::vector<vk::AttachmentDescription> attachments{
        // Single-sampled offscreen image.
        {{}, ImageFormat, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference color_attachment_ref{0, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, nullptr};
    RenderPass = VC.Device->createRenderPassUnique({{}, attachments, subpass});

    ShaderPipelines[SPT::Silhouette] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "Depth.frag"}}},
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList, GenerateColorBlendAttachment(false), std::nullopt, vk::SampleCountFlagBits::e1
    );
}

void SilhouetteRenderPipeline::UpdateDescriptors(vk::DescriptorBufferInfo transform) const {
    const auto &sp = ShaderPipelines.at(SPT::Silhouette);
    const std::vector<vk::WriteDescriptorSet> write_descriptor_sets{
        {*sp->DescriptorSet, sp->GetBinding("TransformUBO"), 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &transform},
    };
    VC.Device->updateDescriptorSets(write_descriptor_sets, {});
}

void SilhouetteRenderPipeline::SetExtent(vk::Extent2D extent) {
    Extent = extent;
    OffscreenImage.Create(
        VC,
        {{}, vk::ImageType::e2D, ImageFormat, vk::Extent3D{Extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, ImageFormat, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
    );

    const std::array image_views{*OffscreenImage.View};
    Framebuffer = VC.Device->createFramebufferUnique({{}, *RenderPass, image_views, Extent.width, Extent.height, 1});
}

void SilhouetteRenderPipeline::Begin(vk::CommandBuffer command_buffer) const {
    static const std::vector<vk::ClearValue> clear_values{{transparent}};
    command_buffer.beginRenderPass({*RenderPass, *Framebuffer, vk::Rect2D{{0, 0}, Extent}, clear_values}, vk::SubpassContents::eInline);
}

EdgeDetectionRenderPipeline::EdgeDetectionRenderPipeline(const VulkanContext &vc) : RenderPipeline(vc) {
    const std::vector<vk::AttachmentDescription> attachments{
        // Single-sampled offscreen image.
        {{}, ImageFormat, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference color_attachment_ref{0, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, nullptr};
    RenderPass = VC.Device->createRenderPassUnique({{}, attachments, subpass});

    ShaderPipelines[SPT::EdgeDetection] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "Sobel.frag"}}},
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip, GenerateColorBlendAttachment(false), std::nullopt, vk::SampleCountFlagBits::e1
    );
}

void EdgeDetectionRenderPipeline::UpdateImageDescriptors(vk::DescriptorImageInfo silhouette_fill_image) const {
    const auto &sp = ShaderPipelines.at(SPT::EdgeDetection);
    const std::vector<vk::WriteDescriptorSet> write_descriptor_sets{
        {*sp->DescriptorSet, sp->GetBinding("Tex"), 0, 1, vk::DescriptorType::eCombinedImageSampler, &silhouette_fill_image},
    };
    VC.Device->updateDescriptorSets(write_descriptor_sets, {});
}

void EdgeDetectionRenderPipeline::SetExtent(vk::Extent2D extent) {
    Extent = extent;
    OffscreenImage.Create(
        VC,
        {{}, vk::ImageType::e2D, ImageFormat, vk::Extent3D{Extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, ImageFormat, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
    );

    const std::array image_views{*OffscreenImage.View};
    Framebuffer = VC.Device->createFramebufferUnique({{}, *RenderPass, image_views, Extent.width, Extent.height, 1});
}

void EdgeDetectionRenderPipeline::Begin(vk::CommandBuffer command_buffer) const {
    static const std::vector<vk::ClearValue> clear_values{{transparent}};
    command_buffer.beginRenderPass({*RenderPass, *Framebuffer, vk::Rect2D{{0, 0}, Extent}, clear_values}, vk::SubpassContents::eInline);
}

Scene::Scene(const VulkanContext &vc)
    : VC(vc), MainRenderPipeline(VC), SilhouetteRenderPipeline(VC), EdgeDetectionRenderPipeline(VC) {
    GeometryInstances.push_back(std::make_unique<GeometryInstance>(VC, Cuboid{{0.5, 0.5, 0.5}}));
    UpdateGeometryEdgeColors();
    UpdateTransform();
    VC.CreateOrUpdateBuffer(LightBuffer, &Light);
    VC.CreateOrUpdateBuffer(SilhouetteDisplayBuffer, &SilhouetteDisplay);
    MainRenderPipeline.UpdateDescriptors(
        {*TransformBuffer.Buffer, 0, VK_WHOLE_SIZE},
        {*LightBuffer.Buffer, 0, VK_WHOLE_SIZE},
        {*ViewProjectionBuffer.Buffer, 0, VK_WHOLE_SIZE},
        {*ViewProjNearFarBuffer.Buffer, 0, VK_WHOLE_SIZE},
        {*SilhouetteDisplayBuffer.Buffer, 0, VK_WHOLE_SIZE}
    );
    SilhouetteRenderPipeline.UpdateDescriptors({*TransformBuffer.Buffer, 0, VK_WHOLE_SIZE});

    Gizmo = std::make_unique<::Gizmo>();
    CompileShaders();
}

Scene::~Scene(){}; // Using unique handles, so no need to manually destroy anything.

void Scene::SetExtent(vk::Extent2D extent) {
    Extent = extent;
    UpdateTransform(); // Transform depends on the aspect ratio.
    MainRenderPipeline.SetExtent(extent);
    SilhouetteRenderPipeline.SetExtent(extent);
    SilhouetteFillImageSampler = VC.Device->createSamplerUnique({
        {},
        vk::Filter::eLinear,
        vk::Filter::eLinear,
        vk::SamplerMipmapMode::eLinear,
        // Prevent edge detection from wrapping around to the other side of the image.
        // Instead, use the pixel value at the nearest edge.
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
    });

    EdgeDetectionRenderPipeline.UpdateImageDescriptors({*SilhouetteFillImageSampler, *SilhouetteRenderPipeline.OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal});
    EdgeDetectionRenderPipeline.SetExtent(extent);
    SilhouetteEdgeImageSampler = VC.Device->createSamplerUnique({{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear});
    MainRenderPipeline.UpdateImageDescriptors({*SilhouetteEdgeImageSampler, *EdgeDetectionRenderPipeline.OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal});
}

void Scene::RecordCommandBuffer() {
    const auto command_buffer = *VC.CommandBuffers[0];
    command_buffer.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});
    command_buffer.setViewport(0, vk::Viewport{0.f, 0.f, float(Extent.width), float(Extent.height), 0.f, 1.f});
    command_buffer.setScissor(0, vk::Rect2D{{0, 0}, Extent});

    const std::vector<vk::ImageMemoryBarrier> image_memory_barriers{{
        {},
        {},
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        *MainRenderPipeline.ResolveImage,
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }};
    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, // No dependency flags.
        {}, // No memory barriers.
        {}, // No buffer memory barriers.
        image_memory_barriers
    );

    const auto &geometry_instance = *GeometryInstances[0];

    SilhouetteRenderPipeline.Begin(command_buffer);
    SilhouetteRenderPipeline.RenderGeometryBuffers(command_buffer, geometry_instance, SPT::Silhouette, GeometryMode::Vertices);
    command_buffer.endRenderPass();

    EdgeDetectionRenderPipeline.Begin(command_buffer);
    EdgeDetectionRenderPipeline.GetShaderPipeline(SPT::EdgeDetection)->RenderQuad(command_buffer);
    command_buffer.endRenderPass();

    MainRenderPipeline.Begin(command_buffer, BackgroundColor);
    if (Mode == RenderMode::Faces) {
        MainRenderPipeline.RenderGeometryBuffers(command_buffer, geometry_instance, SPT::Fill, GeometryMode::Faces);
    } else if (Mode == RenderMode::Edges) {
        MainRenderPipeline.RenderGeometryBuffers(command_buffer, geometry_instance, SPT::Line, GeometryMode::Edges);
    } else if (Mode == RenderMode::FacesAndEdges) {
        MainRenderPipeline.RenderGeometryBuffers(command_buffer, geometry_instance, SPT::Fill, GeometryMode::Faces);
        MainRenderPipeline.RenderGeometryBuffers(command_buffer, geometry_instance, SPT::Line, GeometryMode::Edges);
    } else if (Mode == RenderMode::Smooth) {
        MainRenderPipeline.RenderGeometryBuffers(command_buffer, geometry_instance, SPT::Fill, GeometryMode::Vertices);
    }
    MainRenderPipeline.GetShaderPipeline(SPT::Texture)->RenderQuad(command_buffer);
    if (ShowGrid) MainRenderPipeline.GetShaderPipeline(SPT::Grid)->RenderQuad(command_buffer);

    command_buffer.endRenderPass();

    command_buffer.end();
}

void Scene::SubmitCommandBuffer(vk::Fence fence) const {
    vk::SubmitInfo submit;
    submit.setCommandBuffers(*VC.CommandBuffers[0]);
    VC.Queue.submit(submit, fence);
}

void Scene::CompileShaders() {
    MainRenderPipeline.CompileShaders();
    SilhouetteRenderPipeline.CompileShaders();
    EdgeDetectionRenderPipeline.CompileShaders();
}

void Scene::UpdateGeometryEdgeColors() {
    const auto &edge_color = Mode == RenderMode::FacesAndEdges ? MeshEdgeColor : EdgeColor;
    for (auto &geometry : GeometryInstances) geometry->SetEdgeColor(edge_color);
}

void Scene::UpdateTransform() {
    const float aspect_ratio = Extent.width == 0 || Extent.height == 0 ? 1.f : float(Extent.width) / float(Extent.height);
    const Transform transform{GeometryInstances[0]->Model, Camera.GetViewMatrix(), Camera.GetProjectionMatrix(aspect_ratio)};
    VC.CreateOrUpdateBuffer(TransformBuffer, &transform);
    const ViewProjection vp{transform.View, transform.Projection};
    VC.CreateOrUpdateBuffer(ViewProjectionBuffer, &vp);
    const ViewProjNearFar vpnf{vp.View, vp.Projection, Camera.NearClip, Camera.FarClip};
    VC.CreateOrUpdateBuffer(ViewProjNearFarBuffer, &vpnf);
}

using namespace ImGui;

static vk::ClearColorValue GlmToClearColor(const glm::vec4 &v) { return {v.r, v.g, v.b, v.a}; }

bool Scene::Render() {
    const auto new_extent = GetContentRegionAvail();

    // Handle mouse input.
    if (SelectionMode != SelectionMode::None) {
        auto &geometry = *GeometryInstances[0];
        const auto &mouse_pos = GetMousePos();
        const auto &window_pos = GetCursorScreenPos();
        const glm::vec2 mouse_pos_window = {mouse_pos.x - window_pos.x, mouse_pos.y - window_pos.y};
        const glm::vec2 mouse_pos_clip = {2.f * mouse_pos_window.x / new_extent.x - 1.f, 1.f - 2.f * mouse_pos_window.y / new_extent.y};
        const float aspect_ratio = float(Extent.width) / float(Extent.height);
        const Ray ray = Camera.ClipPosToWorldRay(mouse_pos_clip, aspect_ratio);
        if (SelectionMode == SelectionMode::Face) {
            if (geometry.HighlightFace(geometry.FindFirstIntersectingFace(ray))) SubmitCommandBuffer();
        } else if (SelectionMode == SelectionMode::Vertex) {
            if (geometry.HighlightVertex(geometry.FindNearestVertex(ray))) SubmitCommandBuffer();
        } else if (SelectionMode == SelectionMode::Edge) {
            if (geometry.HighlightEdge(geometry.FindNearestEdge(ray))) SubmitCommandBuffer();
        }
    }

    const auto &bg_color = GlmToClearColor(BgColor);
    const bool extent_changed = Extent.width != new_extent.x || Extent.height != new_extent.y;
    const bool bg_color_changed = BackgroundColor.float32 != bg_color.float32;
    if (!extent_changed && !bg_color_changed) return false;

    BackgroundColor = bg_color;

    if (extent_changed) SetExtent({uint(new_extent.x), uint(new_extent.y)});
    if (extent_changed || bg_color_changed) RecordCommandBuffer();
    SubmitCommandBuffer(*VC.RenderFence);

    // The contract is that the caller may use the resolve image and sampler immediately after `Scene::Render` returns.
    // Returning `true` indicates that the resolve image/sampler have been recreated.
    auto wait_result = VC.Device->waitForFences(*VC.RenderFence, VK_TRUE, UINT64_MAX);
    if (wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    VC.Device->resetFences(*VC.RenderFence);

    return extent_changed;
}

void Scene::RenderGizmo() {
    // Handle mouse scroll.
    const float mouse_wheel = GetIO().MouseWheel;
    if (mouse_wheel != 0 && IsWindowHovered()) Camera.SetTargetDistance(Camera.GetDistance() * (1.f - mouse_wheel / 16.f));

    Gizmo->Begin();
    const float aspect_ratio = float(Extent.width) / float(Extent.height);
    auto &model = GeometryInstances[0]->Model;
    bool view_or_model_changed = Gizmo->Render(Camera, model, aspect_ratio);
    view_or_model_changed |= Camera.Tick();
    if (view_or_model_changed) {
        UpdateTransform();
        SubmitCommandBuffer();
    }
}

void Scene::RenderControls() {
    if (BeginTabBar("Scene controls")) {
        if (BeginTabItem("Camera")) {
            bool camera_changed = false;
            if (Button("Reset camera")) {
                Camera.Position = {0, 0, 2};
                Camera.Target = {0, 0, 0};
                camera_changed = true;
            }
            camera_changed |= SliderFloat3("Position", &Camera.Position.x, -10, 10);
            camera_changed |= SliderFloat3("Target", &Camera.Target.x, -10, 10);
            camera_changed |= SliderFloat("Field of view (deg)", &Camera.FieldOfView, 1, 180);
            camera_changed |= SliderFloat("Near clip", &Camera.NearClip, 0.001f, 10, "%.3f", ImGuiSliderFlags_Logarithmic);
            camera_changed |= SliderFloat("Far clip", &Camera.FarClip, 10, 1000, "%.1f", ImGuiSliderFlags_Logarithmic);
            if (camera_changed) {
                Camera.StopMoving();
                UpdateTransform();
                SubmitCommandBuffer();
            }
            EndTabItem();
        }
        if (BeginTabItem("Light")) {
            bool light_changed = false;
            light_changed |= SliderFloat("Ambient intensity", &Light.ColorAndAmbient[3], 0, 1);
            light_changed |= ColorEdit3("Diffuse color", &Light.ColorAndAmbient[0]);
            light_changed |= SliderFloat3("Direction", &Light.Direction.x, -1, 1);
            if (light_changed) {
                VC.CreateOrUpdateBuffer(LightBuffer, &Light);
                SubmitCommandBuffer();
            }
            EndTabItem();
        }
        if (BeginTabItem("Object")) {
            if (Checkbox("Show grid", &ShowGrid)) {
                RecordCommandBuffer();
                SubmitCommandBuffer();
            }
            SeparatorText("Render");
            int render_mode = int(Mode);
            bool render_mode_changed = RadioButton("Faces and edges", &render_mode, int(RenderMode::FacesAndEdges));
            SameLine();
            render_mode_changed |= RadioButton("Faces", &render_mode, int(RenderMode::Faces));
            SameLine();
            render_mode_changed |= RadioButton("Edges", &render_mode, int(RenderMode::Edges));
            SameLine();
            render_mode_changed |= RadioButton("Smooth", &render_mode, int(RenderMode::Smooth));
            if (render_mode_changed) {
                Mode = RenderMode(render_mode);
                UpdateGeometryEdgeColors(); // Different modes use different edge colors for better visibility.
                RecordCommandBuffer(); // Changing mode can change the rendered shader pipeline(s).
                SubmitCommandBuffer();
            }
            if (Mode == RenderMode::FacesAndEdges || Mode == RenderMode::Edges) {
                auto &edge_color = Mode == RenderMode::FacesAndEdges ? MeshEdgeColor : EdgeColor;
                if (ColorEdit3("Edge color", &edge_color.x)) {
                    UpdateGeometryEdgeColors();
                    SubmitCommandBuffer();
                }
            }
            SeparatorText("Silhouette");
            if (ColorEdit4("Color", &SilhouetteDisplay.Color[0])) {
                VC.CreateOrUpdateBuffer(SilhouetteDisplayBuffer, &SilhouetteDisplay);
                SubmitCommandBuffer();
            }
            SeparatorText("Selection");
            int selection_mode = int(SelectionMode);
            bool selection_mode_changed = RadioButton("None", &selection_mode, int(SelectionMode::None));
            SameLine();
            selection_mode_changed |= RadioButton("Vertex", &selection_mode, int(SelectionMode::Vertex));
            SameLine();
            selection_mode_changed |= RadioButton("Edge", &selection_mode, int(SelectionMode::Edge));
            SameLine();
            selection_mode_changed |= RadioButton("Face", &selection_mode, int(SelectionMode::Face));
            if (selection_mode_changed) SelectionMode = ::SelectionMode(selection_mode);
            TextUnformatted(GeometryInstances[0]->GetHighlightLabel().c_str());
            SeparatorText("Transform");
            Gizmo->RenderDebug();
            EndTabItem();
        }
        if (BeginTabItem("Shader")) {
            if (Button("Recompile shaders")) {
                CompileShaders();
                RecordCommandBuffer();
                SubmitCommandBuffer();
            }
            EndTabItem();
        }
        EndTabBar();
    }
}
