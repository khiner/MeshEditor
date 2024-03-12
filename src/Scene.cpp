#include "Scene.h"

#include <format>
#include <ranges>
#include <vector>

#include <glm/gtx/matrix_decompose.hpp>

#include "imgui.h"

#include "ImGuizmo.h" // imgui must be included before imguizmo.

#include "numeric/mat3.h"

#include "Registry.h"
#include "mesh/primitive/Cuboid.h"
#include "vulkan/VulkanContext.h"

void Capitalize(std::string &str) {
    if (!str.empty() && str[0] >= 'a' && str[0] <= 'z') str[0] += 'A' - 'a';
}

const std::vector AllElements{MeshElement::Face, MeshElement::Vertex, MeshElement::Edge};
const std::vector AllElementsWithNone{MeshElement::None, MeshElement::Face, MeshElement::Vertex, MeshElement::Edge};
const std::vector AllNormalElements{MeshElement::Face, MeshElement::Vertex};

const vk::ClearColorValue Transparent{0, 0, 0, 0};

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

namespace ImageFormat {
const auto Color = vk::Format::eB8G8R8A8Unorm;
const auto Float = vk::Format::eR32G32B32A32Sfloat;
const auto Depth = vk::Format::eD32Sfloat;
} // namespace ImageFormat

struct Gizmo {
    ImGuizmo::OPERATION ActiveOp{ImGuizmo::TRANSLATE};
    bool ShowBounds{false};
    bool ShowModelGizmo{false};

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

        Checkbox("Gizmo", &ShowModelGizmo);
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
    }
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

void RenderPipeline::RenderBuffers(vk::CommandBuffer cb, const VkMeshBuffers &mesh_buffers, SPT spt, const VulkanBuffer &models_buffer) const {
    const auto &shader_pipeline = *ShaderPipelines.at(spt);
    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *shader_pipeline.Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shader_pipeline.PipelineLayout, 0, *shader_pipeline.DescriptorSet, {});
    mesh_buffers.Bind(cb);
    static const vk::DeviceSize models_buffer_offsets[] = {0};
    cb.bindVertexBuffers(1, *models_buffer.Buffer, models_buffer_offsets);
    const uint instance_count = models_buffer.Size / sizeof(mat4);
    mesh_buffers.Draw(cb, instance_count);
}

MainRenderPipeline::MainRenderPipeline(const VulkanContext &vc)
    : RenderPipeline(vc), MsaaSamples(GetMaxUsableSampleCount(VC.PhysicalDevice)) {
    const std::vector<vk::AttachmentDescription> attachments{
        // Depth attachment.
        {{}, ImageFormat::Depth, MsaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
        // Multisampled offscreen image.
        {{}, ImageFormat::Color, MsaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        // Single-sampled resolve.
        {{}, ImageFormat::Color, vk::SampleCountFlagBits::e1, {}, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
    const vk::AttachmentReference color_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::AttachmentReference resolve_attachment_ref{2, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, &resolve_attachment_ref, &depth_attachment_ref};
    RenderPass = VC.Device->createRenderPassUnique({{}, attachments, subpass});

    ShaderPipelines[SPT::Fill] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "Lighting.frag"}}},
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList, CreateColorBlendAttachment(true), CreateDepthStencil(), MsaaSamples
    );
    ShaderPipelines[SPT::Line] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "VertexColor.frag"}}},
        vk::PolygonMode::eLine, vk::PrimitiveTopology::eLineList, CreateColorBlendAttachment(true), CreateDepthStencil(), MsaaSamples
    );
    ShaderPipelines[SPT::Grid] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "GridLines.vert"}, {ShaderType::eFragment, "GridLines.frag"}}},
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip, CreateColorBlendAttachment(true), CreateDepthStencil(true, false), MsaaSamples
    );
    ShaderPipelines[SPT::Texture] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "SilhouetteEdgeTexture.frag"}}},
        // We render all the silhouette edge texture's pixels regardless of the tested depth value,
        // but also explicitly override the depth buffer to make edge pixels "stick" to the mesh they are derived from.
        // We should be able to just set depth testing to false and depth writing to true, but it seems that some GPUs or drivers
        // optimize out depth writes when depth testing is disabled, so instead we configure a depth test that always passes.
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip, CreateColorBlendAttachment(true), CreateDepthStencil(true, true, vk::CompareOp::eAlways), MsaaSamples
    );
    ShaderPipelines[SPT::DebugNormals] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "Normals.frag"}}},
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList, CreateColorBlendAttachment(true), CreateDepthStencil(), MsaaSamples
    );
}

void RenderPipeline::UpdateDescriptors(std::vector<ShaderBindingDescriptor> &&descriptors) const {
    std::vector<vk::WriteDescriptorSet> write_descriptor_sets;
    for (const auto &descriptor : descriptors) {
        const auto &sp = ShaderPipelines.at(descriptor.PipelineType);
        const auto *buffer_info = descriptor.BufferInfo.has_value() ? &(*descriptor.BufferInfo) : nullptr;
        const auto *image_info = descriptor.ImageInfo.has_value() ? &(*descriptor.ImageInfo) : nullptr;
        if (auto ds = sp->CreateWriteDescriptorSet(descriptor.BindingName, buffer_info, image_info)) {
            write_descriptor_sets.push_back(*ds);
        }
    }
    VC.Device->updateDescriptorSets(write_descriptor_sets, {});
}

void MainRenderPipeline::SetExtent(vk::Extent2D extent) {
    Extent = extent;
    const vk::Extent3D e3d{Extent, 1};
    DepthImage.Create(
        VC,
        {{}, vk::ImageType::e2D, ImageFormat::Depth, e3d, 1, 1, MsaaSamples, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, ImageFormat::Depth, {}, {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}}
    );
    OffscreenImage.Create(
        VC,
        {{}, vk::ImageType::e2D, ImageFormat::Color, e3d, 1, 1, MsaaSamples, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, ImageFormat::Color, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
    );
    ResolveImage.Create(
        VC,
        {{}, vk::ImageType::e2D, ImageFormat::Color, e3d, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, ImageFormat::Color, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
    );
    const std::array image_views{*DepthImage.View, *OffscreenImage.View, *ResolveImage.View};
    Framebuffer = VC.Device->createFramebufferUnique({{}, *RenderPass, image_views, Extent.width, Extent.height, 1});
}

void MainRenderPipeline::Begin(vk::CommandBuffer cb, const vk::ClearColorValue &background_color) const {
    const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {background_color}};
    cb.beginRenderPass({*RenderPass, *Framebuffer, vk::Rect2D{{0, 0}, Extent}, clear_values}, vk::SubpassContents::eInline);
}

SilhouetteRenderPipeline::SilhouetteRenderPipeline(const VulkanContext &vc) : RenderPipeline(vc) {
    const std::vector<vk::AttachmentDescription> attachments{
        // Single-sampled offscreen image.
        {{}, ImageFormat::Float, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference color_attachment_ref{0, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, nullptr};
    RenderPass = VC.Device->createRenderPassUnique({{}, attachments, subpass});

    ShaderPipelines[SPT::Silhouette] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "Depth.frag"}}},
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList, CreateColorBlendAttachment(false), std::nullopt, vk::SampleCountFlagBits::e1
    );
}

void SilhouetteRenderPipeline::SetExtent(vk::Extent2D extent) {
    Extent = extent;
    OffscreenImage.Create(
        VC,
        {{}, vk::ImageType::e2D, ImageFormat::Float, vk::Extent3D{Extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, ImageFormat::Float, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
    );

    const std::array image_views{*OffscreenImage.View};
    Framebuffer = VC.Device->createFramebufferUnique({{}, *RenderPass, image_views, Extent.width, Extent.height, 1});
}

void SilhouetteRenderPipeline::Begin(vk::CommandBuffer cb) const {
    static const std::vector<vk::ClearValue> clear_values{{Transparent}};
    cb.beginRenderPass({*RenderPass, *Framebuffer, vk::Rect2D{{0, 0}, Extent}, clear_values}, vk::SubpassContents::eInline);
}

EdgeDetectionRenderPipeline::EdgeDetectionRenderPipeline(const VulkanContext &vc) : RenderPipeline(vc) {
    const std::vector<vk::AttachmentDescription> attachments{
        // Single-sampled offscreen image.
        {{}, ImageFormat::Float, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference color_attachment_ref{0, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, nullptr};
    RenderPass = VC.Device->createRenderPassUnique({{}, attachments, subpass});

    ShaderPipelines[SPT::EdgeDetection] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "MeshEdges.frag"}}},
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip, CreateColorBlendAttachment(false), std::nullopt, vk::SampleCountFlagBits::e1
    );
}

void EdgeDetectionRenderPipeline::SetExtent(vk::Extent2D extent) {
    Extent = extent;
    OffscreenImage.Create(
        VC,
        {{}, vk::ImageType::e2D, ImageFormat::Float, vk::Extent3D{Extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, ImageFormat::Float, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
    );

    const std::array image_views{*OffscreenImage.View};
    Framebuffer = VC.Device->createFramebufferUnique({{}, *RenderPass, image_views, Extent.width, Extent.height, 1});
}

void EdgeDetectionRenderPipeline::Begin(vk::CommandBuffer cb) const {
    static const std::vector<vk::ClearValue> clear_values{{Transparent}};
    cb.beginRenderPass({*RenderPass, *Framebuffer, vk::Rect2D{{0, 0}, Extent}, clear_values}, vk::SubpassContents::eInline);
}

Scene::Scene(const VulkanContext &vc, Registry &r)
    : VC(vc), R(r), MainRenderPipeline(VC), SilhouetteRenderPipeline(VC), EdgeDetectionRenderPipeline(VC) {
    R.AddMesh(Cuboid{{0.5, 0.5, 0.5}});
    CreateOrUpdateBuffers(0);
    UpdateEdgeColors();
    UpdateTransform();
    VC.CreateOrUpdateBuffer(LightsBuffer, &Lights);
    VC.CreateOrUpdateBuffer(SilhouetteDisplayBuffer, &SilhouetteDisplay);
    vk::DescriptorBufferInfo transform_buffer{*TransformBuffer.Buffer, 0, VK_WHOLE_SIZE};
    MainRenderPipeline.UpdateDescriptors({
        {SPT::Fill, "TransformUBO", transform_buffer},
        {SPT::Fill, "LightsUBO", vk::DescriptorBufferInfo{*LightsBuffer.Buffer, 0, VK_WHOLE_SIZE}},
        {SPT::Line, "TransformUBO", transform_buffer},
        {SPT::Grid, "ViewProjNearFarUBO", vk::DescriptorBufferInfo{*ViewProjNearFarBuffer.Buffer, 0, VK_WHOLE_SIZE}},
        {SPT::Texture, "SilhouetteDisplayUBO", vk::DescriptorBufferInfo{*SilhouetteDisplayBuffer.Buffer, 0, VK_WHOLE_SIZE}},
        {SPT::DebugNormals, "TransformUBO", transform_buffer},
    });
    SilhouetteRenderPipeline.UpdateDescriptors({
        {SPT::Silhouette, "TransformUBO", transform_buffer},
    });

    Gizmo = std::make_unique<::Gizmo>();
    CompileShaders();
}

Scene::~Scene(){}; // Using unique handles, so no need to manually destroy anything.

Mesh &Scene::GetSelectedMesh() const { return R.Meshes[SelectedObjectId]; }
mat4 &Scene::GetSelectedModel() const { return R.Models[SelectedObjectId]; }

void Scene::CreateOrUpdateBuffers(uint instance, MeshElementIndex highlighted_element) {
    auto &mesh = R.Meshes[instance];
    auto &buffers = R.ElementBuffers[instance];
    for (const auto element : AllElements) {
        mesh.UpdateNormals(); // todo only update when necessary.
        buffers[element].Set(VC, mesh.GenerateBuffers(element, highlighted_element));
    }
}

void Scene::SetExtent(vk::Extent2D extent) {
    Extent = extent;
    UpdateTransform(); // Transform depends on the aspect ratio.
    MainRenderPipeline.SetExtent(extent);
    SilhouetteRenderPipeline.SetExtent(extent);
    SilhouetteFillImageSampler = VC.Device->createSamplerUnique({
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

    EdgeDetectionRenderPipeline.UpdateDescriptors({
        {SPT::EdgeDetection, "Tex", std::nullopt, vk::DescriptorImageInfo{*SilhouetteFillImageSampler, *SilhouetteRenderPipeline.OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal}},
    });
    EdgeDetectionRenderPipeline.SetExtent(extent);
    SilhouetteEdgeImageSampler = VC.Device->createSamplerUnique({{}, vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest});
    MainRenderPipeline.UpdateDescriptors({
        {SPT::Texture, "SilhouetteEdgeTexture", std::nullopt, vk::DescriptorImageInfo{*SilhouetteEdgeImageSampler, *EdgeDetectionRenderPipeline.OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal}},
    });
}

void Scene::RecordCommandBuffer() {
    const auto cb = *VC.CommandBuffers[0];
    cb.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(Extent.width), float(Extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, Extent});

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
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, // No dependency flags.
        {}, // No memory barriers.
        {}, // No buffer memory barriers.
        image_memory_barriers
    );

    const auto &buffers = R.ElementBuffers[SelectedObjectId];

    SilhouetteRenderPipeline.Begin(cb);

    SilhouetteRenderPipeline.RenderBuffers(cb, buffers.at(MeshElement::Vertex), SPT::Silhouette, ModelsBuffer);
    cb.endRenderPass();

    EdgeDetectionRenderPipeline.Begin(cb);
    EdgeDetectionRenderPipeline.GetShaderPipeline(SPT::EdgeDetection)->RenderQuad(cb);
    cb.endRenderPass();

    MainRenderPipeline.Begin(cb, BackgroundColor);

    const SPT fill_pipeline = ColorMode == ColorMode::Mesh ? SPT::Fill : SPT::DebugNormals;
    if (RenderMode == RenderMode::Faces) {
        MainRenderPipeline.RenderBuffers(cb, buffers.at(MeshElement::Face), fill_pipeline, ModelsBuffer);
    } else if (RenderMode == RenderMode::Edges) {
        MainRenderPipeline.RenderBuffers(cb, buffers.at(MeshElement::Edge), SPT::Line, ModelsBuffer);
    } else if (RenderMode == RenderMode::FacesAndEdges) {
        MainRenderPipeline.RenderBuffers(cb, buffers.at(MeshElement::Face), fill_pipeline, ModelsBuffer);
        MainRenderPipeline.RenderBuffers(cb, buffers.at(MeshElement::Edge), SPT::Line, ModelsBuffer);
    } else if (RenderMode == RenderMode::Vertices) {
        MainRenderPipeline.RenderBuffers(cb, buffers.at(MeshElement::Vertex), fill_pipeline, ModelsBuffer);
    }
    if (SelectedObjectId < R.NormalIndicatorBuffers.size()) {
        const auto &normal_buffers = R.NormalIndicatorBuffers[SelectedObjectId];
        if (auto it = normal_buffers.find(MeshElement::Face); it != normal_buffers.end()) {
            MainRenderPipeline.RenderBuffers(cb, it->second, SPT::Line, ModelsBuffer);
        }
        if (auto it = normal_buffers.find(MeshElement::Vertex); it != normal_buffers.end()) {
            MainRenderPipeline.RenderBuffers(cb, it->second, SPT::Line, ModelsBuffer);
        }
    }
    MainRenderPipeline.GetShaderPipeline(SPT::Texture)->RenderQuad(cb);
    if (ShowGrid) MainRenderPipeline.GetShaderPipeline(SPT::Grid)->RenderQuad(cb);

    cb.endRenderPass();

    cb.end();
}

void Scene::SubmitCommandBuffer(vk::Fence fence) const {
    vk::SubmitInfo submit;
    submit.setCommandBuffers(*VC.CommandBuffers[0]);
    VC.Queue.submit(submit, fence);
}

void Scene::RecordAndSubmitCommandBuffer(vk::Fence fence) {
    RecordCommandBuffer();
    SubmitCommandBuffer(fence);
}

void Scene::CompileShaders() {
    MainRenderPipeline.CompileShaders();
    SilhouetteRenderPipeline.CompileShaders();
    EdgeDetectionRenderPipeline.CompileShaders();
}

void Scene::UpdateEdgeColors() {
    Mesh::EdgeColor = RenderMode == RenderMode::FacesAndEdges ? MeshEdgeColor : EdgeColor;
    for (uint i = 0; i < R.Meshes.size(); ++i) CreateOrUpdateBuffers(i, HighlightedElement);
}

void Scene::UpdateNormalIndicators() {
    auto &normals = R.NormalIndicatorBuffers;
    if (!ShownNormals.empty() && normals.empty()) normals.emplace_back();
    else if (ShownNormals.empty() && !normals.empty()) normals.pop_back();
    if (ShownNormals.empty()) return;

    const auto &mesh = GetSelectedMesh();
    auto &selected_normals = normals[SelectedObjectId];
    for (const auto element : AllNormalElements) {
        if (ShownNormals.contains(element)) selected_normals.emplace(element, VkMeshBuffers{VC, mesh.GenerateNormalBuffers(element)});
        else selected_normals.erase(element);
    }
}

void Scene::UpdateTransform() {
    const float aspect_ratio = Extent.width == 0 || Extent.height == 0 ? 1.f : float(Extent.width) / float(Extent.height);
    ModelsBuffer.Size = R.Models.size() * sizeof(mat4);
    VC.CreateOrUpdateBuffer(ModelsBuffer, R.Models.data());

    // todo update for instancing, only recalculate when model changes.
    const mat4 &model = GetSelectedModel();
    const mat3 normal_to_world = glm::transpose(glm::inverse(mat3(model)));
    const Transform transform{Camera.GetViewMatrix(), Camera.GetProjectionMatrix(aspect_ratio), normal_to_world};
    VC.CreateOrUpdateBuffer(TransformBuffer, &transform);
    const ViewProjNearFar vpnf{transform.View, transform.Projection, Camera.NearClip, Camera.FarClip};
    VC.CreateOrUpdateBuffer(ViewProjNearFarBuffer, &vpnf);
}

using namespace ImGui;

vec2 ToGlm(ImVec2 v) { return {v.x, v.y}; }
vk::ClearColorValue ToClearColor(vec4 v) { return {v.r, v.g, v.b, v.a}; }

// Returns a world space ray from the mouse into the scene.
Ray GetMouseWorldRay(Camera camera, vec2 view_extent) {
    const vec2 mouse_pos = ToGlm(GetMousePos() - GetCursorScreenPos()), content_region = ToGlm(GetContentRegionAvail());
    const vec2 mouse_content_pos = mouse_pos / content_region;
    // Normalized Device Coordinates, $\mathcal{NDC} \in [-1,1]^2$
    const vec2 mouse_pos_ndc = vec2{2 * mouse_content_pos.x - 1, 1 - 2 * mouse_content_pos.y};
    return camera.ClipPosToWorldRay(mouse_pos_ndc, view_extent.x / view_extent.y);
}

vec2 ToGlm(vk::Extent2D e) { return {float(e.width), float(e.height)}; }
vk::Extent2D ToVkExtent(vec2 e) { return {uint(e.x), uint(e.y)}; }

// Mesh::FH Object::FindFirstIntersectingFace(const Ray &world_ray, vec3 *closest_intersect_point_out) const { return GetMesh().FindFirstIntersectingFace(world_ray.WorldToLocal(GetModel()), closest_intersect_point_out); }
// Mesh::VH Object::FindNearestVertex(const Ray &world_ray) const { return GetMesh().FindNearestVertex(world_ray.WorldToLocal(GetModel())); }
// Mesh::EH Object::FindNearestEdge(const Ray &world_ray) const { return GetMesh().FindNearestEdge(world_ray.WorldToLocal(GetModel())); }

bool Scene::Render() {
    // Handle mouse input.
    if (SelectionElement != MeshElement::None) {
        auto &mesh = GetSelectedMesh();
        auto &model = GetSelectedModel();
        const Ray mouse_ray = GetMouseWorldRay(Camera, ToGlm(Extent)).WorldToLocal(model);
        const Mesh::ElementIndex highlighted_element = HighlightedElement;
        if (SelectionElement == MeshElement::Face) {
            const auto fh = mesh.FindFirstIntersectingFace(mouse_ray);
            if (highlighted_element != fh) HighlightedElement = {SelectionElement, fh.idx()};
        } else if (SelectionElement == MeshElement::Vertex) {
            const auto vh = mesh.FindNearestVertex(mouse_ray);
            if (highlighted_element != vh) HighlightedElement = {SelectionElement, vh.idx()};
        } else if (SelectionElement == MeshElement::Edge) {
            const auto eh = mesh.FindNearestEdge(mouse_ray);
            if (highlighted_element != eh) HighlightedElement = {SelectionElement, eh.idx()};
        }
        if (HighlightedElement != highlighted_element) {
            CreateOrUpdateBuffers(SelectedObjectId, HighlightedElement);
            SubmitCommandBuffer();
        }
    }

    const vec2 content_region = ToGlm(GetContentRegionAvail());
    const auto bg_color = ToClearColor(BgColor);
    const bool extent_changed = Extent.width != content_region.x || Extent.height != content_region.y;
    const bool bg_color_changed = BackgroundColor.float32 != bg_color.float32;
    if (!extent_changed && !bg_color_changed) return false;

    BackgroundColor = bg_color;

    if (extent_changed) SetExtent(ToVkExtent(content_region));
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
    // Handle mouse wheel zoom.
    const float mouse_wheel = GetIO().MouseWheel;
    if (mouse_wheel != 0 && IsWindowHovered()) Camera.SetTargetDistance(Camera.GetDistance() * (1.f - mouse_wheel / 16.f));

    Gizmo->Begin();
    const float aspect_ratio = float(Extent.width) / float(Extent.height);
    auto &model = GetSelectedModel();
    const bool model_changed = Gizmo->Render(Camera, model, aspect_ratio);
    const bool view_changed = Camera.Tick();
    if (model_changed || view_changed) {
        UpdateTransform();
        SubmitCommandBuffer();
    }
}

void DecomposeTransform(const glm::mat4 &transform, glm::vec3 &position, glm::vec3 &rotation, glm::vec3 &scale) {
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::quat orientation;
    glm::decompose(transform, scale, orientation, position, skew, perspective);
    rotation = glm::eulerAngles(orientation) * 180.f / glm::pi<float>(); // Convert radians to degrees
}

void Scene::RenderControls() {
    if (BeginTabBar("Scene controls")) {
        if (BeginTabItem("Scene")) {
            if (Checkbox("Show grid", &ShowGrid)) RecordAndSubmitCommandBuffer();
            if (Button("Recompile shaders")) {
                CompileShaders();
                RecordAndSubmitCommandBuffer();
            }
            EndTabItem();
        }
        if (BeginTabItem("Camera")) {
            bool camera_changed = false;
            if (Button("Reset camera")) {
                Camera = CreateDefaultCamera();
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
        if (BeginTabItem("Lights")) {
            bool light_changed = false;
            SeparatorText("View light");
            light_changed |= ColorEdit3("Color", &Lights.ViewColorAndAmbient[0]);
            SeparatorText("Ambient light");
            light_changed |= SliderFloat("Intensity", &Lights.ViewColorAndAmbient[3], 0, 1);
            SeparatorText("Directional light");
            light_changed |= SliderFloat3("Direction", &Lights.Direction[0], -1, 1);
            light_changed |= ColorEdit3("Color", &Lights.DirectionalColorAndIntensity[0]);
            light_changed |= SliderFloat("Intensity", &Lights.DirectionalColorAndIntensity[3], 0, 1);
            if (light_changed) {
                VC.CreateOrUpdateBuffer(LightsBuffer, &Lights);
                SubmitCommandBuffer();
            }
            EndTabItem();
        }
        if (BeginTabItem("Object")) {
            if (CollapsingHeader("Render")) {
                SeparatorText("Render mode");
                int render_mode = int(RenderMode);
                bool render_mode_changed = RadioButton("Faces and edges##Render", &render_mode, int(RenderMode::FacesAndEdges));
                SameLine();
                render_mode_changed |= RadioButton("Faces##Render", &render_mode, int(RenderMode::Faces));
                SameLine();
                render_mode_changed |= RadioButton("Edges##Render", &render_mode, int(RenderMode::Edges));
                SameLine();
                render_mode_changed |= RadioButton("Vertices##Render", &render_mode, int(RenderMode::Vertices));

                int color_mode = int(ColorMode);
                bool color_mode_changed = false;
                if (RenderMode != RenderMode::Edges) {
                    SeparatorText("Fill color mode");
                    color_mode_changed |= RadioButton("Mesh##Color", &color_mode, int(ColorMode::Mesh));
                    color_mode_changed |= RadioButton("Normals##Color", &color_mode, int(ColorMode::Normals));
                }
                if (render_mode_changed || color_mode_changed) {
                    RenderMode = ::RenderMode(render_mode);
                    ColorMode = ::ColorMode(color_mode);
                    UpdateEdgeColors(); // Different modes use different edge colors for better visibility.
                    RecordAndSubmitCommandBuffer(); // Changing mode can change the rendered shader pipeline(s).
                }
                if (RenderMode == RenderMode::FacesAndEdges || RenderMode == RenderMode::Edges) {
                    auto &edge_color = RenderMode == RenderMode::FacesAndEdges ? MeshEdgeColor : EdgeColor;
                    if (ColorEdit3("Edge color", &edge_color.x)) {
                        UpdateEdgeColors();
                        SubmitCommandBuffer();
                    }
                }

                SeparatorText("Normal indicators");

                const uint before_normal_count = ShownNormals.size();
                for (const auto element : AllNormalElements) {
                    bool show_normals = ShownNormals.contains(element);
                    std::string name = to_string(element);
                    Capitalize(name);
                    if (Checkbox(name.c_str(), &show_normals)) {
                        if (show_normals) ShownNormals.insert(element);
                        else ShownNormals.erase(element);
                    }
                    if (element != AllNormalElements.back()) SameLine();
                }
                if (before_normal_count != ShownNormals.size()) {
                    UpdateNormalIndicators();
                    RecordAndSubmitCommandBuffer();
                }

                SeparatorText("Silhouette");
                if (ColorEdit4("Color", &SilhouetteDisplay.Color[0])) {
                    VC.CreateOrUpdateBuffer(SilhouetteDisplayBuffer, &SilhouetteDisplay);
                    SubmitCommandBuffer();
                }
            }
            if (CollapsingHeader("Selection")) {
                int selection_mode = int(SelectionElement);
                for (const auto element : AllElementsWithNone) {
                    std::string name = to_string(element);
                    Capitalize(name);
                    name += "##Selection";
                    if (RadioButton(name.c_str(), &selection_mode, int(element))) SelectionElement = MeshElement(element);
                    if (element != AllElementsWithNone.back()) SameLine();
                }
                const std::string highlight_label = HighlightedElement.is_valid() ? std::format("Hovered {}: {}", to_string(HighlightedElement.Element), HighlightedElement.idx()) : "Hovered: None";
                TextUnformatted(highlight_label.c_str());
            }
            if (CollapsingHeader("Transform")) {
                auto &model = GetSelectedModel();
                glm::vec3 position, rotation, scale;
                DecomposeTransform(model, position, rotation, scale);
                bool transform_changed = false;
                transform_changed |= DragFloat3("Position", &position[0], 0.01f);
                transform_changed |= DragFloat3("Rotation (deg)", &rotation[0], 1, -90, 90, "%.0f");
                transform_changed |= DragFloat3("Scale", &scale[0], 0.01f, 0.01f, 10);
                if (transform_changed) {
                    model =
                        glm::translate(position) *
                        mat4{glm::quat{{glm::radians(rotation.x), glm::radians(rotation.y), glm::radians(rotation.z)}}} *
                        glm::scale(scale);
                    UpdateTransform();
                    SubmitCommandBuffer();
                }
                Gizmo->RenderDebug();
            }
            EndTabItem();
        }
        EndTabBar();
    }
}
