#include "Scene.h"

#include <format>
#include <ranges>

#include <glm/gtx/matrix_decompose.hpp>

#include "imgui.h"

#include "ImGuizmo.h" // imgui must be included before imguizmo.

#include "numeric/mat3.h"

#include "mesh/MeshBuffers.h"
#include "mesh/primitive/Cuboid.h"
#include "mesh/primitive/IcoSphere.h"
#include "vulkan/VulkanContext.h"

#include <print>

void Capitalize(std::string &str) {
    if (!str.empty() && str[0] >= 'a' && str[0] <= 'z') str[0] += 'A' - 'a';
}

const std::vector AllNormalElements{MeshElement::Face, MeshElement::Vertex};

using BuffersByElement = std::unordered_map<MeshElement, MeshBuffers>;
struct MeshVkData {
    std::unordered_map<entt::entity, BuffersByElement> Main, NormalIndicators;
};

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

    void Render(Camera &camera, mat4 &model, float aspect_ratio, bool &view_changed, bool &model_changed) const {
        using namespace ImGui;

        static const float ViewManipulateSize = 128;

        const auto &window_pos = GetWindowPos();
        const auto view_manipulate_pos = window_pos + ImVec2{GetWindowContentRegionMax().x - ViewManipulateSize, GetWindowContentRegionMin().y};
        auto camera_view = camera.GetViewMatrix();
        const float camera_distance = camera.GetDistance();
        view_changed = ImGuizmo::ViewManipulate(&camera_view[0][0], camera_distance, view_manipulate_pos, {ViewManipulateSize, ViewManipulateSize}, 0);
        if (view_changed) camera.SetPositionFromView(camera_view);

        auto camera_projection = camera.GetProjectionMatrix(aspect_ratio);
        model_changed = ShowModelGizmo && ImGuizmo::Manipulate(&camera_view[0][0], &camera_projection[0][0], ActiveOp, ImGuizmo::LOCAL, &model[0][0]);
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

void RenderPipeline::RenderBuffers(vk::CommandBuffer cb, const MeshBuffers &mesh_buffers, SPT spt, const VulkanBuffer &models_buffer) const {
    const auto &shader_pipeline = *ShaderPipelines.at(spt);
    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *shader_pipeline.Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shader_pipeline.PipelineLayout, 0, *shader_pipeline.DescriptorSet, {});

    // Bind buffers
    static const vk::DeviceSize vertex_buffer_offsets[] = {0}, models_buffer_offsets[] = {0};
    cb.bindVertexBuffers(0, *mesh_buffers.Vertices.Buffer, vertex_buffer_offsets);
    cb.bindIndexBuffer(*mesh_buffers.Indices.Buffer, 0, vk::IndexType::eUint32);
    cb.bindVertexBuffers(1, *models_buffer.Buffer, models_buffer_offsets);

    // Draw
    const uint index_count = mesh_buffers.Indices.Size / sizeof(uint);
    const uint instance_count = models_buffer.Size / sizeof(Model);
    cb.drawIndexed(index_count, instance_count, 0, 0, 0);
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

Scene::Scene(const VulkanContext &vc, entt::registry &r)
    : VC(vc), R(r), MeshVkData(std::make_unique<::MeshVkData>()), MainRenderPipeline(VC),
      SilhouetteRenderPipeline(VC), EdgeDetectionRenderPipeline(VC) {
    UpdateEdgeColors();
    VC.CreateBuffer(ModelsBuffer, 10 * sizeof(Model));
    VC.CreateBuffer(TransformBuffer, sizeof(ViewProj));
    VC.CreateBuffer(ViewProjNearFarBuffer, sizeof(ViewProjNearFar));
    UpdateViewProj();
    AddMesh(Cuboid({0.5, 0.5, 0.5}));

    VC.CreateBuffer(LightsBuffer, std::vector{Lights});
    VC.CreateBuffer(SilhouetteDisplayBuffer, std::vector{SilhouetteDisplay});
    vk::DescriptorBufferInfo transform_buffer{*TransformBuffer.Buffer, 0, VK_WHOLE_SIZE};
    MainRenderPipeline.UpdateDescriptors({
        {SPT::Fill, "ViewProjectionUBO", transform_buffer},
        {SPT::Fill, "LightsUBO", vk::DescriptorBufferInfo{*LightsBuffer.Buffer, 0, VK_WHOLE_SIZE}},
        {SPT::Line, "ViewProjectionUBO", transform_buffer},
        {SPT::Grid, "ViewProjNearFarUBO", vk::DescriptorBufferInfo{*ViewProjNearFarBuffer.Buffer, 0, VK_WHOLE_SIZE}},
        {SPT::Texture, "SilhouetteDisplayUBO", vk::DescriptorBufferInfo{*SilhouetteDisplayBuffer.Buffer, 0, VK_WHOLE_SIZE}},
        {SPT::DebugNormals, "ViewProjectionUBO", transform_buffer},
    });
    SilhouetteRenderPipeline.UpdateDescriptors({
        {SPT::Silhouette, "ViewProjectionUBO", transform_buffer},
    });

    Gizmo = std::make_unique<::Gizmo>();
    CompileShaders();
}

Scene::~Scene(){}; // Using unique handles, so no need to manually destroy anything.

void Scene::AddMesh(Mesh &&mesh) {
    const auto entity = R.create();
    MeshBufferIndices.emplace(entity, MeshBufferIndices.size());

    BuffersByElement mesh_buffers{};
    for (auto element : AllElements) { // todo only create buffers for viewed elements.
        auto &buffers = mesh_buffers[element];
        VC.CreateBuffer(buffers.Vertices, mesh.CreateVertices(element));
        VC.CreateBuffer(buffers.Indices, mesh.CreateIndices(element));
    }

    MeshVkData->Main.emplace(entity, std::move(mesh_buffers));
    MeshVkData->NormalIndicators.emplace(entity, BuffersByElement{});

    R.emplace<Mesh>(entity, std::move(mesh));
    R.emplace<Model>(entity, 1);

    // VC.CreateBuffer(ModelsBuffer, sizeof(Model) * MeshVkData->Main.size());
    // R.view<Mesh>().each([this](auto entity, auto &) { UpdateTransform(entity); });
    UpdateTransform(entity);
    SelectedEntity = entity;
}

Mesh &Scene::GetSelectedMesh() const { return R.get<Mesh>(SelectedEntity); }
Model &Scene::GetSelectedModel() const { return R.get<Model>(SelectedEntity); }
void Scene::SetSelectedModel(mat4 &&model) {
    R.replace<Model>(SelectedEntity, std::move(model));
    UpdateTransform(SelectedEntity);
}

void Scene::UpdateMeshBuffers(entt::entity entity, MeshElementIndex highlight_element) {
    auto &mesh = R.get<Mesh>(entity);
    auto &mesh_buffers = MeshVkData->Main.at(entity);
    for (auto element : AllElements) { // todo only update buffers for viewed elements.
        VC.UpdateBuffer(mesh_buffers[element].Vertices, mesh.CreateVertices(element, highlight_element));
    }
}

void Scene::SetExtent(vk::Extent2D extent) {
    Extent = extent;
    UpdateViewProj(); // Depends on the aspect ratio.
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

std::vector<std::pair<SPT, MeshElement>> GetPipelineElements(RenderMode render_mode, ColorMode color_mode) {
    const SPT fill_pipeline = color_mode == ColorMode::Mesh ? SPT::Fill : SPT::DebugNormals;
    switch (render_mode) {
        case RenderMode::Faces: return {{fill_pipeline, MeshElement::Face}};
        case RenderMode::Edges: return {{SPT::Line, MeshElement::Edge}};
        case RenderMode::FacesAndEdges: return {{fill_pipeline, MeshElement::Face}, {SPT::Line, MeshElement::Edge}};
        case RenderMode::Vertices: return {{fill_pipeline, MeshElement::Vertex}};
        case RenderMode::None: return {};
    }
}

void Scene::RecordCommandBuffer() {
    const auto cb = *VC.CommandBuffers[0];
    cb.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(Extent.width), float(Extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, Extent});

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, // No dependency flags.
        {}, // No memory barriers.
        {}, // No buffer memory barriers.
        std::vector<vk::ImageMemoryBarrier>{{
            {},
            {},
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            *MainRenderPipeline.ResolveImage,
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }}
    );

    const auto &buffers = MeshVkData->Main.at(SelectedEntity);

    SilhouetteRenderPipeline.Begin(cb);
    SilhouetteRenderPipeline.RenderBuffers(cb, buffers.at(MeshElement::Vertex), SPT::Silhouette, ModelsBuffer);
    cb.endRenderPass();

    EdgeDetectionRenderPipeline.Begin(cb);
    EdgeDetectionRenderPipeline.GetShaderPipeline(SPT::EdgeDetection)->RenderQuad(cb);
    cb.endRenderPass();

    MainRenderPipeline.Begin(cb, BackgroundColor);
    for (const auto [pipeline, element] : GetPipelineElements(RenderMode, ColorMode)) {
        MainRenderPipeline.RenderBuffers(cb, buffers.at(element), pipeline, ModelsBuffer);
    }
    MainRenderPipeline.GetShaderPipeline(SPT::Texture)->RenderQuad(cb);
    const auto &normals = MeshVkData->NormalIndicators.at(SelectedEntity);
    for (auto normal_element : AllNormalElements) {
        if (auto it = normals.find(normal_element); it != normals.end()) {
            MainRenderPipeline.RenderBuffers(cb, it->second, SPT::Line, ModelsBuffer);
        }
    }
    // R.view<Mesh>().each([this, &cb](auto entity, auto &) {});
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
    R.view<Mesh>().each([this](auto entity, auto &) { UpdateMeshBuffers(entity, HighlightedElement); });
}

void Scene::UpdateViewProj() {
    const float aspect_ratio = Extent.width == 0 || Extent.height == 0 ? 1.f : float(Extent.width) / float(Extent.height);
    const ViewProj view_proj{Camera.GetViewMatrix(), Camera.GetProjectionMatrix(aspect_ratio)};
    VC.UpdateBuffer(TransformBuffer, &view_proj);
    const ViewProjNearFar vpnf{view_proj.View, view_proj.Projection, Camera.NearClip, Camera.FarClip};
    VC.UpdateBuffer(ViewProjNearFarBuffer, &vpnf);
}

void Scene::UpdateTransform(entt::entity entity) {
    const auto &model = R.get<Model>(entity);
    const uint i = MeshBufferIndices.at(entity);
    VC.UpdateBuffer(ModelsBuffer, &model, i * sizeof(Model), sizeof(Model));
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

bool Scene::Render() {
    // Handle mouse input.
    if (SelectionElement != MeshElement::None) {
        auto &mesh = GetSelectedMesh();
        const auto &model = GetSelectedModel();
        const Ray mouse_ray = GetMouseWorldRay(Camera, ToGlm(Extent)).WorldToLocal(model.Transform);
        const Mesh::ElementIndex highlight_element = HighlightedElement;
        if (SelectionElement == MeshElement::Face) {
            const auto fh = mesh.FindFirstIntersectingFace(mouse_ray);
            if (highlight_element != fh) HighlightedElement = {SelectionElement, fh.idx()};
        } else if (SelectionElement == MeshElement::Vertex) {
            const auto vh = mesh.FindNearestVertex(mouse_ray);
            if (highlight_element != vh) HighlightedElement = {SelectionElement, vh.idx()};
        } else if (SelectionElement == MeshElement::Edge) {
            const auto eh = mesh.FindNearestEdge(mouse_ray);
            if (highlight_element != eh) HighlightedElement = {SelectionElement, eh.idx()};
        }
        if (HighlightedElement != highlight_element) {
            UpdateMeshBuffers(SelectedEntity, HighlightedElement);
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
    mat4 model = GetSelectedModel().Transform;
    bool view_changed, model_changed;
    Gizmo->Render(Camera, model, aspect_ratio, view_changed, model_changed);
    view_changed |= Camera.Tick();
    if (model_changed || view_changed) {
        if (model_changed) SetSelectedModel(std::move(model));
        if (view_changed) UpdateViewProj();
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
                UpdateViewProj();
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
                VC.UpdateBuffer(LightsBuffer, &Lights);
                SubmitCommandBuffer();
            }
            EndTabItem();
        }
        if (BeginTabItem("Object")) {
            if (CollapsingHeader("Add")) {
                if (Button("Cuboid")) AddMesh(Cuboid({0.5, 0.5, 0.5}));
                if (Button("IcoSphere")) AddMesh(IcoSphere(0.5, 3));
            }
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
                {
                    SeparatorText("Normal indicators");
                    auto &normals = MeshVkData->NormalIndicators.at(SelectedEntity);
                    for (const auto element : AllNormalElements) {
                        bool has_normals = normals.contains(element);
                        std::string name = to_string(element);
                        Capitalize(name);
                        if (Checkbox(name.c_str(), &has_normals)) {
                            if (has_normals) {
                                const auto &mesh = GetSelectedMesh();
                                MeshBuffers buffers;
                                VC.CreateBuffer(buffers.Vertices, mesh.CreateNormalVertices(element));
                                VC.CreateBuffer(buffers.Indices, mesh.CreateNormalIndices(element));
                                normals.emplace(element, std::move(buffers));
                            } else {
                                normals.erase(element);
                            }
                            RecordAndSubmitCommandBuffer();
                        }
                        if (element != AllNormalElements.back()) SameLine();
                    }
                }
                SeparatorText("Silhouette");
                if (ColorEdit4("Color", &SilhouetteDisplay.Color[0])) {
                    VC.UpdateBuffer(SilhouetteDisplayBuffer, &SilhouetteDisplay);
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
                const auto &model = GetSelectedModel().Transform;
                glm::vec3 pos, rot, scale;
                DecomposeTransform(model, pos, rot, scale);
                bool model_changed = false;
                model_changed |= DragFloat3("Position", &pos[0], 0.01f);
                model_changed |= DragFloat3("Rotation (deg)", &rot[0], 1, -90, 90, "%.0f");
                model_changed |= DragFloat3("Scale", &scale[0], 0.01f, 0.01f, 10);
                if (model_changed) {
                    SetSelectedModel(glm::translate(pos) * glm::mat4{glm::quat{{glm::radians(rot.x), glm::radians(rot.y), glm::radians(rot.z)}}} * glm::scale(scale));
                    SubmitCommandBuffer();
                }
                Gizmo->RenderDebug();
            }
            EndTabItem();
        }
        EndTabBar();
    }
}
