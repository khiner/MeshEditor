#include "Scene.h"

#include <format>
#include <ranges>

#include <glm/gtx/matrix_decompose.hpp>

#include "imgui.h"

#include "ImGuizmo.h" // imgui must be included before imguizmo.

#include "numeric/mat3.h"

#include "mesh/Primitives.h"
#include "vulkan/VulkanContext.h"

struct SceneNode {
    entt::entity parent = entt::null;
    std::vector<entt::entity> children;
    // Maps entities to their index in the models buffer. Includes parent. Only present in parent nodes.
    // This allows for contiguous storage of models in the buffer, with erases but no inserts (only appends, which avoids shuffling memory regions).
    std::unordered_map<entt::entity, uint> model_indices;
};

// Simple wrapper around vertex and index buffers.
struct VkRenderBuffers {
    VulkanBuffer Vertices, Indices;

    VkRenderBuffers(const VulkanContext &vc, std::vector<Vertex3D> &&vertices, std::vector<uint> &&indices)
        : Vertices(vc.CreateBuffer(vk::BufferUsageFlagBits::eVertexBuffer, std::move(vertices))),
          Indices(vc.CreateBuffer(vk::BufferUsageFlagBits::eIndexBuffer, std::move(indices))) {}

    VkRenderBuffers(const VulkanContext &vc, RenderBuffers &&buffers)
        : Vertices(vc.CreateBuffer(vk::BufferUsageFlagBits::eVertexBuffer, std::move(buffers.Vertices))),
          Indices(vc.CreateBuffer(vk::BufferUsageFlagBits::eIndexBuffer, std::move(buffers.Indices))) {}

    template<size_t N>
    VkRenderBuffers(const VulkanContext &vc, std::vector<Vertex3D> &&vertices, const std::array<uint, N> &indices)
        : Vertices(vc.CreateBuffer(vk::BufferUsageFlagBits::eVertexBuffer, std::move(vertices))),
          Indices(vc.CreateBuffer(vk::BufferUsageFlagBits::eIndexBuffer, indices)) {}
};

using MeshBuffers = std::unordered_map<MeshElement, VkRenderBuffers>;
struct MeshVkData {
    std::unordered_map<entt::entity, MeshBuffers> Main, NormalIndicators;
    std::unordered_map<entt::entity, VulkanBuffer> Models;
    std::unordered_map<entt::entity, VkRenderBuffers> Boxes;
    std::unordered_map<entt::entity, VkRenderBuffers> BvhBoxes;
};

std::vector<Vertex3D> CreateBoxVertices(const BBox &box, const vec4 &color) {
    const auto &corners = box.Corners();
    std::vector<Vertex3D> vertices;
    vertices.reserve(corners.size());
    // Normals don't matter for wireframes.
    for (auto &corner : corners) vertices.emplace_back(corner, vec3{}, color);
    return vertices;
}

const std::vector AllNormalElements{MeshElement::Vertex, MeshElement::Face};

const vk::ClearColorValue Transparent{0, 0, 0, 0};

namespace Format {
const auto Vec3 = vk::Format::eR32G32B32Sfloat;
const auto Vec4 = vk::Format::eR32G32B32A32Sfloat;
} // namespace Format

namespace ImageFormat {
const auto Color = vk::Format::eB8G8R8A8Unorm;
const auto Float = vk::Format::eR32G32B32A32Sfloat;
const auto Depth = vk::Format::eD32Sfloat;
} // namespace ImageFormat

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

vk::PipelineVertexInputStateCreateInfo CreateVertexInputState() {
    static const std::vector<vk::VertexInputBindingDescription> bindings{
        {0, sizeof(Vertex3D), vk::VertexInputRate::eVertex},
        {1, 2 * sizeof(mat4), vk::VertexInputRate::eInstance},
    };
    static const std::vector<vk::VertexInputAttributeDescription> attrs{
        {0, 0, Format::Vec3, offsetof(Vertex3D, Position)},
        {1, 0, Format::Vec3, offsetof(Vertex3D, Normal)},
        {2, 0, Format::Vec4, offsetof(Vertex3D, Color)},
        // Model mat4, one vec4 per row
        {3, 1, Format::Vec4, 0},
        {4, 1, Format::Vec4, sizeof(vec4)},
        {5, 1, Format::Vec4, 2 * sizeof(vec4)},
        {6, 1, Format::Vec4, 3 * sizeof(vec4)},
        // Inverse model mat4, one vec4 per row
        {7, 1, Format::Vec4, 4 * sizeof(vec4)},
        {8, 1, Format::Vec4, 5 * sizeof(vec4)},
        {9, 1, Format::Vec4, 6 * sizeof(vec4)},
        {10, 1, Format::Vec4, 7 * sizeof(vec4)},
    };
    return {{}, bindings, attrs};
}

void Capitalize(std::string &str) {
    if (!str.empty() && str[0] >= 'a' && str[0] <= 'z') str[0] += 'A' - 'a';
}

struct Gizmo {
    ImGuizmo::OPERATION ActiveOp{ImGuizmo::TRANSLATE};
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

    void Render(Camera &camera, bool &view_changed) const {
        using namespace ImGui;

        static const float ViewManipulateSize = 128;

        const auto &window_pos = GetWindowPos();
        const auto view_manipulate_pos = window_pos + ImVec2{GetWindowContentRegionMax().x - ViewManipulateSize, GetWindowContentRegionMin().y};
        auto camera_view = camera.GetViewMatrix();
        const float camera_distance = camera.GetDistance();
        view_changed = ImGuizmo::ViewManipulate(&camera_view[0][0], camera_distance, view_manipulate_pos, {ViewManipulateSize, ViewManipulateSize}, 0);
        if (view_changed) camera.SetPositionFromView(camera_view);
    }

    void Render(Camera &camera, mat4 &model, float aspect_ratio, bool &view_changed, bool &model_changed) const {
        using namespace ImGui;

        Render(camera, view_changed);
        auto camera_view = camera.GetViewMatrix();
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

void RenderPipeline::Render(vk::CommandBuffer cb, SPT spt, const VulkanBuffer &vertices, const VulkanBuffer &indices, const VulkanBuffer &models, std::optional<uint> model_index) const {
    const auto &shader_pipeline = *ShaderPipelines.at(spt);
    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *shader_pipeline.Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shader_pipeline.PipelineLayout, 0, *shader_pipeline.DescriptorSet, {});

    // Bind buffers
    static const vk::DeviceSize vertex_buffer_offsets[] = {0}, models_buffer_offsets[] = {0};
    cb.bindVertexBuffers(0, {*vertices.DeviceBuffer}, vertex_buffer_offsets);
    cb.bindIndexBuffer(*indices.DeviceBuffer, 0, vk::IndexType::eUint32);
    cb.bindVertexBuffers(1, {*models.DeviceBuffer}, models_buffer_offsets);

    // Draw
    const uint index_count = indices.Size / sizeof(uint);
    const uint first_instance = model_index.value_or(0);
    const uint instance_count = model_index.has_value() ? 1 : models.Size / sizeof(Model);
    cb.drawIndexed(index_count, instance_count, 0, 0, first_instance);
}

void RenderPipeline::Render(vk::CommandBuffer cb, SPT spt, const VkRenderBuffers &render_buffers, const VulkanBuffer &models, std::optional<uint> model_index) const {
    Render(cb, spt, render_buffers.Vertices, render_buffers.Indices, models, model_index);
}

MainPipeline::MainPipeline(const VulkanContext &vc)
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
        CreateVertexInputState(),
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList, CreateColorBlendAttachment(true), CreateDepthStencil(), MsaaSamples
    );
    ShaderPipelines[SPT::Line] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "VertexColor.frag"}}},
        CreateVertexInputState(),
        vk::PolygonMode::eLine, vk::PrimitiveTopology::eLineList, CreateColorBlendAttachment(true), CreateDepthStencil(), MsaaSamples
    );
    ShaderPipelines[SPT::Grid] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "GridLines.vert"}, {ShaderType::eFragment, "GridLines.frag"}}},
        vk::PipelineVertexInputStateCreateInfo{},
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip, CreateColorBlendAttachment(true), CreateDepthStencil(true, false), MsaaSamples
    );
    ShaderPipelines[SPT::Texture] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "SilhouetteEdgeTexture.frag"}}},
        vk::PipelineVertexInputStateCreateInfo{},
        // We render all the silhouette edge texture's pixels regardless of the tested depth value,
        // but also explicitly override the depth buffer to make edge pixels "stick" to the mesh they are derived from.
        // We should be able to just set depth testing to false and depth writing to true, but it seems that some GPUs or drivers
        // optimize out depth writes when depth testing is disabled, so instead we configure a depth test that always passes.
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip, CreateColorBlendAttachment(true), CreateDepthStencil(true, true, vk::CompareOp::eAlways), MsaaSamples
    );
    ShaderPipelines[SPT::DebugNormals] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "Normals.frag"}}},
        CreateVertexInputState(),
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

void MainPipeline::SetExtent(vk::Extent2D extent) {
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

void MainPipeline::Begin(vk::CommandBuffer cb, const vk::ClearColorValue &background_color) const {
    const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {background_color}};
    cb.beginRenderPass({*RenderPass, *Framebuffer, vk::Rect2D{{0, 0}, Extent}, clear_values}, vk::SubpassContents::eInline);
}

SilhouettePipeline::SilhouettePipeline(const VulkanContext &vc) : RenderPipeline(vc) {
    const std::vector<vk::AttachmentDescription> attachments{
        // Single-sampled offscreen image.
        {{}, ImageFormat::Float, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference color_attachment_ref{0, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, nullptr};
    RenderPass = VC.Device->createRenderPassUnique({{}, attachments, subpass});

    ShaderPipelines[SPT::Silhouette] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "Depth.frag"}}},
        CreateVertexInputState(),
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList, CreateColorBlendAttachment(false), std::nullopt, vk::SampleCountFlagBits::e1
    );
}

void SilhouettePipeline::SetExtent(vk::Extent2D extent) {
    Extent = extent;
    OffscreenImage.Create(
        VC,
        {{}, vk::ImageType::e2D, ImageFormat::Float, vk::Extent3D{Extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, ImageFormat::Float, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
    );

    const std::array image_views{*OffscreenImage.View};
    Framebuffer = VC.Device->createFramebufferUnique({{}, *RenderPass, image_views, Extent.width, Extent.height, 1});
}

void SilhouettePipeline::Begin(vk::CommandBuffer cb) const {
    static const std::vector<vk::ClearValue> clear_values{{Transparent}};
    cb.beginRenderPass({*RenderPass, *Framebuffer, vk::Rect2D{{0, 0}, Extent}, clear_values}, vk::SubpassContents::eInline);
}

EdgeDetectionPipeline::EdgeDetectionPipeline(const VulkanContext &vc) : RenderPipeline(vc) {
    const std::vector<vk::AttachmentDescription> attachments{
        // Single-sampled offscreen image.
        {{}, ImageFormat::Float, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference color_attachment_ref{0, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, nullptr};
    RenderPass = VC.Device->createRenderPassUnique({{}, attachments, subpass});

    ShaderPipelines[SPT::EdgeDetection] = std::make_unique<ShaderPipeline>(
        *VC.Device, *VC.DescriptorPool, Shaders{{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "MeshEdges.frag"}}},
        vk::PipelineVertexInputStateCreateInfo{},
        vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip, CreateColorBlendAttachment(false), std::nullopt, vk::SampleCountFlagBits::e1
    );
}

void EdgeDetectionPipeline::SetExtent(vk::Extent2D extent) {
    Extent = extent;
    OffscreenImage.Create(
        VC,
        {{}, vk::ImageType::e2D, ImageFormat::Float, vk::Extent3D{Extent, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, ImageFormat::Float, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
    );

    const std::array image_views{*OffscreenImage.View};
    Framebuffer = VC.Device->createFramebufferUnique({{}, *RenderPass, image_views, Extent.width, Extent.height, 1});
}

void EdgeDetectionPipeline::Begin(vk::CommandBuffer cb) const {
    static const std::vector<vk::ClearValue> clear_values{{Transparent}};
    cb.beginRenderPass({*RenderPass, *Framebuffer, vk::Rect2D{{0, 0}, Extent}, clear_values}, vk::SubpassContents::eInline);
}

Scene::Scene(const VulkanContext &vc, entt::registry &r)
    : VC(vc), R(r), MeshVkData(std::make_unique<::MeshVkData>()), MainPipeline(VC),
      SilhouettePipeline(VC), EdgeDetectionPipeline(VC) {
    UpdateEdgeColors();
    TransformBuffer = std::make_unique<VulkanBuffer>(VC.CreateBuffer(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(ViewProj)));
    ViewProjNearFarBuffer = std::make_unique<VulkanBuffer>(VC.CreateBuffer(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(ViewProjNearFar)));
    UpdateTransformBuffers();

    LightsBuffer = std::make_unique<VulkanBuffer>(VC.CreateBuffer(vk::BufferUsageFlagBits::eUniformBuffer, std::vector{Lights}));
    SilhouetteDisplayBuffer = std::make_unique<VulkanBuffer>(VC.CreateBuffer(vk::BufferUsageFlagBits::eUniformBuffer, std::vector{SilhouetteDisplay}));
    vk::DescriptorBufferInfo transform_buffer{*TransformBuffer->DeviceBuffer, 0, VK_WHOLE_SIZE};
    MainPipeline.UpdateDescriptors({
        {SPT::Fill, "ViewProjectionUBO", transform_buffer},
        {SPT::Fill, "LightsUBO", vk::DescriptorBufferInfo{*LightsBuffer->DeviceBuffer, 0, VK_WHOLE_SIZE}},
        {SPT::Line, "ViewProjectionUBO", transform_buffer},
        {SPT::Grid, "ViewProjNearFarUBO", vk::DescriptorBufferInfo{*ViewProjNearFarBuffer->DeviceBuffer, 0, VK_WHOLE_SIZE}},
        {SPT::Texture, "SilhouetteDisplayUBO", vk::DescriptorBufferInfo{*SilhouetteDisplayBuffer->DeviceBuffer, 0, VK_WHOLE_SIZE}},
        {SPT::DebugNormals, "ViewProjectionUBO", transform_buffer},
    });
    SilhouettePipeline.UpdateDescriptors({
        {SPT::Silhouette, "ViewProjectionUBO", transform_buffer},
    });

    Gizmo = std::make_unique<::Gizmo>();
    CompileShaders();

    AddPrimitive(Primitive::Cube, {1}, false);
}

Scene::~Scene(){}; // Using unique handles, so no need to manually destroy anything.

// Get the model VK buffer index.
// Returns `std::nullopt` if the entity is not visible (and thus does not have a rendered model).
std::optional<uint> Scene::GetModelBufferIndex(entt::entity entity) {
    if (entity == entt::null) return std::nullopt;

    const auto &model_indices = R.get<SceneNode>(GetParentEntity(entity)).model_indices;
    if (const auto it = model_indices.find(entity); it != model_indices.end()) return it->second;
    return std::nullopt;
}

Mesh &Scene::GetMesh(entt::entity entity) const { return R.get<Mesh>(GetParentEntity(entity)); }

void Scene::SetVisible(entt::entity entity, bool visible) {
    const bool already_visible = R.all_of<Visible>(entity);
    if ((visible && already_visible) || (!visible && !already_visible)) return;

    const auto parent = GetParentEntity(entity);
    auto &parent_node = R.get<SceneNode>(parent);
    auto &model_indices = parent_node.model_indices;
    if (visible) {
        R.emplace<Visible>(entity);
        // Insert model index as the max value + 1.
        const uint new_model_index = entity == parent || model_indices.empty() ?
            0 :
            std::max_element(model_indices.begin(), model_indices.end(), [](const auto &a, const auto &b) { return a.second < b.second; })->second + 1;
        for (auto &[_, model_index] : model_indices) {
            if (model_index >= new_model_index) ++model_index;
        }
        model_indices.emplace(entity, new_model_index);
        UpdateModelBuffer(entity);
    } else {
        R.remove<Visible>(entity);
        const uint old_model_index = *GetModelBufferIndex(entity);
        VC.EraseBufferRegion(MeshVkData->Models.at(GetParentEntity(entity)), old_model_index * sizeof(Model), sizeof(Model));
        model_indices.erase(entity);
        for (auto &[_, model_index] : model_indices) {
            if (model_index > old_model_index) --model_index;
        }
    }
}

entt::entity Scene::AddMesh(Mesh &&mesh, const mat4 &transform, bool submit, bool select, bool visible) {
    const auto entity = R.create();
    Model model{mat4(transform)};

    auto node = R.emplace<SceneNode>(entity); // No parent or children.
    R.emplace<Mesh>(entity, std::move(mesh));
    R.emplace<Model>(entity, model);

    MeshVkData->Models.emplace(entity, VC.CreateBuffer(vk::BufferUsageFlagBits::eVertexBuffer, sizeof(Model)));
    SetVisible(entity, true); // Always set visibility to true first, since this sets up the model buffer/indices.
    if (!visible) SetVisible(entity, false);

    MeshBuffers mesh_buffers{};
    for (auto element : AllElements) { // todo only create buffers for viewed elements.
        mesh_buffers.emplace(element, VkRenderBuffers{VC, mesh.CreateVertices(element), mesh.CreateIndices(element)});
    }
    MeshVkData->Main.emplace(entity, std::move(mesh_buffers));
    MeshVkData->NormalIndicators.emplace(entity, MeshBuffers{});

    if (ShowBoundingBoxes) {
        MeshVkData->Boxes.emplace(entity, VkRenderBuffers{VC, CreateBoxVertices(mesh.BoundingBox, EdgeColor), BBox::EdgeIndices});
    }

    if (select) SelectEntity(entity);
    if (submit) RecordAndSubmitCommandBuffer();
    return entity;
}
entt::entity Scene::AddMesh(const fs::path &file_path, const mat4 &transform, bool submit, bool select, bool visible) {
    return AddMesh(Mesh{file_path}, transform, submit, select, visible);
}

entt::entity Scene::AddPrimitive(Primitive primitive, const mat4 &transform, bool submit, bool select, bool visible) {
    auto entity = AddMesh(CreateDefaultPrimitive(primitive), transform, submit, select, visible);
    R.emplace<Primitive>(entity, primitive);
    return entity;
}

void Scene::ClearMeshes() {
    for (auto entity : R.view<Mesh>()) DestroyEntity(entity, false);
    RecordAndSubmitCommandBuffer();
}

void Scene::ReplaceMesh(entt::entity entity, Mesh &&mesh) {
    for (auto &[element, buffers] : MeshVkData->Main.at(entity)) {
        VC.UpdateBuffer(buffers.Vertices, mesh.CreateVertices(element));
        VC.UpdateBuffer(buffers.Indices, mesh.CreateIndices(element));
    }
    for (auto &[element, buffers] : MeshVkData->NormalIndicators.at(entity)) {
        VC.UpdateBuffer(buffers.Vertices, mesh.CreateNormalVertices(element));
        VC.UpdateBuffer(buffers.Indices, mesh.CreateNormalIndices(element));
    }
    if (auto buffers = MeshVkData->Boxes.find(entity); buffers != MeshVkData->Boxes.end()) {
        VC.UpdateBuffer(buffers->second.Vertices, CreateBoxVertices(mesh.BoundingBox, EdgeColor));
        // Box indices are always the same.
    }

    R.replace<Mesh>(entity, std::move(mesh));
}

entt::entity Scene::AddInstance(entt::entity parent, mat4 &&transform, bool visible) {
    const auto entity = R.create();
    // For now, we assume one-level deep hierarchy, so we don't allocate a models buffer for the instance.
    R.emplace<SceneNode>(entity, parent);
    auto &parent_node = R.get<SceneNode>(parent);
    parent_node.children.emplace_back(entity);
    R.emplace<Model>(entity, std::move(transform));
    SetVisible(entity, visible);
    SelectEntity(entity);

    return entity;
}

void Scene::DestroyEntity(entt::entity entity, bool submit) {
    if (entity == SelectedEntity) SelectEntity(entt::null);
    if (const auto mesh_entity = GetParentEntity(entity); mesh_entity != entity) return DestroyInstance(entity, submit);

    MeshVkData->Main.erase(entity);
    MeshVkData->NormalIndicators.erase(entity);
    MeshVkData->Models.erase(entity);
    MeshVkData->Boxes.erase(entity);

    const auto &node = R.get<SceneNode>(entity);
    for (const auto child : node.children) {
        if (child == SelectedEntity) SelectEntity(entt::null);
        R.destroy(child);
    }
    R.destroy(entity);
    if (submit) RecordAndSubmitCommandBuffer();
}

void Scene::DestroyInstance(entt::entity instance, bool submit) {
    if (instance == SelectedEntity) SelectEntity(entt::null);

    SetVisible(instance, false);
    std::erase(R.get<SceneNode>(R.get<SceneNode>(instance).parent).children, instance);
    R.destroy(instance);
    if (submit) RecordAndSubmitCommandBuffer();
}

entt::entity Scene::GetParentEntity(entt::entity entity) const {
    if (entity == entt::null) return entt::null;

    const auto &node = R.get<SceneNode>(entity);
    return node.parent == entt::null ? entity : GetParentEntity(node.parent);
}

Mesh &Scene::GetSelectedMesh() const { return R.get<Mesh>(GetParentEntity(SelectedEntity)); }

void Scene::SetModel(entt::entity entity, mat4 &&model, bool submit) {
    R.replace<Model>(entity, std::move(model));
    UpdateModelBuffer(entity);
    if (submit) SubmitCommandBuffer();
}

mat4 Scene::GetModel(entt::entity entity) const { return R.get<Model>(entity).Transform; }

void Scene::UpdateRenderBuffers(entt::entity mesh_entity, const MeshElementIndex &highlight_element) {
    auto &mesh = R.get<Mesh>(mesh_entity);
    auto &mesh_buffers = MeshVkData->Main.at(mesh_entity);
    const Mesh::ElementIndex highlight(highlight_element);
    for (auto element : AllElements) { // todo only update buffers for viewed elements.
        VC.UpdateBuffer(mesh_buffers.at(element).Vertices, mesh.CreateVertices(element, highlight));
    }
}

void Scene::SetExtent(vk::Extent2D extent) {
    Extent = extent;
    UpdateTransformBuffers(); // Depends on the aspect ratio.
    MainPipeline.SetExtent(extent);
    SilhouettePipeline.SetExtent(extent);
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

    EdgeDetectionPipeline.UpdateDescriptors({
        {SPT::EdgeDetection, "Tex", std::nullopt, vk::DescriptorImageInfo{*SilhouetteFillImageSampler, *SilhouettePipeline.OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal}},
    });
    EdgeDetectionPipeline.SetExtent(extent);
    SilhouetteEdgeImageSampler = VC.Device->createSamplerUnique({{}, vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest});
    MainPipeline.UpdateDescriptors({
        {SPT::Texture, "SilhouetteEdgeTexture", std::nullopt, vk::DescriptorImageInfo{*SilhouetteEdgeImageSampler, *EdgeDetectionPipeline.OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal}},
    });
}

std::vector<std::pair<SPT, MeshElement>> GetPipelineElements(RenderMode render_mode, ColorMode color_mode) {
    const SPT fill_pipeline = color_mode == ColorMode::Mesh ? SPT::Fill : SPT::DebugNormals;
    switch (render_mode) {
        case RenderMode::Vertices: return {{fill_pipeline, MeshElement::Vertex}};
        case RenderMode::Edges: return {{SPT::Line, MeshElement::Edge}};
        case RenderMode::Faces: return {{fill_pipeline, MeshElement::Face}};
        case RenderMode::FacesAndEdges: return {{fill_pipeline, MeshElement::Face}, {SPT::Line, MeshElement::Edge}};
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
            *MainPipeline.ResolveImage,
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }}
    );

    const auto selected_mesh_entity = GetParentEntity(SelectedEntity);
    const auto selected_model_buffer_index = GetModelBufferIndex(SelectedEntity);
    const bool render_silhouette = selected_model_buffer_index && SelectionMode == SelectionMode::Object;
    if (render_silhouette) {
        // Render the silhouette edges for the selected mesh instance.
        SilhouettePipeline.Begin(cb);
        SilhouettePipeline.Render(
            cb,
            SPT::Silhouette,
            MeshVkData->Main.at(selected_mesh_entity).at(MeshElement::Vertex),
            MeshVkData->Models.at(selected_mesh_entity),
            *selected_model_buffer_index
        );
        cb.endRenderPass();

        EdgeDetectionPipeline.Begin(cb);
        EdgeDetectionPipeline.GetShaderPipeline(SPT::EdgeDetection)->RenderQuad(cb);
        cb.endRenderPass();
    }

    // Render meshes.
    // todo reorganize mesh VK buffers to reduce the number of draw calls and pipeline switches.
    // todo next up:
    //      - update `MeshVkData->Models` and `GetModelBufferIndex` to keep `Models` contiguous with only visible, or
    //      - keep all models in the `MeshVkData` but then update `drawIndexed` to use a different strategy:
    //        -  https://www.reddit.com/r/vulkan/comments/b7u2hu/way_to_draw_multiple_meshes_with_different/
    //           vkCmdDrawIndexedIndirectCount & put the offsets in a UBO that you index with gl_DrawId.
    const auto &meshes = R.view<Mesh>();
    MainPipeline.Begin(cb, BackgroundColor);
    meshes.each([this, &cb](auto entity, auto &) {
        const auto &buffers = MeshVkData->Main.at(entity);
        const auto &models = MeshVkData->Models.at(entity);
        for (const auto [pipeline, element] : GetPipelineElements(RenderMode, ColorMode)) {
            MainPipeline.Render(cb, pipeline, buffers.at(element), models);
        }
    });

    // Render silhouette edge texture.
    if (render_silhouette) MainPipeline.GetShaderPipeline(SPT::Texture)->RenderQuad(cb);

    // Render normal indicators.
    meshes.each([this, &cb](auto entity, auto &) {
        const auto &buffers = MeshVkData->NormalIndicators.at(entity);
        const auto &models = MeshVkData->Models.at(entity);
        for (const auto &[element, normal_indicators] : buffers) {
            MainPipeline.Render(cb, SPT::Line, normal_indicators, models);
        }
    });

    if (ShowBoundingBoxes) {
        meshes.each([this, &cb](auto entity, auto &) {
            if (auto buffers = MeshVkData->Boxes.find(entity); buffers != MeshVkData->Boxes.end()) {
                MainPipeline.Render(cb, SPT::Line, buffers->second, MeshVkData->Models.at(entity));
            }
        });
    }
    if (ShowBvhBoxes) {
        meshes.each([this, &cb](auto entity, auto &) {
            if (auto buffers = MeshVkData->BvhBoxes.find(entity); buffers != MeshVkData->BvhBoxes.end()) {
                MainPipeline.Render(cb, SPT::Line, buffers->second, MeshVkData->Models.at(entity));
            }
        });
    }

    if (ShowGrid) MainPipeline.GetShaderPipeline(SPT::Grid)->RenderQuad(cb);

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
    MainPipeline.CompileShaders();
    SilhouettePipeline.CompileShaders();
    EdgeDetectionPipeline.CompileShaders();
}

void Scene::UpdateEdgeColors() {
    Mesh::EdgeColor = RenderMode == RenderMode::FacesAndEdges ? MeshEdgeColor : EdgeColor;
    R.view<Mesh>().each([this](auto entity, auto &) { UpdateRenderBuffers(entity, SelectedElement); });
}

void Scene::UpdateTransformBuffers() {
    const float aspect_ratio = Extent.width == 0 || Extent.height == 0 ? 1.f : float(Extent.width) / float(Extent.height);
    const ViewProj view_proj{Camera.GetViewMatrix(), Camera.GetProjectionMatrix(aspect_ratio)};
    VC.UpdateBuffer(*TransformBuffer, &view_proj);

    const ViewProjNearFar vpnf{view_proj.View, view_proj.Projection, Camera.NearClip, Camera.FarClip};
    VC.UpdateBuffer(*ViewProjNearFarBuffer, &vpnf);
}

void Scene::UpdateModelBuffer(entt::entity entity) {
    if (const auto buffer_index = GetModelBufferIndex(entity)) {
        const auto &model = R.get<Model>(entity);
        VC.UpdateBuffer(MeshVkData->Models.at(GetParentEntity(entity)), &model, *buffer_index * sizeof(Model), sizeof(Model));
    }
}

using namespace ImGui;

vec2 ToGlm(ImVec2 v) { return {v.x, v.y}; }
vk::ClearColorValue ToClearColor(vec4 v) { return {v.r, v.g, v.b, v.a}; }

// Returns a world space ray from the mouse into the scene.
Ray GetMouseWorldRay(Camera camera, vec2 view_extent) {
    // Mouse pos in content region
    const vec2 mouse_pos = ToGlm((GetMousePos() - GetCursorScreenPos()) / GetContentRegionAvail());
    // Normalized Device Coordinates, $\mathcal{NDC} \in [-1,1]^2$
    const vec2 mouse_pos_ndc{2 * mouse_pos.x - 1, 1 - 2 * mouse_pos.y};
    return camera.ClipPosToWorldRay(mouse_pos_ndc, view_extent.x / view_extent.y);
}

vec2 ToGlm(vk::Extent2D e) { return {float(e.width), float(e.height)}; }
vk::Extent2D ToVkExtent(vec2 e) { return {uint(e.x), uint(e.y)}; }

bool Scene::Render() {
    if (Extent.width != 0 && Extent.height != 0) {
        // Handle keyboard input.
        if (IsKeyPressed(ImGuiKey_Tab)) {
            SetSelectionMode(SelectionMode == SelectionMode::Object ? SelectionMode::Edit : SelectionMode::Object);
        }
        if (SelectionMode == SelectionMode::Edit) {
            if (IsKeyPressed(ImGuiKey_1)) SelectionElement = MeshElement::Vertex;
            if (IsKeyPressed(ImGuiKey_2)) SelectionElement = MeshElement::Edge;
            if (IsKeyPressed(ImGuiKey_3)) SelectionElement = MeshElement::Face;
        }
        if (SelectedEntity != entt::null && (IsKeyPressed(ImGuiKey_Delete) || IsKeyPressed(ImGuiKey_Backspace))) {
            DestroyEntity(SelectedEntity);
        }

        // Handle mouse input.
        if (IsWindowHovered() && IsMouseClicked(ImGuiMouseButton_Left)) {
            const auto mouse_world_ray = GetMouseWorldRay(Camera, ToGlm(Extent));
            if (SelectedEntity != entt::null && SelectionMode == SelectionMode::Edit && SelectionElement != MeshElement::None && R.all_of<Visible>(SelectedEntity)) {
                const auto &model = R.get<Model>(SelectedEntity);
                const auto mouse_ray = mouse_world_ray.WorldToLocal(model.Transform);
                const auto before_selected_element = SelectedElement;
                SelectedElement = {SelectionElement, -1};
                {
                    const auto &mesh = GetSelectedMesh();
                    if (SelectionElement == MeshElement::Vertex) SelectedElement = Mesh::ElementIndex{mesh.FindNearestVertex(mouse_ray)};
                    else if (SelectionElement == MeshElement::Edge) SelectedElement = Mesh::ElementIndex{mesh.FindNearestEdge(mouse_ray)};
                    else if (SelectionElement == MeshElement::Face) SelectedElement = Mesh::ElementIndex{mesh.FindNearestIntersectingFace(mouse_ray)};
                }
                if (SelectedElement != before_selected_element) {
                    UpdateRenderBuffers(GetParentEntity(SelectedEntity), SelectedElement);
                    SubmitCommandBuffer();
                }
            } else if (SelectionMode == SelectionMode::Object && (GetIO().KeyCtrl || GetIO().KeySuper)) {
                static std::multimap<float, entt::entity> hovered_entities_by_distance;
                hovered_entities_by_distance.clear();
                R.view<Model>().each([this, &mouse_world_ray](auto entity, const auto &model) {
                    if (!R.all_of<Visible>(entity)) return;

                    const auto &mesh = R.get<Mesh>(GetParentEntity(entity));
                    const auto mouse_ray = mouse_world_ray.WorldToLocal(model.Transform);
                    if (auto intersect_distance = mesh.Intersect(mouse_ray)) hovered_entities_by_distance.emplace(*intersect_distance, entity);
                });

                std::vector<entt::entity> sorted_hovered_entities;
                for (const auto &[distance, entity] : hovered_entities_by_distance) sorted_hovered_entities.emplace_back(entity);
                if (!sorted_hovered_entities.empty()) {
                    // Cycle through hovered entities.
                    auto it = std::ranges::find(sorted_hovered_entities, SelectedEntity);
                    if (it != sorted_hovered_entities.end()) ++it;
                    if (it == sorted_hovered_entities.end()) it = sorted_hovered_entities.begin();
                    SelectEntity(*it, true);
                } else {
                    SelectEntity(entt::null, true);
                }
            }
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
    if (SelectedEntity != entt::null) {
        mat4 model = R.get<Model>(SelectedEntity).Transform;
        bool view_changed, model_changed;
        Gizmo->Render(Camera, model, aspect_ratio, view_changed, model_changed);
        view_changed |= Camera.Tick();
        if (model_changed || view_changed) {
            if (model_changed) SetModel(SelectedEntity, std::move(model), false);
            if (view_changed) UpdateTransformBuffers();
            SubmitCommandBuffer();
        }
    } else {
        bool view_changed;
        Gizmo->Render(Camera, view_changed);
        view_changed |= Camera.Tick();
        if (view_changed) {
            UpdateTransformBuffers();
            SubmitCommandBuffer();
        }
    }
}

void DecomposeTransform(const mat4 &transform, vec3 &position, vec3 &rotation, vec3 &scale) {
    static vec3 skew;
    static vec4 perspective;
    static glm::quat orientation;
    glm::decompose(transform, scale, orientation, position, skew, perspective);
    rotation = glm::eulerAngles(orientation) * 180.f / glm::pi<float>(); // Convert radians to degrees
}

std::optional<Mesh> PrimitiveEditor(Primitive primitive, bool is_create = true) {
    const char *create_label = is_create ? "Add" : "Update";
    if (primitive == Primitive::Rect) {
        static vec2 size = {1.0, 1.0};
        InputFloat2("Size", &size.x);
        if (Button(create_label)) return Rect(size / 2.f);
    } else if (primitive == Primitive::Circle) {
        static float r = 0.5;
        InputFloat("Radius", &r);
        if (Button(create_label)) return Circle(r);
    } else if (primitive == Primitive::Cube) {
        static vec3 size = {1.0, 1.0, 1.0};
        InputFloat3("Size", &size.x);
        if (Button(create_label)) return Cuboid(size / 2.f);
    } else if (primitive == Primitive::IcoSphere) {
        static float r = 0.5;
        static int subdivisions = 3;
        InputFloat("Radius", &r);
        InputInt("Subdivisions", &subdivisions);
        if (Button(create_label)) return IcoSphere(r, uint(subdivisions));
    } else if (primitive == Primitive::UVSphere) {
        static float r = 0.5;
        InputFloat("Radius", &r);
        if (Button(create_label)) return UVSphere(r);
    } else if (primitive == Primitive::Torus) {
        static vec2 radii = {0.5, 0.2};
        static glm::ivec2 n_segments = {32, 16};
        InputFloat2("Major/minor radius", &radii.x);
        InputInt2("Major/minor segments", &n_segments.x);
        if (Button(create_label)) return Torus(radii.x, radii.y, uint(n_segments.x), uint(n_segments.y));
    } else if (primitive == Primitive::Cylinder) {
        static float r = 1, h = 1;
        InputFloat("Radius", &r);
        InputFloat("Height", &h);
        if (Button(create_label)) return Cylinder(r, h);
    } else if (primitive == Primitive::Cone) {
        static float r = 1, h = 1;
        InputFloat("Radius", &r);
        InputFloat("Height", &h);
        if (Button(create_label)) return Cone(r, h);
    }

    return std::nullopt;
}

void Scene::RenderConfig() {
    if (BeginTabBar("Scene controls")) {
        if (BeginTabItem("Object")) {
            {
                int selection_mode = int(SelectionMode);
                bool selection_mode_changed = false;
                PushID("SelectionMode");
                AlignTextToFramePadding();
                TextUnformatted("Selection mode:");
                SameLine();
                selection_mode_changed |= RadioButton("Object", &selection_mode, int(SelectionMode::Object));
                SameLine();
                selection_mode_changed |= RadioButton("Edit", &selection_mode, int(SelectionMode::Edit));
                if (selection_mode_changed) SetSelectionMode(::SelectionMode(selection_mode));
                if (SelectionMode == SelectionMode::Edit) {
                    AlignTextToFramePadding();
                    TextUnformatted("Edit mode:");
                    SameLine();
                    int element_selection_mode = int(SelectionElement);
                    for (const auto element : AllElements) {
                        std::string name = to_string(element);
                        Capitalize(name);
                        if (RadioButton(name.c_str(), &element_selection_mode, int(element))) SelectionElement = MeshElement(element);
                        if (element != AllElements.back()) SameLine();
                    }
                    Text("Selected %s: %s", to_string(SelectionElement).c_str(), SelectedElement.is_valid() ? std::to_string(SelectedElement.idx()).c_str() : "None");
                    if (SelectionElement == MeshElement::Vertex && SelectedElement.is_valid() && SelectedEntity != entt::null) {
                        const auto &mesh = GetSelectedMesh();
                        const auto pos = mesh.GetPosition(Mesh::VH(SelectedElement.idx()));
                        Text("Vertex %d: (%.4f, %.4f, %.4f)", SelectedElement.idx(), pos.x, pos.y, pos.z);
                    }
                }
                PopID();
            }
            if (SelectedEntity != entt::null) {
                PushID(uint(SelectedEntity));
                Text("Selected object: 0x%08x", uint(SelectedEntity));
                Indent();

                const auto &node = R.get<SceneNode>(SelectedEntity);
                if (auto parent_entity = node.parent; parent_entity != entt::null) {
                    AlignTextToFramePadding();
                    Text("Parent: 0x%08x", uint(parent_entity));
                    SameLine();
                    if (Button("Select")) SelectEntity(parent_entity, true);
                }
                if (!node.children.empty() && CollapsingHeader("Children")) {
                    for (const auto child : node.children) {
                        PushID(uint(child));
                        AlignTextToFramePadding();
                        Text("0x%08x", uint(child));
                        SameLine();
                        if (Button("Select")) SelectEntity(child, true);
                        PopID();
                    }
                }
                const auto &selected_mesh = GetSelectedMesh();
                TextUnformatted(
                    std::format("Vertices|Edges|Faces: {:L} | {:L} | {:L}", selected_mesh.GetVertexCount(), selected_mesh.GetEdgeCount(), selected_mesh.GetFaceCount()).c_str()
                );
                Text("Model buffer index: %s", GetModelBufferIndex(SelectedEntity) ? std::to_string(*GetModelBufferIndex(SelectedEntity)).c_str() : "None");
                Unindent();
                bool visible = R.all_of<Visible>(SelectedEntity);
                if (Checkbox("Visible", &visible)) {
                    SetVisible(SelectedEntity, visible);
                    RecordAndSubmitCommandBuffer();
                }

                if (const auto *primitive = R.try_get<Primitive>(SelectedEntity)) {
                    // Editor for the selected entity's primitive type.
                    if (auto new_mesh = PrimitiveEditor(*primitive, false)) {
                        ReplaceMesh(SelectedEntity, std::move(*new_mesh));
                        RecordAndSubmitCommandBuffer();
                    }
                }
                if (Button("Add instance")) {
                    AddInstance(GetParentEntity(SelectedEntity));
                    RecordAndSubmitCommandBuffer();
                }

                if (CollapsingHeader("Transform")) {
                    const auto &model = R.get<Model>(SelectedEntity).Transform;
                    glm::vec3 pos, rot, scale;
                    DecomposeTransform(model, pos, rot, scale);
                    bool model_changed = false;
                    model_changed |= DragFloat3("Position", &pos[0], 0.01f);
                    model_changed |= DragFloat3("Rotation (deg)", &rot[0], 1, -90, 90, "%.0f");
                    model_changed |= DragFloat3("Scale", &scale[0], 0.01f, 0.01f, 10);
                    if (model_changed) {
                        SetModel(SelectedEntity, glm::translate(pos) * mat4{glm::quat{{glm::radians(rot.x), glm::radians(rot.y), glm::radians(rot.z)}}} * glm::scale(scale));
                    }
                    Gizmo->RenderDebug();
                }
                PopID();
            } else {
                Text("Selected object: None");
            }

            if (CollapsingHeader("Add primitive")) {
                PushID("AddPrimitive");
                static int current_primitive_edit = int(Primitive::Cube);
                uint i = 0; // For line breaks
                for (const auto primitive : AllPrimitives) {
                    if (i++ % 3 != 0) SameLine();
                    RadioButton(to_string(primitive).c_str(), &current_primitive_edit, int(primitive));
                }
                if (auto mesh = PrimitiveEditor(Primitive(current_primitive_edit), true)) {
                    R.emplace<Primitive>(AddMesh(std::move(*mesh)), Primitive(current_primitive_edit));
                }
                PopID();
            }

            if (CollapsingHeader("All objects")) {
                for (auto entity : R.view<SceneNode>()) {
                    const auto &node = R.get<SceneNode>(entity);
                    if (node.parent == entt::null) {
                        if (TreeNodeEx(&node, ImGuiTreeNodeFlags_DefaultOpen, "0x%08x", uint(entity))) {
                            if (Button("Select")) SelectEntity(entity, true);
                            if (Button("Delete")) DestroyEntity(entity);
                            TreePop();
                        }
                    }
                }
            }
            EndTabItem();
        }

        if (BeginTabItem("Render")) {
            if (Checkbox("Show grid", &ShowGrid)) RecordAndSubmitCommandBuffer();
            if (Button("Recompile shaders")) {
                CompileShaders();
                RecordAndSubmitCommandBuffer();
            }
            SeparatorText("Render mode");
            PushID("RenderMode");
            int render_mode = int(RenderMode);
            bool render_mode_changed = RadioButton("Vertices", &render_mode, int(RenderMode::Vertices));
            SameLine();
            render_mode_changed |= RadioButton("Edges", &render_mode, int(RenderMode::Edges));
            SameLine();
            render_mode_changed |= RadioButton("Faces", &render_mode, int(RenderMode::Faces));
            SameLine();
            render_mode_changed |= RadioButton("Faces and edges", &render_mode, int(RenderMode::FacesAndEdges));
            PopID();

            int color_mode = int(ColorMode);
            bool color_mode_changed = false;
            if (RenderMode != RenderMode::Edges) {
                SeparatorText("Fill color mode");
                PushID("ColorMode");
                color_mode_changed |= RadioButton("Mesh", &color_mode, int(ColorMode::Mesh));
                color_mode_changed |= RadioButton("Normals", &color_mode, int(ColorMode::Normals));
                PopID();
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
            // todo go back to storing normal settings in a map of element type to bool,
            //   and ensure meshes/instances are created with the current normals
            if (SelectedEntity != entt::null) {
                const auto mesh_entity = GetParentEntity(SelectedEntity);
                const auto &mesh = R.get<Mesh>(mesh_entity);
                auto &normals = MeshVkData->NormalIndicators.at(mesh_entity);
                for (const auto element : AllNormalElements) {
                    bool has_normals = normals.contains(element);
                    std::string name = to_string(element);
                    Capitalize(name);
                    if (Checkbox(name.c_str(), &has_normals)) {
                        if (has_normals) {
                            normals.emplace(element, VkRenderBuffers{VC, mesh.CreateNormalVertices(element), mesh.CreateNormalIndices(element)});
                        } else {
                            normals.erase(element);
                        }
                        RecordAndSubmitCommandBuffer();
                    }
                    if (element != AllNormalElements.back()) SameLine();
                }
                if (Checkbox("BVH boxes", &ShowBvhBoxes)) {
                    auto &buffers = MeshVkData->BvhBoxes;
                    if (ShowBvhBoxes) buffers.emplace(mesh_entity, VkRenderBuffers{VC, mesh.CreateBvhBuffers(EdgeColor)});
                    else buffers.erase(mesh_entity);
                    RecordAndSubmitCommandBuffer();
                }
                SameLine(); // For Bounding boxes checkbox
            }
            if (Checkbox("Bounding boxes", &ShowBoundingBoxes)) {
                auto &buffers = MeshVkData->Boxes;
                R.view<Mesh>().each([&](auto entity, auto &mesh) {
                    if (ShowBoundingBoxes) buffers.emplace(entity, VkRenderBuffers{VC, CreateBoxVertices(mesh.BoundingBox, EdgeColor), BBox::EdgeIndices});
                    else buffers.erase(entity);
                });
                RecordAndSubmitCommandBuffer();
            }
            SeparatorText("Silhouette");
            if (ColorEdit4("Color", &SilhouetteDisplay.Color[0])) {
                VC.UpdateBuffer(*SilhouetteDisplayBuffer, &SilhouetteDisplay);
                SubmitCommandBuffer();
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
                UpdateTransformBuffers();
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
                VC.UpdateBuffer(*LightsBuffer, &Lights);
                SubmitCommandBuffer();
            }
            EndTabItem();
        }
        EndTabBar();
    }
}
