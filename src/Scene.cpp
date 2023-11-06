#include "Scene.h"

#include "imgui.h"
#include <shaderc/shaderc.hpp>

#include "ImGuizmo.h"

#include "File.h"
#include "Geometry/Primitive/Cuboid.h"

#ifdef DEBUG_BUILD
static const fs::path ShadersDir = "../src/Shaders"; // Relative to `build/`.
#elif defined(RELEASE_BUILD)
// All files in `src/Shaders` are copied to `build/Shaders` at build time.
static const fs::path ShadersDir = "Shaders";
#endif

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

    bool Render(Camera &camera, glm::mat4 &model, float aspect_ratio = 1) const {
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

GeometryInstance::GeometryInstance(const Scene &scene, Geometry &&geometry)
    : S(scene), G(std::make_unique<Geometry>(std::move(geometry))) {
    static const std::vector AllModes{RenderMode::Flat, RenderMode::Smooth, RenderMode::Lines};
    for (const auto mode : AllModes) {
        Buffers buffers;
        std::vector<Vertex3D> vertices = G->GenerateVertices(mode);
        buffers.VertexBuffer.Size = sizeof(Vertex3D) * vertices.size();
        S.CreateOrUpdateBuffer(buffers.VertexBuffer, vertices.data());

        std::vector<uint> indices = G->GenerateIndices(mode);
        buffers.IndexBuffer.Size = sizeof(uint) * indices.size();
        S.CreateOrUpdateBuffer(buffers.IndexBuffer, indices.data());
        BuffersForMode[mode] = std::move(buffers);
    }
}

void ImageResource::Create(const VulkanContext &vc, vk::ImageCreateInfo image_info, vk::ImageViewCreateInfo view_info, vk::MemoryPropertyFlags mem_props) {
    const auto &device = vc.Device;
    Image = device->createImageUnique(image_info);
    const auto mem_reqs = device->getImageMemoryRequirements(*Image);
    Memory = device->allocateMemoryUnique({mem_reqs.size, vc.FindMemoryType(mem_reqs.memoryTypeBits, mem_props)});
    device->bindImageMemory(*Image, *Memory, 0);
    view_info.image = *Image;
    View = device->createImageViewUnique(view_info);
}

ShaderPipeline::ShaderPipeline(
    const Scene &scene,
    const fs::path &vert_shader_path, const fs::path &frag_shader_path,
    vk::PolygonMode polygon_mode, vk::PrimitiveTopology topology
)
    : S(scene), VertexShaderPath(vert_shader_path), FragmentShaderPath(frag_shader_path), PolygonMode(polygon_mode), Topology(topology) {}

void ShaderPipeline::CompileShaders() {
    static const shaderc::Compiler compiler;
    static shaderc::CompileOptions compile_opts;
    compile_opts.SetOptimizationLevel(shaderc_optimization_level_performance);

    static const vk::PipelineViewportStateCreateInfo viewport_state{{}, 1, nullptr, 1, nullptr};
    static const vk::PipelineMultisampleStateCreateInfo multisampling{{}, S.MsaaSamples, false};
    static const vk::PipelineColorBlendAttachmentState color_blend_attachment{
        {},
        vk::BlendFactor::eOne, // srcCol
        vk::BlendFactor::eZero, // dstCol
        vk::BlendOp::eAdd, // colBlend
        vk::BlendFactor::eOne, // srcAlpha
        vk::BlendFactor::eZero, // dstAlpha
        vk::BlendOp::eAdd, // alphaBlend
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    static const vk::PipelineColorBlendStateCreateInfo color_blending{{}, false, vk::LogicOp::eCopy, 1, &color_blend_attachment};
    static const std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    static const vk::PipelineDynamicStateCreateInfo dynamic_state{{}, dynamic_states};
    static const vk::VertexInputBindingDescription vertex_binding{0, sizeof(Vertex3D), vk::VertexInputRate::eVertex};
    static const std::vector<vk::VertexInputAttributeDescription> vertex_attrs{
        {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex3D, Position)},
        {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex3D, Normal)},
        {2, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex3D, Color)},
    };
    static const vk::PipelineVertexInputStateCreateInfo vertex_input_state{{}, vertex_binding, vertex_attrs};

    static const vk::PipelineDepthStencilStateCreateInfo depth_stencil_state{
        {}, // flags
        VK_TRUE, // depthTestEnable
        VK_TRUE, // depthWriteEnable
        vk::CompareOp::eLess, // depthCompareOp
        VK_FALSE, // depthBoundsTestEnable
        VK_FALSE, // stencilTestEnable
        {}, // front (stencil state for front faces)
        {}, // back (stencil state for back faces)
        0.0f, // minDepthBounds
        1.0f // maxDepthBounds
    };

    const std::string vert_shader = File::Read(ShadersDir / VertexShaderPath);
    const std::string frag_shader = File::Read(ShadersDir / FragmentShaderPath);
    const vk::PipelineRasterizationStateCreateInfo rasterizer{{}, false, false, PolygonMode, {}, vk::FrontFace::eCounterClockwise, {}, {}, {}, {}, 1.0f};
    const vk::PipelineInputAssemblyStateCreateInfo input_assemply{{}, Topology, false};

    const auto &device = S.VC.Device;
    const auto vert_shader_spv = compiler.CompileGlslToSpv(vert_shader, shaderc_glsl_vertex_shader, "vertex shader", compile_opts);
    if (vert_shader_spv.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error(std::format("Failed to compile vertex shader: {}", vert_shader_spv.GetErrorMessage()));
    }
    const std::vector<uint> vert_shader_code{vert_shader_spv.cbegin(), vert_shader_spv.cend()};
    const auto vert_shader_module = device->createShaderModuleUnique({{}, vert_shader_code});

    const auto frag_shader_spv = compiler.CompileGlslToSpv(frag_shader, shaderc_glsl_fragment_shader, "fragment shader", compile_opts);
    if (frag_shader_spv.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error(std::format("Failed to compile fragment shader: {}", frag_shader_spv.GetErrorMessage()));
    }
    const std::vector<uint> frag_shader_code{frag_shader_spv.cbegin(), frag_shader_spv.cend()};
    const auto frag_shader_module = device->createShaderModuleUnique({{}, frag_shader_code});

    const std::vector<vk::PipelineShaderStageCreateInfo> shader_stages{
        {{}, vk::ShaderStageFlagBits::eVertex, *vert_shader_module, "main"},
        {{}, vk::ShaderStageFlagBits::eFragment, *frag_shader_module, "main"},
    };

    auto pipeline_result = device->createGraphicsPipelineUnique(
        {},
        {
            {},
            shader_stages,
            &vertex_input_state,
            &input_assemply,
            nullptr,
            &viewport_state,
            &rasterizer,
            &multisampling,
            &depth_stencil_state,
            &color_blending,
            &dynamic_state,
            *PipelineLayout,
            *S.RenderPass,
        }
    );
    if (pipeline_result.result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to create graphics pipeline: {}", vk::to_string(pipeline_result.result)));
    }
    Pipeline = std::move(pipeline_result.value);
}

FillShaderPipeline::FillShaderPipeline(const Scene &s, const fs::path &vert_shader_path, const fs::path &frag_shader_path)
    : ShaderPipeline(s, vert_shader_path, frag_shader_path, vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList) {
    std::vector<vk::DescriptorSetLayoutBinding> bindings{
        {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        {1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment},
    };
    DescriptorSetLayout = S.VC.Device->createDescriptorSetLayoutUnique({{}, bindings});
    PipelineLayout = S.VC.Device->createPipelineLayoutUnique({{}, 1, &(*DescriptorSetLayout), 0});

    vk::DescriptorSetAllocateInfo alloc_info{*S.VC.DescriptorPool, 1, &(*DescriptorSetLayout)};
    DescriptorSet = std::move(S.VC.Device->allocateDescriptorSetsUnique(alloc_info).front());

    vk::DescriptorBufferInfo transform_buffer_info{*S.TransformBuffer.Buffer, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo light_buffer_info{*S.LightBuffer.Buffer, 0, VK_WHOLE_SIZE};
    std::vector<vk::WriteDescriptorSet> write_descriptor_sets{
        {*DescriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &transform_buffer_info},
        {*DescriptorSet, 1, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &light_buffer_info},
    };
    S.VC.Device->updateDescriptorSets(write_descriptor_sets, {});
}

LineShaderPipeline::LineShaderPipeline(const Scene &s, const fs::path &vert_shader_path, const fs::path &frag_shader_path)
    : ShaderPipeline(s, vert_shader_path, frag_shader_path, vk::PolygonMode::eLine, vk::PrimitiveTopology::eLineList) {
    std::vector<vk::DescriptorSetLayoutBinding> bindings{
        {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex},
    };
    DescriptorSetLayout = S.VC.Device->createDescriptorSetLayoutUnique({{}, bindings});
    PipelineLayout = S.VC.Device->createPipelineLayoutUnique({{}, 1, &(*DescriptorSetLayout), 0});

    vk::DescriptorSetAllocateInfo alloc_info{*S.VC.DescriptorPool, 1, &(*DescriptorSetLayout)};
    DescriptorSet = std::move(S.VC.Device->allocateDescriptorSetsUnique(alloc_info).front());

    vk::DescriptorBufferInfo transform_buffer_info{*S.TransformBuffer.Buffer, 0, VK_WHOLE_SIZE};
    std::vector<vk::WriteDescriptorSet> write_descriptor_sets{
        {*DescriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &transform_buffer_info},
    };
    S.VC.Device->updateDescriptorSets(write_descriptor_sets, {});
}

Scene::Scene(const VulkanContext &vc)
    : VC(vc),
      MsaaSamples(GetMaxUsableSampleCount(VC.PhysicalDevice)),
      CommandPool(VC.Device->createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, VC.QueueFamily})),
      CommandBuffers(VC.Device->allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, FramebufferCount})),
      TransferCommandBuffers(VC.Device->allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, FramebufferCount})),
      RenderFence(VC.Device->createFenceUnique({})) {
    const float aspect_ratio = 1; // Initial aspect ratio doesn't matter, it will be updated on the first render.
    const Transform initial_transform{I, Camera.GetViewMatrix(), Camera.GetProjectionMatrix(aspect_ratio)};
    CreateOrUpdateBuffer(TransformBuffer, &initial_transform);
    CreateOrUpdateBuffer(LightBuffer, &Light);
    FillShaderPipeline = std::make_unique<::FillShaderPipeline>(*this, "Transform.vert", "Lighting.frag");
    LineShaderPipeline = std::make_unique<::LineShaderPipeline>(*this, "Transform.vert", "Basic.frag");

    TextureSampler = VC.Device->createSamplerUnique({{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear});
    Gizmo = std::make_unique<::Gizmo>();
    const std::vector<vk::AttachmentDescription> attachments{
        // Depth attachment.
        {{}, vk::Format::eD32Sfloat, MsaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
        // Multisampled offscreen image.
        {{}, ImageFormat, MsaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        // Single-sampled resolve.
        {{}, ImageFormat, vk::SampleCountFlagBits::e1, {}, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal}; // Reference to the depth attachment.
    const vk::AttachmentReference color_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::AttachmentReference resolve_attachment_ref{2, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, &resolve_attachment_ref, &depth_attachment_ref};

    RenderPass = VC.Device->createRenderPassUnique({{}, attachments, subpass});

    FillShaderPipeline->CompileShaders();
    LineShaderPipeline->CompileShaders();

    Geometries.emplace_back(*this, Cuboid{{0.5, 0.5, 0.5}});
}

Scene::~Scene(){}; // Using unique handles, so no need to manually destroy anything.

void Scene::SetExtent(vk::Extent2D extent) {
    Extent = extent;

    const auto transform = GetTransform();
    CreateOrUpdateBuffer(TransformBuffer, &transform);

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

void Scene::RecordCommandBuffer() {
    const auto &command_buffer = CommandBuffers[0];
    command_buffer->begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});
    command_buffer->setViewport(0, vk::Viewport{0.f, 0.f, float(Extent.width), float(Extent.height), 0.f, 1.f});
    command_buffer->setScissor(0, vk::Rect2D{{0, 0}, Extent});

    const std::vector<vk::ImageMemoryBarrier> image_memory_barriers{{
        {},
        {},
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        *ResolveImage,
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }};
    command_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, // No dependency flags.
        {}, // No memory barriers.
        {}, // No buffer memory barriers.
        image_memory_barriers
    );

    const auto *shader_pipeline = GetShaderPipeline();
    // Clear values for the depth, color, and (placeholder) resolve attachments.
    const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {BackgroundColor}, {}};
    command_buffer->beginRenderPass({*RenderPass, *Framebuffer, vk::Rect2D{{0, 0}, Extent}, clear_values}, vk::SubpassContents::eInline);
    command_buffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *shader_pipeline->Pipeline);

    vk::DescriptorSet descriptor_sets[] = {*shader_pipeline->DescriptorSet};
    command_buffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shader_pipeline->PipelineLayout, 0, 1, descriptor_sets, 0, nullptr);

    const auto &geometry_buffers = Geometries[0].GetBuffers(Mode);
    const vk::Buffer vertex_buffers[] = {*geometry_buffers.VertexBuffer.Buffer};
    const vk::DeviceSize offsets[] = {0};
    command_buffer->bindVertexBuffers(0, 1, vertex_buffers, offsets);
    command_buffer->bindIndexBuffer(*geometry_buffers.IndexBuffer.Buffer, 0, vk::IndexType::eUint32);
    command_buffer->drawIndexed(geometry_buffers.IndexBuffer.Size / sizeof(uint), 1, 0, 0, 0);
    command_buffer->endRenderPass();
    command_buffer->end();
}

void Scene::SubmitCommandBuffer(vk::Fence fence) const {
    vk::SubmitInfo submit;
    submit.setCommandBuffers(*CommandBuffers[0]);
    VC.Queue.submit(submit, fence);
}

void Scene::RecompileShaders() {
    GetShaderPipeline()->CompileShaders();
    RecordCommandBuffer();
    SubmitCommandBuffer();
}

bool Scene::Render(uint width, uint height, const vk::ClearColorValue &bg_color) {
    const bool viewport_changed = Extent.width != width || Extent.height != height;
    const bool bg_color_changed = BackgroundColor.float32 != bg_color.float32;
    if (!viewport_changed && !bg_color_changed) return false;

    BackgroundColor = bg_color;

    if (viewport_changed) SetExtent({width, height});
    if (viewport_changed || bg_color_changed) RecordCommandBuffer();
    SubmitCommandBuffer(*RenderFence);

    // The contract is that the caller may use the resolve image and sampler immediately after `Scene::Render` returns.
    // Returning `true` indicates that the resolve image/sampler have been recreated.
    auto wait_result = VC.Device->waitForFences(*RenderFence, VK_TRUE, UINT64_MAX);
    if (wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    VC.Device->resetFences(*RenderFence);

    return viewport_changed;
}

void Scene::CreateOrUpdateBuffer(Buffer &buffer, const void *data, bool force_recreate) const {
    const auto &device = VC.Device;
    const auto &queue = VC.Queue;

    // Optionally create the staging buffer and its memory.
    if (force_recreate || !buffer.StagingBuffer || !buffer.StagingMemory) {
        buffer.StagingBuffer = device->createBufferUnique({{}, buffer.Size, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive});
        const auto staging_mem_reqs = device->getBufferMemoryRequirements(*buffer.StagingBuffer);
        buffer.StagingMemory = device->allocateMemoryUnique({staging_mem_reqs.size, VC.FindMemoryType(staging_mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)});
        device->bindBufferMemory(*buffer.StagingBuffer, *buffer.StagingMemory, 0);
    }

    // Copy data to the staging buffer.
    void *mapped_data = device->mapMemory(*buffer.StagingMemory, 0, buffer.Size);
    memcpy(mapped_data, data, size_t(buffer.Size));
    device->unmapMemory(*buffer.StagingMemory);

    // Optionally create the device buffer and its memory.
    if (force_recreate || !buffer.Buffer || !buffer.Memory) {
        buffer.Buffer = device->createBufferUnique({{}, buffer.Size, vk::BufferUsageFlagBits::eTransferDst | buffer.Usage, vk::SharingMode::eExclusive});
        const auto buffer_mem_reqs = device->getBufferMemoryRequirements(*buffer.Buffer);
        buffer.Memory = device->allocateMemoryUnique({buffer_mem_reqs.size, VC.FindMemoryType(buffer_mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)});
        device->bindBufferMemory(*buffer.Buffer, *buffer.Memory, 0);
    }

    // Copy data from the staging buffer to the device buffer.
    const auto &command_buffer = TransferCommandBuffers[0];
    command_buffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::BufferCopy copy_region;
    copy_region.size = buffer.Size;
    command_buffer->copyBuffer(*buffer.StagingBuffer, *buffer.Buffer, copy_region);
    command_buffer->end();

    // TODO we should use a separate fence/semaphores for buffer updates and rendering.
    vk::SubmitInfo submit;
    submit.setCommandBuffers(*command_buffer);
    queue.submit(submit, *RenderFence);

    auto wait_result = device->waitForFences(*RenderFence, VK_TRUE, UINT64_MAX);
    if (wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    device->resetFences(*RenderFence);
}

Transform Scene::GetTransform() const {
    const float aspect_ratio = float(Extent.width) / float(Extent.height);
    return {ModelTransform, Camera.GetViewMatrix(), Camera.GetProjectionMatrix(aspect_ratio)};
}

void Scene::UpdateTransform() {
    const auto transform = GetTransform();
    CreateOrUpdateBuffer(TransformBuffer, &transform);
    SubmitCommandBuffer();
}

void Scene::UpdateLight() {
    CreateOrUpdateBuffer(LightBuffer, &Light);
    SubmitCommandBuffer();
}

using namespace ImGui;

void Scene::RenderGizmo() {
    // Handle mouse scroll.
    const float mouse_wheel = GetIO().MouseWheel;
    if (mouse_wheel != 0 && IsWindowHovered()) Camera.SetTargetDistance(Camera.GetDistance() * (1.f - mouse_wheel / 16.f));

    Gizmo->Begin();
    const float aspect_ratio = float(Extent.width) / float(Extent.height);
    bool view_or_model_changed = Gizmo->Render(Camera, ModelTransform, aspect_ratio);
    view_or_model_changed |= Camera.Tick();
    if (view_or_model_changed) UpdateTransform();
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
            }
            EndTabItem();
        }
        if (BeginTabItem("Light")) {
            bool light_changed = false;
            light_changed |= SliderFloat("Ambient intensity", &Light.ColorAndAmbient[3], 0, 1);
            light_changed |= ColorEdit3("Diffuse color", &Light.ColorAndAmbient[0]);
            light_changed |= SliderFloat3("Direction", &Light.Direction.x, -1, 1);
            if (light_changed) UpdateLight();
            EndTabItem();
        }
        if (BeginTabItem("Object")) {
            int render_mode = int(Mode);
            bool render_mode_changed = RadioButton("Flat", &render_mode, int(RenderMode::Flat));
            SameLine();
            render_mode_changed |= RadioButton("Smooth", &render_mode, int(RenderMode::Smooth));
            SameLine();
            render_mode_changed |= RadioButton("Lines", &render_mode, int(RenderMode::Lines));
            if (render_mode_changed) {
                Mode = RenderMode(render_mode);
                RecordCommandBuffer();
                SubmitCommandBuffer();
            }
            Gizmo->RenderDebug();
            EndTabItem();
        }
        if (BeginTabItem("Shader")) {
            if (Button("Recompile shaders")) RecompileShaders();
            EndTabItem();
        }
        EndTabBar();
    }
}
