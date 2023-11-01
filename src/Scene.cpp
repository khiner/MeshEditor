#include "Scene.h"

#include "imgui.h"
#include <shaderc/shaderc.hpp>

#include "ImGuizmo.h"

#include "File.h"

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
    bool Render(Camera &camera) const {
        using namespace ImGui;

        const auto content_region = GetContentRegionAvail();
        const auto &window_pos = GetWindowPos();
        ImGuizmo::BeginFrame();
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetRect(window_pos.x, window_pos.y + GetTextLineHeightWithSpacing(), content_region.x, content_region.y);

        static const float ViewManipulateSize = 128;
        const auto view_manipulate_pos = window_pos + ImVec2{GetWindowContentRegionMax().x - ViewManipulateSize, GetWindowContentRegionMin().y};
        auto camera_view = camera.GetViewMatrix();
        const float camera_distance = camera.GetDistance();
        const bool changed = ImGuizmo::ViewManipulate(&camera_view[0][0], camera_distance, view_manipulate_pos, {ViewManipulateSize, ViewManipulateSize}, 0);
        camera.SetPositionFromView(camera_view);

        return changed;
    }

    void RenderDebug() {
        using namespace ImGui;
        using namespace ImGuizmo;

        SeparatorText("Gizmo");
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
};

static std::vector<Vertex3D> GenerateCubeVertices() {
    std::vector<Vertex3D> vertices;
    glm::vec3 positions[] = {
        {-0.5f, -0.5f, -0.5f},
        {0.5f, -0.5f, -0.5f},
        {0.5f, 0.5f, -0.5f},
        {-0.5f, 0.5f, -0.5f},
        {-0.5f, -0.5f, 0.5f},
        {0.5f, -0.5f, 0.5f},
        {0.5f, 0.5f, 0.5f},
        {-0.5f, 0.5f, 0.5f}};
    glm::vec3 normals[] = {
        {0.0f, 0.0f, -1.0f}, // Front
        {0.0f, 0.0f, 1.0f}, // Back
        {-1.0f, 0.0f, 0.0f}, // Left
        {1.0f, 0.0f, 0.0f}, // Right
        {0.0f, 1.0f, 0.0f}, // Top
        {0.0f, -1.0f, 0.0f} // Bottom
    };
    glm::vec4 colors[] = {
        {1.0f, 0.0f, 0.0f, 1.0f}, // Front
        {0.0f, 1.0f, 0.0f, 1.0f}, // Back
        {0.0f, 0.0f, 1.0f, 1.0f}, // Left
        {1.0f, 1.0f, 0.0f, 1.0f}, // Right
        {1.0f, 0.0f, 1.0f, 1.0f}, // Top
        {0.0f, 1.0f, 1.0f, 1.0f} // Bottom
    };
    uint8_t faces[] = {
        0, 1, 2, 2, 3, 0, // Front
        4, 5, 6, 6, 7, 4, // Back
        0, 1, 5, 5, 4, 0, // Bottom
        1, 2, 6, 6, 5, 1, // Right
        2, 3, 7, 7, 6, 2, // Top
        3, 0, 4, 4, 7, 3 // Left
    };
    for (int i = 0; i < 36; ++i) {
        uint8_t vertex_index = faces[i];
        uint8_t face_index = i / 6;
        vertices.push_back({positions[vertex_index], normals[face_index], colors[face_index]});
    }
    return vertices;
}

static std::vector<uint16_t> GenerateCubeIndices() {
    std::vector<uint16_t> indices;
    for (uint16_t i = 0; i < 36; ++i) indices.push_back(i);
    return indices;
}

static const std::vector<Vertex3D> CubeVertices = GenerateCubeVertices();
static const std::vector<uint16_t> CubeIndices = GenerateCubeIndices();

Scene::Scene(const VulkanContext &vc)
    : VC(vc),
      MsaaSamples(GetMaxUsableSampleCount(VC.PhysicalDevice)),
      CommandPool(VC.Device->createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, VC.QueueFamily})),
      CommandBuffers(VC.Device->allocateCommandBuffersUnique({CommandPool.get(), vk::CommandBufferLevel::ePrimary, FrameBufferCount})),
      TextureSampler(VC.Device->createSamplerUnique({{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear})),
      ShaderPipeline(*this),
      Gizmo(std::make_unique<::Gizmo>()) {
    // Perform depth testing, render into a multisampled offscreen image, then resolve into a single-sampled resolve image.
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

    CompileShaders();
}

Scene::~Scene(){}; // Using unique handles, so no need to manually destroy anything.

bool Scene::Render(uint width, uint height, const vk::ClearColorValue &bg_color) {
    if (Extent.width == width && Extent.height == height && !Dirty) return false;

    Dirty = false;
    Extent = vk::Extent2D{width, height};
    VC.Device->waitIdle();

    // Create a depth image, allocate memory, bind it, and create a depth image view
    const auto depth_image = VC.Device->createImageUnique({
        {},
        vk::ImageType::e2D,
        vk::Format::eD32Sfloat,
        vk::Extent3D{width, height, 1},
        1,
        1,
        vk::SampleCountFlagBits::e4,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::SharingMode::eExclusive,
    });
    const auto depth_mem_reqs = VC.Device->getImageMemoryRequirements(depth_image.get());
    const auto depth_image_memory = VC.Device->allocateMemoryUnique({depth_mem_reqs.size, VC.FindMemoryType(depth_mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)});
    VC.Device->bindImageMemory(depth_image.get(), depth_image_memory.get(), 0);
    const auto depth_image_view = VC.Device->createImageViewUnique({{}, depth_image.get(), vk::ImageViewType::e2D, vk::Format::eD32Sfloat, {}, {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}});

    // Create an offscreen image to render the scene into.
    const auto offscreen_image = VC.Device->createImageUnique({
        {},
        vk::ImageType::e2D,
        ImageFormat,
        vk::Extent3D{width, height, 1},
        1,
        1,
        MsaaSamples,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
        vk::SharingMode::eExclusive,
    });
    const auto image_mem_reqs = VC.Device->getImageMemoryRequirements(offscreen_image.get());
    const auto offscreen_image_memory = VC.Device->allocateMemoryUnique({image_mem_reqs.size, VC.FindMemoryType(image_mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)});
    VC.Device->bindImageMemory(offscreen_image.get(), offscreen_image_memory.get(), 0);
    const auto offscreen_image_view = VC.Device->createImageViewUnique({{}, offscreen_image.get(), vk::ImageViewType::e2D, ImageFormat, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});

    ResolveImage = VC.Device->createImageUnique({
        {},
        vk::ImageType::e2D,
        ImageFormat,
        vk::Extent3D{width, height, 1},
        1,
        1,
        vk::SampleCountFlagBits::e1, // Single-sampled resolve image.
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
        vk::SharingMode::eExclusive,
    });

    const auto resolve_image_mem_reqs = VC.Device->getImageMemoryRequirements(ResolveImage.get());
    ResolveImageMemory = VC.Device->allocateMemoryUnique({resolve_image_mem_reqs.size, VC.FindMemoryType(resolve_image_mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)});
    VC.Device->bindImageMemory(ResolveImage.get(), ResolveImageMemory.get(), 0);
    ResolveImageView = VC.Device->createImageViewUnique({{}, ResolveImage.get(), vk::ImageViewType::e2D, ImageFormat, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});

    const std::array image_views{*depth_image_view, *offscreen_image_view, *ResolveImageView};
    const auto framebuffer = VC.Device->createFramebufferUnique({{}, RenderPass.get(), image_views, width, height, 1});

    const auto &command_buffer = CommandBuffers[0];
    const vk::Viewport viewport{0.f, 0.f, float(width), float(height), 0.f, 1.f};
    const vk::Rect2D scissor{{0, 0}, Extent};
    command_buffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    command_buffer->setViewport(0, {viewport});
    command_buffer->setScissor(0, {scissor});

    const vk::ImageMemoryBarrier barrier{
        {},
        {},
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        ResolveImage.get(),
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    command_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::DependencyFlags{},
        0, nullptr, // No memory barriers.
        0, nullptr, // No buffer memory barriers.
        1, &barrier // 1 image memory barrier.
    );

    const vk::ClearValue clear_values[3] = {
        vk::ClearValue{vk::ClearDepthStencilValue{1.0f, 0}}, // Clear value for the depth attachment.
        vk::ClearValue{bg_color}, // Clear value for the color attachment.
        vk::ClearValue{}, // Placeholder for the resolve attachment (it's not being cleared, so the value is ignored).
    };
    command_buffer->beginRenderPass({RenderPass.get(), framebuffer.get(), vk::Rect2D{{0, 0}, Extent}, 3, clear_values}, vk::SubpassContents::eInline);
    command_buffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *ShaderPipeline.Pipeline);

    vk::DescriptorSet descriptor_sets[] = {ShaderPipeline.DescriptorSet.get()};
    command_buffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *ShaderPipeline.PipelineLayout, 0, 1, descriptor_sets, 0, nullptr);

    const vk::Buffer vertex_buffers[] = {*ShaderPipeline.VertexBuffer.Buffer};
    const vk::DeviceSize offsets[] = {0};
    command_buffer->bindVertexBuffers(0, 1, vertex_buffers, offsets);
    command_buffer->bindIndexBuffer(*ShaderPipeline.IndexBuffer.Buffer, 0, vk::IndexType::eUint16);

    command_buffer->drawIndexed(uint(CubeIndices.size()), 1, 0, 0, 0);
    command_buffer->endRenderPass();
    command_buffer->end();

    vk::SubmitInfo submit;
    submit.setCommandBuffers(*command_buffer);
    VC.Queue.submit(submit);
    VC.Device->waitIdle();

    return true;
}

void Scene::CompileShaders() {
    ShaderPipeline.CompileShaders();
    Dirty = true;
}

ShaderPipeline::ShaderPipeline(const Scene &scene)
    : S(scene),
      TransferCommandBuffers(S.VC.Device->allocateCommandBuffersUnique({*S.CommandPool, vk::CommandBufferLevel::ePrimary, S.FrameBufferCount})) {
    // Create descriptor set layout for transform and light buffers
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment},
    };
    DescriptorSetLayout = S.VC.Device->createDescriptorSetLayoutUnique({{}, bindings});
    PipelineLayout = S.VC.Device->createPipelineLayoutUnique({{}, 1, &(*DescriptorSetLayout), 0});

    CreateVertexBuffers(CubeVertices);
    CreateIndexBuffers(CubeIndices);

    // const float aspect_ratio = S.Extent.width / float(S.Extent.height);
    const float aspect_ratio = 1;
    CreateTransformBuffers({I, S.Camera.GetViewMatrix(), S.Camera.GetProjectionMatrix(aspect_ratio)});
    CreateLightBuffers(S.Light);

    // Allocate DescriptorSet
    vk::DescriptorSetAllocateInfo alloc_info{*S.VC.DescriptorPool, 1, &(*DescriptorSetLayout)};
    DescriptorSet = std::move(S.VC.Device->allocateDescriptorSetsUnique(alloc_info).front());

    // Update DescriptorSet
    vk::DescriptorBufferInfo transform_buffer_info{*TransformBuffer.Buffer, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo light_buffer_info{*LightBuffer.Buffer, 0, VK_WHOLE_SIZE};
    std::array<vk::WriteDescriptorSet, 2> write_descriptor_sets = {
        vk::WriteDescriptorSet{*DescriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &transform_buffer_info},
        vk::WriteDescriptorSet{*DescriptorSet, 1, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &light_buffer_info}};
    S.VC.Device->updateDescriptorSets(write_descriptor_sets, {});
}

void ShaderPipeline::CompileShaders() {
    static const shaderc::Compiler compiler;
    static shaderc::CompileOptions compile_opts;
    compile_opts.SetOptimizationLevel(shaderc_optimization_level_performance);

    static const vk::PipelineInputAssemblyStateCreateInfo input_assemply{{}, vk::PrimitiveTopology::eTriangleList, false};
    static const vk::PipelineViewportStateCreateInfo viewport_state{{}, 1, nullptr, 1, nullptr};
    static const vk::PipelineRasterizationStateCreateInfo rasterizer{{}, false, false, vk::PolygonMode::eFill, {}, vk::FrontFace::eCounterClockwise, {}, {}, {}, {}, 1.0f};
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

    const std::string VertShader = File::Read(ShadersDir / "Transform" / "Transform.vert");
    const std::string FragShader = File::Read(ShadersDir / "Transform" / "Transform.frag");

    const auto &device = S.VC.Device;
    const auto vert_shader_spv = compiler.CompileGlslToSpv(VertShader, shaderc_glsl_vertex_shader, "vertex shader", compile_opts);
    if (vert_shader_spv.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error(std::format("Failed to compile vertex shader: {}", vert_shader_spv.GetErrorMessage()));
    }
    const std::vector<uint> vert_shader_code{vert_shader_spv.cbegin(), vert_shader_spv.cend()};
    const auto vert_shader_module = device->createShaderModuleUnique({{}, vert_shader_code});

    const auto frag_shader_spv = compiler.CompileGlslToSpv(FragShader, shaderc_glsl_fragment_shader, "fragment shader", compile_opts);
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

void ShaderPipeline::CreateOrUpdateBuffer(Buffer &buffer, const void *data) {
    const auto &device = S.VC.Device;
    const auto &queue = S.VC.Queue;

    // If the staging buffer or its memory hasn't been created yet, create them
    if (!buffer.StagingBuffer || !buffer.StagingMemory) {
        buffer.StagingBuffer = device->createBufferUnique({{}, buffer.Size, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive});
        const auto staging_mem_reqs = device->getBufferMemoryRequirements(*buffer.StagingBuffer);
        buffer.StagingMemory = device->allocateMemoryUnique({staging_mem_reqs.size, S.VC.FindMemoryType(staging_mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)});
        device->bindBufferMemory(*buffer.StagingBuffer, *buffer.StagingMemory, 0);
    }

    // Copy data to the staging buffer
    void *mapped_data = device->mapMemory(*buffer.StagingMemory, 0, buffer.Size);
    memcpy(mapped_data, data, size_t(buffer.Size));
    device->unmapMemory(*buffer.StagingMemory);

    // If the device buffer or its memory hasn't been created yet, create them
    if (!buffer.Buffer || !buffer.Memory) {
        buffer.Buffer = device->createBufferUnique({{}, buffer.Size, buffer.Usage, vk::SharingMode::eExclusive});
        const auto buffer_mem_reqs = device->getBufferMemoryRequirements(*buffer.Buffer);
        buffer.Memory = device->allocateMemoryUnique({buffer_mem_reqs.size, S.VC.FindMemoryType(buffer_mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)});
        device->bindBufferMemory(*buffer.Buffer, *buffer.Memory, 0);
    }

    // Prepare a command buffer to copy data from the staging buffer to the device buffer
    const auto &command_buffer = TransferCommandBuffers[0];
    command_buffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::BufferCopy copy_region;
    copy_region.size = buffer.Size;
    command_buffer->copyBuffer(*buffer.StagingBuffer, *buffer.Buffer, copy_region);
    command_buffer->end();

    // Submit the command buffer for execution
    vk::SubmitInfo submit;
    submit.setCommandBuffers(*command_buffer);
    queue.submit(submit, nullptr);
    queue.waitIdle();
}

void ShaderPipeline::CreateVertexBuffers(const std::vector<Vertex3D> &vertices) {
    VertexBuffer.Size = sizeof(vertices[0]) * vertices.size();
    VertexBuffer.Usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer;
    CreateOrUpdateBuffer(VertexBuffer, vertices.data());
}

void ShaderPipeline::CreateIndexBuffers(const std::vector<uint16_t> &indices) {
    IndexBuffer.Size = sizeof(indices[0]) * indices.size();
    IndexBuffer.Usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer;
    CreateOrUpdateBuffer(IndexBuffer, indices.data() );
}

void ShaderPipeline::CreateTransformBuffers(const Transform &transform) {
    TransformBuffer.Size = sizeof(transform);
    TransformBuffer.Usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eUniformBuffer;
    CreateOrUpdateBuffer(TransformBuffer, &transform);
}

void ShaderPipeline::CreateLightBuffers(const Light &light) {
    LightBuffer.Size = sizeof(light);
    LightBuffer.Usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eUniformBuffer;
    CreateOrUpdateBuffer(LightBuffer, &light);
}

using namespace ImGui;

void Scene::RenderGizmo() {
    bool view_changed = false;
    if (Gizmo->Render(Camera)) view_changed = true;

    const auto &io = ImGui::GetIO();
    const bool window_hovered = IsWindowHovered();
    if (window_hovered && io.MouseWheel != 0) {
        Camera.SetTargetDistance(Camera.GetDistance() * (1.f - io.MouseWheel / 16.f));
    }
    if (Camera.Tick()) view_changed = true;

    if (view_changed) {
        const float aspect_ratio = float(Extent.width) / float(Extent.height);
        const Transform transform{I, Camera.GetViewMatrix(), Camera.GetProjectionMatrix(aspect_ratio)};
        ShaderPipeline.CreateOrUpdateBuffer(ShaderPipeline.TransformBuffer, &transform);
        Dirty = true;
    }
}

void Scene::RenderControls() {
    if (Button("Recompile shaders")) CompileShaders();
    Gizmo->RenderDebug();
}
