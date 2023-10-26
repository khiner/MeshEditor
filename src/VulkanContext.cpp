#include "VulkanContext.h"

#include <shaderc/shaderc.hpp>

#include "imgui_impl_vulkan.h"

void VulkanContext::Init(std::vector<const char *> extensions) {
    const auto instance_props = vk::enumerateInstanceExtensionProperties();
    if (IsExtensionAvailable(instance_props, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }

    vk::InstanceCreateFlags flags;
    if (IsExtensionAvailable(instance_props, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }

    extensions.push_back("VK_EXT_debug_utils");
    auto validation_layers = std::vector<const char *>{"VK_LAYER_KHRONOS_validation"};

    vk::ApplicationInfo app("", {}, "", {}, {});
    vk::InstanceCreateInfo instance_info{flags, &app, validation_layers, extensions};
    Instance = vk::createInstanceUnique(instance_info);

    auto dldi = vk::DispatchLoaderDynamic(Instance.get(), vkGetInstanceProcAddr);
    auto messenger = Instance->createDebugUtilsMessengerEXTUnique(
        vk::DebugUtilsMessengerCreateInfoEXT{
            {},
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo,
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            DebugCallback,
        },
        nullptr, dldi
    );

    PhysicalDevice = FindPhysicalDevice();

    const auto queue_family_props = PhysicalDevice.getQueueFamilyProperties();
    QueueFamily = std::distance(
        queue_family_props.begin(),
        std::find_if(queue_family_props.begin(), queue_family_props.end(), [](const auto &qfp) {
            return qfp.queueFlags & vk::QueueFlagBits::eGraphics;
        })
    );
    if (QueueFamily == static_cast<uint32_t>(-1)) throw std::runtime_error("No graphics queue family found.");

    // Create logical device (with 1 queue).
    const std::vector<const char *> device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, "VK_KHR_portability_subset"};
    const std::array<float, 1> queue_priority = {1.0f};
    const vk::DeviceQueueCreateInfo queue_info{{}, QueueFamily, 1, queue_priority.data()};
    const vk::DeviceCreateInfo device_info({}, queue_info, {}, device_extensions);
    Device = PhysicalDevice.createDeviceUnique(device_info);
    Queue = Device->getQueue(QueueFamily, 0);

    // Create descriptor pool.
    const std::array<vk::DescriptorPoolSize, 1> pool_sizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 2),
    };
    const vk::DescriptorPoolCreateInfo pool_info{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 2, pool_sizes};
    DescriptorPool = Device->createDescriptorPoolUnique(pool_info);
}

void VulkanContext::Uninit() {
    // Using unique handles, so no need to manually destroy anything.
}

vk::PhysicalDevice VulkanContext::FindPhysicalDevice() const {
    const auto physical_devices = Instance->enumeratePhysicalDevices();
    if (physical_devices.empty()) throw std::runtime_error("No Vulkan devices found.");

    for (const auto &device : physical_devices) {
        vk::PhysicalDeviceProperties properties = device.getProperties();
        if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) return device;
    }
    return physical_devices[0];
}

uint32_t VulkanContext::FindMemoryType(uint32_t type_filter, vk::MemoryPropertyFlags prop_flags) {
    vk::PhysicalDeviceMemoryProperties mem_props = PhysicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & prop_flags) == prop_flags) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

void VulkanContext::CreateTriangleContext(uint32_t width, uint32_t height) {
    TC.Extent = vk::Extent2D{width, height};

    static const std::string vert_shader = R"vertexshader(
        #version 450
        #extension GL_ARB_separate_shader_objects : enable

        out gl_PerVertex {
            vec4 gl_Position;
        };

        layout(location = 0) out vec3 fragColor;

        vec2 positions[3] = vec2[](
            vec2(0.0, -0.5),
            vec2(0.5, 0.5),
            vec2(-0.5, 0.5)
        );
        vec3 colors[3] = vec3[](
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
            fragColor = colors[gl_VertexIndex];
        }
        )vertexshader";

    static const std::string frag_shader = R"frag_shader(
        #version 450
        #extension GL_ARB_separate_shader_objects : enable

        layout(location = 0) in vec3 fragColor;

        layout(location = 0) out vec4 outColor;

        void main() {
            outColor = vec4(fragColor, 1.0);
        }
        )frag_shader";

    shaderc::CompileOptions options;
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

    shaderc::Compiler compiler;
    auto vert_shader_module = compiler.CompileGlslToSpv(vert_shader, shaderc_glsl_vertex_shader, "vertex shader", options);
    if (vert_shader_module.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error(std::format("Failed to compile vertex shader: {}", vert_shader_module.GetErrorMessage()));
    }
    auto vert_shader_code = std::vector<uint32_t>{vert_shader_module.cbegin(), vert_shader_module.cend()};
    auto vert_size = std::distance(vert_shader_code.begin(), vert_shader_code.end());
    auto vert_shader_info = vk::ShaderModuleCreateInfo{{}, vert_size * sizeof(uint32_t), vert_shader_code.data()};
    TC.VertexShaderModule = Device->createShaderModuleUnique(vert_shader_info);

    auto frag_shader_module = compiler.CompileGlslToSpv(frag_shader, shaderc_glsl_fragment_shader, "fragment shader", options);
    if (frag_shader_module.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error(std::format("Failed to compile fragment shader: {}", frag_shader_module.GetErrorMessage()));
    }
    auto frag_shader_code = std::vector<uint32_t>{frag_shader_module.cbegin(), frag_shader_module.cend()};
    auto frag_size = std::distance(frag_shader_code.begin(), frag_shader_code.end());
    auto frag_shader_info = vk::ShaderModuleCreateInfo{{}, frag_size * sizeof(uint32_t), frag_shader_code.data()};
    TC.FragmentShaderModule = Device->createShaderModuleUnique(frag_shader_info);

    auto vert_shader_stage_info = vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eVertex, *TC.VertexShaderModule, "main"};
    auto frag_shader_stage_info = vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eFragment, *TC.FragmentShaderModule, "main"};
    auto pipeline_shader_stages = std::vector<vk::PipelineShaderStageCreateInfo>{vert_shader_stage_info, frag_shader_stage_info};

    auto vertex_input_info = vk::PipelineVertexInputStateCreateInfo{{}, 0u, nullptr, 0u, nullptr};
    auto input_assemply = vk::PipelineInputAssemblyStateCreateInfo{{}, vk::PrimitiveTopology::eTriangleList, false};
    auto viewport = vk::Viewport{0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
    auto scissor = vk::Rect2D{{0, 0}, TC.Extent};
    auto viewport_state = vk::PipelineViewportStateCreateInfo{{}, 1, &viewport, 1, &scissor};

    auto rasterizer = vk::PipelineRasterizationStateCreateInfo{
        {}, /*depthClamp*/ false,
        /*rasterizeDiscard*/ false,
        vk::PolygonMode::eFill,
        {},
        /*frontFace*/ vk::FrontFace::eCounterClockwise,
        {},
        {},
        {},
        {},
        1.0f};

    auto multisampling = vk::PipelineMultisampleStateCreateInfo{{}, vk::SampleCountFlagBits::e1, false, 1.0};
    auto color_blend_attachment = vk::PipelineColorBlendAttachmentState{
        {}, /*srcCol*/ vk::BlendFactor::eOne,
        /*dstCol*/ vk::BlendFactor::eZero,
        /*colBlend*/ vk::BlendOp::eAdd,
        /*srcAlpha*/ vk::BlendFactor::eOne,
        /*dstAlpha*/ vk::BlendFactor::eZero,
        /*alphaBlend*/ vk::BlendOp::eAdd,
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    auto color_blending = vk::PipelineColorBlendStateCreateInfo{{}, /*logicOpEnable=*/false, vk::LogicOp::eCopy, /*attachmentCount=*/1, /*colourAttachments=*/&color_blend_attachment};

    TC.PipelineLayout = Device->createPipelineLayoutUnique({}, nullptr);

    auto format = vk::Format::eB8G8R8A8Unorm;
    auto color_attachment = vk::AttachmentDescription{
        {},
        format,
        vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eClear,
        vk::AttachmentStoreOp::eStore,
        {},
        {},
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eShaderReadOnlyOptimal};
    auto color_attachment_ref = vk::AttachmentReference{0, vk::ImageLayout::eColorAttachmentOptimal};

    auto subpass = vk::SubpassDescription{{}, vk::PipelineBindPoint::eGraphics, /*inAttachmentCount*/ 0, nullptr, 1, &color_attachment_ref};

    auto subpass_dependency = vk::SubpassDependency{VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite};
    TC.RenderPass = Device->createRenderPassUnique(vk::RenderPassCreateInfo{{}, 1, &color_attachment, 1, &subpass, 1, &subpass_dependency});

    auto pipeline_info = vk::GraphicsPipelineCreateInfo{{}, 2, pipeline_shader_stages.data(), &vertex_input_info, &input_assemply, nullptr, &viewport_state, &rasterizer, &multisampling, nullptr, &color_blending, nullptr, *TC.PipelineLayout, *TC.RenderPass, 0};
    TC.GraphicsPipeline = Device->createGraphicsPipelineUnique({}, pipeline_info).value;

    // Create an offscreen image to render the triangle into.
    vk::ImageCreateInfo image_info{
        vk::ImageCreateFlags{},
        vk::ImageType::e2D,
        format,
        vk::Extent3D{width, height, 1},
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
        vk::ImageLayout::eUndefined};

    TC.OffscreenImage = Device->createImageUnique(image_info);
    vk::MemoryRequirements mem_reqs = Device->getImageMemoryRequirements(TC.OffscreenImage.get());
    vk::MemoryAllocateInfo mem_alloc_info{mem_reqs.size, FindMemoryType(mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)};
    auto offscreen_memory = Device->allocateMemoryUnique(mem_alloc_info);
    Device->bindImageMemory(TC.OffscreenImage.get(), offscreen_memory.get(), 0);

    vk::ImageViewCreateInfo image_view_info{
        vk::ImageViewCreateFlags{},
        TC.OffscreenImage.get(),
        vk::ImageViewType::e2D,
        format,
        vk::ComponentMapping{},
        vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

    TC.OffscreenImageView = Device->createImageViewUnique(image_view_info);

    // Create a framebuffer using the offscreen image.
    const uint32_t framebuffer_count = 1;
    vk::FramebufferCreateInfo framebuffer_info{vk::FramebufferCreateFlags{}, TC.RenderPass.get(), 1, &(*TC.OffscreenImageView), width, height, 1};
    TC.Framebuffer = Device->createFramebufferUnique(framebuffer_info);
    TC.CommandPool = Device->createCommandPoolUnique({{}, QueueFamily});
    TC.CommandBuffers = Device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(TC.CommandPool.get(), vk::CommandBufferLevel::ePrimary, framebuffer_count));

    // Image layout transition to VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL before rendering.
    vk::ImageMemoryBarrier barrier;
    barrier.oldLayout = vk::ImageLayout::eUndefined;
    barrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = TC.OffscreenImage.get();
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    const auto &command_buffer = TC.CommandBuffers[0];
    command_buffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    command_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::DependencyFlags{}, 0, nullptr, 0, nullptr, 1, &barrier
    );
    vk::ClearValue clear_value{};
    const auto render_pass_begin_info = vk::RenderPassBeginInfo{TC.RenderPass.get(), TC.Framebuffer.get(), vk::Rect2D{{0, 0}, TC.Extent}, 1, &clear_value};
    command_buffer->beginRenderPass(render_pass_begin_info, vk::SubpassContents::eInline);
    command_buffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *TC.GraphicsPipeline);
    command_buffer->draw(3, 1, 0, 0);
    command_buffer->endRenderPass();
    command_buffer->end();

    vk::SubmitInfo submit;
    submit.setCommandBuffers(command_buffer.get());
    Queue.submit(submit);
    Device->waitIdle();

    vk::SamplerCreateInfo sampler_info;
    sampler_info.magFilter = vk::Filter::eLinear;
    sampler_info.minFilter = vk::Filter::eLinear;
    sampler_info.addressModeU = vk::SamplerAddressMode::eRepeat;
    sampler_info.addressModeV = vk::SamplerAddressMode::eRepeat;
    sampler_info.addressModeW = vk::SamplerAddressMode::eRepeat;
    sampler_info.anisotropyEnable = VK_FALSE;
    sampler_info.maxAnisotropy = 1;
    sampler_info.borderColor = vk::BorderColor::eIntOpaqueBlack;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = vk::CompareOp::eAlways;
    sampler_info.mipmapMode = vk::SamplerMipmapMode::eLinear;

    TC.TextureSampler = Device->createSamplerUnique(sampler_info);
    TC.DescriptorSet = ImGui_ImplVulkan_AddTexture(TC.TextureSampler.get(), TC.OffscreenImageView.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}
