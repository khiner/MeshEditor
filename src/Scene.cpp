#include "Scene.h"

#include <shaderc/shaderc.hpp>

static const std::string TriangleVertShader = R"vertexshader(
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

static const std::string TriangleFragShader = R"frag_shader(
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
)frag_shader";

static const auto ImageFormat = vk::Format::eB8G8R8A8Unorm;

vk::SampleCountFlagBits GetMaxUsableSampleCount(const vk::PhysicalDevice physical_device) {
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

Scene::Scene(const VulkanContext &vc) : VC(vc) {
    shaderc::CompileOptions options;
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

    shaderc::Compiler compiler;
    const auto vert_shader_spv = compiler.CompileGlslToSpv(TriangleVertShader, shaderc_glsl_vertex_shader, "vertex shader", options);
    if (vert_shader_spv.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error(std::format("Failed to compile vertex shader: {}", vert_shader_spv.GetErrorMessage()));
    }
    const std::vector<uint> vert_shader_code{vert_shader_spv.cbegin(), vert_shader_spv.cend()};
    const auto vert_shader_module = VC.Device->createShaderModuleUnique({{}, vert_shader_code});

    const auto frag_shader_spv = compiler.CompileGlslToSpv(TriangleFragShader, shaderc_glsl_fragment_shader, "fragment shader", options);
    if (frag_shader_spv.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error(std::format("Failed to compile fragment shader: {}", frag_shader_spv.GetErrorMessage()));
    }
    const std::vector<uint> frag_shader_code{frag_shader_spv.cbegin(), frag_shader_spv.cend()};
    const auto frag_shader_module = VC.Device->createShaderModuleUnique({{}, frag_shader_code});

    const std::vector<vk::PipelineShaderStageCreateInfo> shader_stages{
        {{}, vk::ShaderStageFlagBits::eVertex, *vert_shader_module, "main"},
        {{}, vk::ShaderStageFlagBits::eFragment, *frag_shader_module, "main"},
    };

    // Render multisampled into the offscreen image, then resolve into a single-sampled resolve image.
    TC.MsaaSamples = GetMaxUsableSampleCount(VC.PhysicalDevice);
    const std::vector<vk::AttachmentDescription> attachments{
        // Multi-sampled offscreen image.
        {{}, ImageFormat, TC.MsaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        // Single-sampled resolve.
        {{}, ImageFormat, vk::SampleCountFlagBits::e1, {}, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference color_attachment_ref{0, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::AttachmentReference resolve_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, &resolve_attachment_ref};
    TC.RenderPass = VC.Device->createRenderPassUnique({{}, attachments, subpass});

    const vk::PipelineVertexInputStateCreateInfo vertex_input_info{{}, 0u, nullptr, 0u, nullptr};
    const vk::PipelineInputAssemblyStateCreateInfo input_assemply{{}, vk::PrimitiveTopology::eTriangleList, false};
    const vk::PipelineViewportStateCreateInfo viewport_state{{}, 1, nullptr, 1, nullptr};
    const vk::PipelineRasterizationStateCreateInfo rasterizer{{}, false, false, vk::PolygonMode::eFill, {}, vk::FrontFace::eCounterClockwise, {}, {}, {}, {}, 1.0f};
    const vk::PipelineMultisampleStateCreateInfo multisampling{{}, TC.MsaaSamples, false};
    const vk::PipelineColorBlendAttachmentState color_blend_attachment{
        {},
        /*srcCol*/ vk::BlendFactor::eOne,
        /*dstCol*/ vk::BlendFactor::eZero,
        /*colBlend*/ vk::BlendOp::eAdd,
        /*srcAlpha*/ vk::BlendFactor::eOne,
        /*dstAlpha*/ vk::BlendFactor::eZero,
        /*alphaBlend*/ vk::BlendOp::eAdd,
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    const vk::PipelineColorBlendStateCreateInfo color_blending{{}, false, vk::LogicOp::eCopy, 1, &color_blend_attachment};
    const std::array dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamic_state_info{{}, dynamic_states};

    auto pipeline_layout = VC.Device->createPipelineLayoutUnique({}, nullptr);
    const vk::GraphicsPipelineCreateInfo pipeline_info{
        {},
        shader_stages,
        &vertex_input_info,
        &input_assemply,
        nullptr,
        &viewport_state,
        &rasterizer,
        &multisampling,
        nullptr,
        &color_blending,
        &dynamic_state_info,
        *pipeline_layout,
        *TC.RenderPass,
    };
    TC.GraphicsPipeline = VC.Device->createGraphicsPipelineUnique({}, pipeline_info).value;

    static const uint framebuffer_count = 1;
    TC.CommandPool = VC.Device->createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, VC.QueueFamily});
    TC.CommandBuffers = VC.Device->allocateCommandBuffersUnique({TC.CommandPool.get(), vk::CommandBufferLevel::ePrimary, framebuffer_count});

    vk::SamplerCreateInfo sampler_info;
    sampler_info.magFilter = vk::Filter::eLinear;
    sampler_info.minFilter = vk::Filter::eLinear;
    TC.TextureSampler = VC.Device->createSamplerUnique(sampler_info);
}

bool Scene::Render(uint width, uint height, const vk::ClearColorValue &bg_color) {
    if (TC.Extent.width == width && TC.Extent.height == height) return false;

    TC.Extent = vk::Extent2D{width, height};
    VC.Device->waitIdle();

    // Create an offscreen image to render the scene into.
    const auto offscreen_image = VC.Device->createImageUnique({
        {},
        vk::ImageType::e2D,
        ImageFormat,
        vk::Extent3D{width, height, 1},
        1,
        1,
        TC.MsaaSamples,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
        vk::SharingMode::eExclusive,
    });
    const auto image_mem_reqs = VC.Device->getImageMemoryRequirements(offscreen_image.get());
    const auto offscreen_image_memory = VC.Device->allocateMemoryUnique({image_mem_reqs.size, VC.FindMemoryType(image_mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)});
    VC.Device->bindImageMemory(offscreen_image.get(), offscreen_image_memory.get(), 0);
    const auto offscreen_image_view = VC.Device->createImageViewUnique({{}, offscreen_image.get(), vk::ImageViewType::e2D, ImageFormat, vk::ComponentMapping{}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});

    TC.ResolveImage = VC.Device->createImageUnique({
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

    const auto resolve_image_mem_reqs = VC.Device->getImageMemoryRequirements(TC.ResolveImage.get());
    TC.ResolveImageMemory = VC.Device->allocateMemoryUnique({resolve_image_mem_reqs.size, VC.FindMemoryType(resolve_image_mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)});
    VC.Device->bindImageMemory(TC.ResolveImage.get(), TC.ResolveImageMemory.get(), 0);
    TC.ResolveImageView = VC.Device->createImageViewUnique({{}, TC.ResolveImage.get(), vk::ImageViewType::e2D, ImageFormat, vk::ComponentMapping{}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});

    const std::array image_views{*offscreen_image_view, *TC.ResolveImageView};
    const auto framebuffer = VC.Device->createFramebufferUnique({{}, TC.RenderPass.get(), image_views, width, height, 1});

    const auto &command_buffer = TC.CommandBuffers[0];
    const vk::Viewport viewport{0.f, 0.f, float(width), float(height), 0.f, 1.f};
    const vk::Rect2D scissor{{0, 0}, TC.Extent};
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
        TC.ResolveImage.get(),
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

    const vk::ClearValue clear_value{bg_color};
    command_buffer->beginRenderPass({TC.RenderPass.get(), framebuffer.get(), vk::Rect2D{{0, 0}, TC.Extent}, 1, &clear_value}, vk::SubpassContents::eInline);
    command_buffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *TC.GraphicsPipeline);
    command_buffer->draw(3, 1, 0, 0);
    command_buffer->endRenderPass();
    command_buffer->end();

    vk::SubmitInfo submit;
    submit.setCommandBuffers(command_buffer.get());
    VC.Queue.submit(submit);
    VC.Device->waitIdle();

    return true;
}
