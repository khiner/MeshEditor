#include "Scene.h"

#include <shaderc/shaderc.hpp>

#include "File.h"

#ifdef DEBUG_BUILD
static const fs::path ShadersDir = "../src/Shaders"; // Relative to `build/`.
#elif defined(RELEASE_BUILD)
// All files in `src/Shaders` are copied to `build/Shaders` at build time.
static const fs::path ShadersDir = "Shaders";
#endif

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

Scene::Scene(const VulkanContext &vc)
    : VC(vc),
      MsaaSamples(GetMaxUsableSampleCount(VC.PhysicalDevice)),
      CommandPool(VC.Device->createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, VC.QueueFamily})),
      CommandBuffers(VC.Device->allocateCommandBuffersUnique({CommandPool.get(), vk::CommandBufferLevel::ePrimary, FrameBufferCount})),
      TextureSampler(VC.Device->createSamplerUnique({{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear})),
      ShaderPipeline(*this) {
    // Render into a multisampled offscreen image, then resolve into a single-sampled resolve image.
    const std::vector<vk::AttachmentDescription> attachments{
        // Multisampled offscreen image.
        {{}, ImageFormat, MsaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        // Single-sampled resolve.
        {{}, ImageFormat, vk::SampleCountFlagBits::e1, {}, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference color_attachment_ref{0, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::AttachmentReference resolve_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, &resolve_attachment_ref};
    RenderPass = VC.Device->createRenderPassUnique({{}, attachments, subpass});

    CompileShaders();
}

bool Scene::Render(uint width, uint height, const vk::ClearColorValue &bg_color) {
    if (Extent.width == width && Extent.height == height && !HasNewShaders) return false;

    HasNewShaders = false;
    Extent = vk::Extent2D{width, height};
    VC.Device->waitIdle();

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
    const auto offscreen_image_view = VC.Device->createImageViewUnique({{}, offscreen_image.get(), vk::ImageViewType::e2D, ImageFormat, vk::ComponentMapping{}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});

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
    ResolveImageView = VC.Device->createImageViewUnique({{}, ResolveImage.get(), vk::ImageViewType::e2D, ImageFormat, vk::ComponentMapping{}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});

    const std::array image_views{*offscreen_image_view, *ResolveImageView};
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

    const vk::ClearValue clear_value{bg_color};
    command_buffer->beginRenderPass({RenderPass.get(), framebuffer.get(), vk::Rect2D{{0, 0}, Extent}, 1, &clear_value}, vk::SubpassContents::eInline);
    command_buffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *ShaderPipeline.Pipeline);
    command_buffer->draw(3, 1, 0, 0);
    command_buffer->endRenderPass();
    command_buffer->end();

    vk::SubmitInfo submit;
    submit.setCommandBuffers(command_buffer.get());
    VC.Queue.submit(submit);
    VC.Device->waitIdle();

    return true;
}

void Scene::CompileShaders() {
    ShaderPipeline.CompileShaders();
    HasNewShaders = true;
}

ShaderPipeline::ShaderPipeline(const Scene &scene)
    : S(scene), PipelineLayout(S.VC.Device->createPipelineLayoutUnique({})) {}

void ShaderPipeline::CompileShaders() {
    static shaderc::Compiler compiler;
    static shaderc::CompileOptions compile_opts;
    compile_opts.SetOptimizationLevel(shaderc_optimization_level_performance);

    static const vk::PipelineVertexInputStateCreateInfo vertex_input_info{{}, 0u, nullptr, 0u, nullptr};
    static const vk::PipelineInputAssemblyStateCreateInfo input_assemply{{}, vk::PrimitiveTopology::eTriangleList, false};
    static const vk::PipelineViewportStateCreateInfo viewport_state{{}, 1, nullptr, 1, nullptr};
    static const vk::PipelineRasterizationStateCreateInfo rasterizer{{}, false, false, vk::PolygonMode::eFill, {}, vk::FrontFace::eCounterClockwise, {}, {}, {}, {}, 1.0f};
    static const vk::PipelineMultisampleStateCreateInfo multisampling{{}, S.MsaaSamples, false};
    static const vk::PipelineColorBlendAttachmentState color_blend_attachment{
        {},
        /*srcCol*/ vk::BlendFactor::eOne,
        /*dstCol*/ vk::BlendFactor::eZero,
        /*colBlend*/ vk::BlendOp::eAdd,
        /*srcAlpha*/ vk::BlendFactor::eOne,
        /*dstAlpha*/ vk::BlendFactor::eZero,
        /*alphaBlend*/ vk::BlendOp::eAdd,
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    static const vk::PipelineColorBlendStateCreateInfo color_blending{{}, false, vk::LogicOp::eCopy, 1, &color_blend_attachment};
    static const std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    static vk::PipelineDynamicStateCreateInfo dynamic_state_info{{}, dynamic_states};

    const std::string VertShader = File::Read(ShadersDir / "Triangle.vert");
    const std::string FragShader = File::Read(ShadersDir / "Triangle.frag");

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
        *PipelineLayout,
        *S.RenderPass,
    };
    auto pipeline_result = device->createGraphicsPipelineUnique({}, pipeline_info);
    if (pipeline_result.result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to create graphics pipeline: {}", vk::to_string(pipeline_result.result)));
    }
    Pipeline = std::move(pipeline_result.value);
}
