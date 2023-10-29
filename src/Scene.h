#pragma once

#include "VulkanContext.h"

#include <glm/glm.hpp>

struct Scene {
    Scene(const VulkanContext &);
    ~Scene() = default; // Using unique handles, so no need to manually destroy anything.

    // Returns true if the scene was updated.
    bool Render(uint width, uint height, const glm::vec4 &bg_color);

    struct TriangleContext {
        vk::UniqueCommandPool CommandPool;
        std::vector<vk::UniqueCommandBuffer> CommandBuffers;

        vk::UniqueShaderModule VertexShaderModule;
        vk::UniqueShaderModule FragmentShaderModule;
        std::vector<vk::PipelineShaderStageCreateInfo> ShaderStages;
        vk::UniqueSampler TextureSampler;

        vk::UniquePipelineLayout PipelineLayout;
        vk::UniquePipeline GraphicsPipeline;
        vk::UniqueRenderPass RenderPass;
        vk::UniqueFramebuffer Framebuffer; // Single framebuffer for offscreen rendering.

        vk::Extent2D Extent;
        vk::SampleCountFlagBits MsaaSamples;

        // The scene is rendered to this image.
        vk::UniqueImage OffscreenImage;
        vk::UniqueImageView OffscreenImageView;
        vk::UniqueDeviceMemory OffscreenImageMemory;

        // The image is resolved to this image with MSAA.
        vk::UniqueImage ResolveImage;
        vk::UniqueImageView ResolveImageView;
        vk::UniqueDeviceMemory ResolveImageMemory;
    };

    const VulkanContext &VC;
    TriangleContext TC;
};