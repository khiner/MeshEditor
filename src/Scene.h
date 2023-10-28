#pragma once

#include "VulkanContext.h"

struct Scene {
    Scene(const VulkanContext &, uint width, uint height);
    ~Scene() = default; // Using unique handles, so no need to manually destroy anything.

    struct TriangleContext {
        vk::UniqueShaderModule VertexShaderModule;
        vk::UniqueShaderModule FragmentShaderModule;
        vk::UniquePipelineLayout PipelineLayout;
        vk::UniquePipeline GraphicsPipeline;
        vk::UniqueRenderPass RenderPass;
        vk::UniqueCommandPool CommandPool;
        vk::UniqueFramebuffer Framebuffer; // Single framebuffer for offscreen rendering.
        std::vector<vk::UniqueCommandBuffer> CommandBuffers;

        vk::Extent2D Extent;

        // The scene is rendered to this image.
        vk::UniqueImage OffscreenImage;
        vk::UniqueImageView OffscreenImageView;
        vk::UniqueDeviceMemory OffscreenImageMemory;

        // The image is resolved to this image with MSAA.
        vk::UniqueImage ResolveImage;
        vk::UniqueImageView ResolveImageView;
        vk::UniqueDeviceMemory ResolveImageMemory;

        vk::UniqueSampler TextureSampler;
        vk::DescriptorSet DescriptorSet; // Not unique, since this is returned by `ImGui_ImplVulkan_AddTexture` as a `VkDescriptorSet`.
    };

    const VulkanContext &VC;
    TriangleContext TC;
};