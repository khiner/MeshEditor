#pragma once

#include "VulkanContext.h"

struct Scene {
    Scene(const VulkanContext &);
    ~Scene() = default; // Using unique handles, so no need to manually destroy anything.

    // Returns true if the scene was updated.
    bool Render(uint width, uint height, const vk::ClearColorValue &bg_color);

    struct TriangleContext {
        vk::SampleCountFlagBits MsaaSamples;
        vk::UniqueRenderPass RenderPass;
        vk::UniquePipeline GraphicsPipeline;
        vk::UniqueCommandPool CommandPool;
        std::vector<vk::UniqueCommandBuffer> CommandBuffers;
        vk::UniqueSampler TextureSampler;
        vk::Extent2D Extent;

        // The scene is rendered to an offscreen image and then resolved to this image using MSAA.
        vk::UniqueImage ResolveImage;
        vk::UniqueImageView ResolveImageView;
        vk::UniqueDeviceMemory ResolveImageMemory;
    };

    const VulkanContext &VC;
    TriangleContext TC;
};