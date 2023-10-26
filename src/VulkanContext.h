#pragma once

#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "Log.h"

inline static void CheckVk(VkResult err) {
    if (err != 0) throw std::runtime_error(std::format("Vulkan error: {}", int(err)));
}

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

inline static bool IsExtensionAvailable(const std::vector<vk::ExtensionProperties> &properties, const char *extension) {
    for (const vk::ExtensionProperties &p : properties)
        if (strcmp(p.extensionName, extension) == 0)
            return true;
    return false;
}

struct VulkanContext {
    vk::UniqueInstance Instance;
    vk::PhysicalDevice PhysicalDevice;
    vk::UniqueDevice Device;
    uint32_t QueueFamily = (uint32_t)-1;
    vk::Queue Queue;
    vk::UniquePipelineCache PipelineCache;
    vk::UniqueDescriptorPool DescriptorPool;

    void Init(std::vector<const char *> extensions);
    void Uninit();

    // Find a discrete GPU, or the first available (integrated) GPU.
    vk::PhysicalDevice FindPhysicalDevice() const;
    uint32_t FindMemoryType(uint32_t type_filter, vk::MemoryPropertyFlags props);

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

        vk::UniqueImage OffscreenImage;
        vk::UniqueImageView OffscreenImageView;

        vk::UniqueSampler TextureSampler;
        vk::DescriptorSet DescriptorSet; // Not unique, since this is returned by `ImGui_ImplVulkan_AddTexture` as a `VkDescriptorSet`.
    };

    TriangleContext TC;

    // Populates `TC`.
    void CreateTriangleContext(uint32_t width, uint32_t height);
};
