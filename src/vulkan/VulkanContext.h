#pragma once

#include <vulkan/vulkan.hpp>

#include <vector>

constexpr auto VkApiVersion = VK_API_VERSION_1_4;

vk::PhysicalDevice FindPhysicalDevice(const vk::UniqueInstance &);

// Headless-capable Vulkan bootstrap. Pass swapchain-related extensions in `instance_extensions`
// (and set `with_swapchain=true`) for windowed rendering; pass empty + `with_swapchain=false`
// for tests or other offscreen use.
struct VulkanContext {
    VulkanContext(std::vector<const char *> instance_extensions, bool with_swapchain = true);
    ~VulkanContext() = default;

    vk::UniqueInstance Instance;
    vk::PhysicalDevice PhysicalDevice;
    vk::UniqueDevice Device;

    uint32_t QueueFamily{uint32_t(-1)};
    vk::Queue Queue;
    vk::UniquePipelineCache PipelineCache;
    vk::UniqueDescriptorPool DescriptorPool;
};
