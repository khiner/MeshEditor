#pragma once

#include "vulkan/VulkanResources.h"

#include <vulkan/vulkan.hpp>

constexpr auto VkApiVersion = VK_API_VERSION_1_4;

// For offscreen/test use pass empty `instance_extensions` and `with_swapchain=false`.
struct VulkanContext {
    VulkanContext(std::vector<const char *> instance_extensions, bool with_swapchain = true);

    vk::UniqueInstance Instance;
    vk::PhysicalDevice PhysicalDevice;
    vk::UniqueDevice Device;

    uint32_t QueueFamily{uint32_t(-1)};
    vk::Queue Queue;

    // Non-owning handle view
    VulkanResources Resources() const { return {*Instance, PhysicalDevice, *Device, QueueFamily, Queue}; }
};
