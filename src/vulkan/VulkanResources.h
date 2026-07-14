#pragma once

#include <vulkan/vulkan.hpp>

struct VulkanResources {
    vk::Instance Instance;
    vk::PhysicalDevice PhysicalDevice;
    vk::Device Device;
    uint32_t QueueFamily;
    vk::Queue Queue;
    // Max anisotropy for texture samplers (1 = off).
    float MaxSamplerAnisotropy{1.f};
};
