#pragma once

#include <vulkan/vulkan.hpp>

#include <cstdint>

struct SceneVulkanResources {
    vk::Instance Instance;
    vk::PhysicalDevice PhysicalDevice;
    vk::Device Device;
    uint32_t QueueFamily;
    vk::Queue Queue;
};
