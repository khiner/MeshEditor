#pragma once

#include <vulkan/vulkan.hpp>

struct VulkanBuffer {
    vk::BufferUsageFlags Usage;
    vk::DeviceSize Size{0};

    // GPU buffer.
    vk::UniqueBuffer Buffer{};
    vk::UniqueDeviceMemory Memory{};

    // Host staging buffer, used to transfer data to the GPU.
    vk::UniqueBuffer StagingBuffer{};
    vk::UniqueDeviceMemory StagingMemory{};
};
