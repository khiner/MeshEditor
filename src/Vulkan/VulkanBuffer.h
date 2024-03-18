#pragma once

#include "VmaBuffer.h"

struct VulkanBuffer {
    vk::BufferUsageFlags Usage;
    vk::DeviceSize Size;

    // Host staging buffer, used to transfer data to the GPU.
    VmaBuffer HostBuffer;

    // GPU buffer.
    VmaBuffer DeviceBuffer;
};
