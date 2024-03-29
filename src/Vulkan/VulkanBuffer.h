#pragma once

#include "VmaBuffer.h"

struct VulkanBuffer {
    vk::BufferUsageFlags Usage;
    vk::DeviceSize Size; // Currently used size in bytes (not necessarily the allocated size).

    VmaBuffer HostBuffer; // Host staging buffer, used to transfer data to the GPU.
    VmaBuffer DeviceBuffer; // GPU buffer.

    // Assumes host and device buffer sizes are the same.
    vk::DeviceSize GetAllocatedSize() const { return DeviceBuffer.GetAllocatedSize(); }
};
