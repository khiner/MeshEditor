#pragma once

#include "VmaBuffer.h"

struct VulkanBuffer {
    vk::BufferUsageFlags Usage;
    vk::DeviceSize Size;
    VmaBuffer HostBuffer, DeviceBuffer;

    vk::DeviceSize GetAllocatedSize() const { return DeviceBuffer.GetAllocatedSize(); }
};

// Simple wrapper around vertex and index buffers.
struct VkRenderBuffers {
    VulkanBuffer Vertices, Indices;
};

struct VulkanBufferAllocator {
    VulkanBufferAllocator(vk::PhysicalDevice physical, vk::Device device, VkInstance instance)
        : BufferAllocator(physical, device, instance) {}
    ~VulkanBufferAllocator() = default;

    VulkanBuffer CreateBuffer(vk::BufferUsageFlags usage, vk::DeviceSize bytes) const {
        return {
            usage,
            bytes,
            CreateStagingBuffer(bytes),
            // Device buffer: device (GPU)-local
            BufferAllocator.CreateVmaBuffer(bytes, vk::BufferUsageFlagBits::eTransferDst | usage, MemoryUsage::GpuOnly),
        };
    }

    // Host-visible and coherent CPU staging buffer
    VmaBuffer CreateStagingBuffer(vk::DeviceSize bytes) const {
        return BufferAllocator.CreateVmaBuffer(bytes, vk::BufferUsageFlagBits::eTransferSrc, MemoryUsage::CpuOnly);
    }

private:
    VmaBufferAllocator BufferAllocator;
};
