#pragma once

#include "VmaBuffer.h"

struct VulkanBuffer {
    vk::BufferUsageFlags Usage;
    vk::DeviceSize Size; // Used size (not allocated size)
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

    VulkanBuffer Allocate(vk::BufferUsageFlags usage, vk::DeviceSize bytes) const {
        return {
            usage,
            bytes,
            AllocateStaging(bytes),
            // Device buffer: device (GPU)-local
            BufferAllocator.Allocate(bytes, MemoryUsage::GpuOnly, usage),
        };
    }

    // Host-visible and coherent CPU staging buffer
    VmaBuffer AllocateStaging(vk::DeviceSize bytes) const {
        return BufferAllocator.Allocate(bytes, MemoryUsage::CpuOnly);
    }

private:
    VmaBufferAllocator BufferAllocator;
};
