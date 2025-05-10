#pragma once

#include "VmaBuffer.h"

// mvk as in "MeshEditor Vulkan" or "My Vulkan"
namespace mvk {
struct Buffer {
    vk::BufferUsageFlags Usage;
    vk::DeviceSize Size; // Used size (not allocated size)
    VmaBuffer HostBuffer, DeviceBuffer;

    vk::DeviceSize GetAllocatedSize() const { return DeviceBuffer.GetAllocatedSize(); }
};

// Simple wrapper around vertex and index buffers.
struct RenderBuffers {
    Buffer Vertices, Indices;
};

struct BufferAllocator {
    BufferAllocator(vk::PhysicalDevice physical, vk::Device device, VkInstance instance)
        : Vma(physical, device, instance) {}
    ~BufferAllocator() = default;

    Buffer Allocate(vk::BufferUsageFlags usage, vk::DeviceSize bytes) const {
        return {
            usage,
            0, // Used size (not allocated size)
            AllocateStaging(bytes),
            // Device buffer: device (GPU)-local
            Vma.Allocate(bytes, MemoryUsage::GpuOnly, usage),
        };
    }

    // Host-visible and coherent CPU staging buffer
    VmaBuffer AllocateStaging(vk::DeviceSize bytes) const {
        return Vma.Allocate(bytes, MemoryUsage::CpuOnly);
    }

private:
    VmaBufferAllocator Vma;
};
} // namespace mvk
