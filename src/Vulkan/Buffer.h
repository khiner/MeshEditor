#pragma once

#include <vulkan/vulkan.hpp>

#include <memory>

// Forwards to avoid including `vk_mem_alloc.h`.
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T *;

namespace mvk {
struct Buffer {
    vk::BufferUsageFlags Usage;
    vk::Buffer HostBuffer; // Host (staging) buffer (CPU)
    vk::Buffer DeviceBuffer; // Device buffer (GPU)
    vk::DeviceSize Size{0}; // Used size (not allocated size)
};

enum class MemoryUsage {
    Unknown,
    CpuOnly,
    CpuToGpu,
    GpuOnly,
    GpuToCpu,
};

struct BufferAllocator {
    BufferAllocator(vk::PhysicalDevice, vk::Device, VkInstance);
    ~BufferAllocator();

    vk::Buffer Allocate(vk::DeviceSize, MemoryUsage, vk::BufferUsageFlags = vk::BufferUsageFlagBits::eTransferSrc) const;

    mvk::Buffer AllocateMvk(vk::BufferUsageFlags usage, vk::DeviceSize bytes) const {
        return {usage, Allocate(bytes, MemoryUsage::CpuOnly), Allocate(bytes, MemoryUsage::GpuOnly, usage)};
    }

    const void *GetData(vk::Buffer) const;
    vk::DeviceSize GetAllocatedSize(vk::Buffer) const;
    vk::DeviceSize GetAllocatedSize(const mvk::Buffer &b) const { return GetAllocatedSize(b.DeviceBuffer); }

    void WriteRegion(vk::Buffer, const void *data, vk::DeviceSize offset, vk::DeviceSize bytes);
    void MoveRegion(vk::Buffer, vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize bytes);

private:
    void *GetMappedData(vk::Buffer);

    VmaAllocator Vma{nullptr};
};
} // namespace mvk
