#pragma once

#include <vulkan/vulkan.hpp>

#include <memory>
#include <span>

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

    mvk::Buffer AllocateMvk(vk::DeviceSize size, vk::BufferUsageFlags usage) const {
        return {usage, Allocate(size, MemoryUsage::CpuOnly), Allocate(size, MemoryUsage::GpuOnly, usage)};
    }

    std::span<const std::byte> GetData(vk::Buffer) const;
    vk::DeviceSize GetAllocatedSize(vk::Buffer) const;
    vk::DeviceSize GetAllocatedSize(const mvk::Buffer &b) const { return GetAllocatedSize(b.DeviceBuffer); }

    void WriteRegion(vk::Buffer, std::span<const std::byte>, vk::DeviceSize offset = 0);
    void MoveRegion(vk::Buffer, vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize size);

private:
    std::span<std::byte> GetMappedData(vk::Buffer);

    VmaAllocator Vma{nullptr};
};
} // namespace mvk
