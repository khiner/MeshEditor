#pragma once

#include <vulkan/vulkan.hpp>

#include <memory>
#include <span>

// Forwards to avoid including `vk_mem_alloc.h`.
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T *;

namespace mvk {
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

    std::span<const std::byte> GetData(vk::Buffer) const;
    vk::DeviceSize GetAllocatedSize(vk::Buffer) const;

    void WriteRegion(vk::Buffer, std::span<const std::byte>, vk::DeviceSize offset = 0) const;
    void MoveRegion(vk::Buffer, vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize size) const;

private:
    std::span<std::byte> GetMappedData(vk::Buffer) const;

    VmaAllocator Vma{nullptr};
};
} // namespace mvk
