#pragma once

#include <vulkan/vulkan.hpp>

#include <memory>

// Forwards to avoid including `vk_mem_alloc.h`.
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T *;
using VmaPool = struct VmaPool_T *;
using VmaAllocation = struct VmaAllocation_T *;
struct VmaAllocationInfo;

enum class MemoryUsage {
    Unknown,
    CpuOnly,
    CpuToGpu,
    GpuOnly,
    GpuToCpu,
};

struct VmaBuffer {
    VmaBuffer(const VmaAllocator &, VmaAllocation, const VmaAllocationInfo &, vk::Buffer);
    VmaBuffer(VmaBuffer &&) noexcept;

    ~VmaBuffer();

    VmaBuffer &operator=(VmaBuffer &&) noexcept;

    vk::Buffer operator*() const { return Buffer; }
    const void *GetData() const;
    vk::DeviceSize GetAllocatedSize() const;

    void WriteRegion(const void *data, vk::DeviceSize offset, vk::DeviceSize bytes);
    void MoveRegion(vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize bytes);

private:
    void *GetMappedData();

    const VmaAllocator &Allocator;
    struct AllocationInfo;
    std::unique_ptr<AllocationInfo> Allocation;
    vk::Buffer Buffer;
};

struct VmaBufferAllocator {
    VmaBufferAllocator(vk::PhysicalDevice, vk::Device, VkInstance);
    ~VmaBufferAllocator();

    VmaBuffer Allocate(vk::DeviceSize, MemoryUsage, vk::BufferUsageFlags = vk::BufferUsageFlagBits::eTransferSrc) const;

private:
    VmaAllocator Allocator{};
    struct AllocatorInfo;
    std::unique_ptr<AllocatorInfo> Info;
};
