#pragma once

#include <memory>
#include <vulkan/vulkan.hpp>

// Forwards to avoid including `vk_mem_alloc.h`.
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T *;
using VmaPool = struct VmaPool_T *;
using VmaAllocation = struct VmaAllocation_T *;
struct VmaAllocationInfo;

enum class MemoryUsage {
    Unknown,
    GpuOnly,
    CpuOnly,
    CpuToGpu,
    GpuToCpu,
};

struct VmaBuffer {
    VmaBuffer(const VmaAllocator &, vk::DeviceSize, vk::BufferUsageFlags, MemoryUsage);
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

struct VulkanBuffer {
    vk::BufferUsageFlags Usage;
    vk::DeviceSize Size;
    VmaBuffer HostBuffer, DeviceBuffer;

    vk::DeviceSize GetAllocatedSize() const { return DeviceBuffer.GetAllocatedSize(); }
};

struct VulkanBufferAllocator {
    VulkanBufferAllocator(vk::PhysicalDevice, vk::Device, VkInstance);
    ~VulkanBufferAllocator();

    VmaBuffer CreateVmaBuffer(vk::DeviceSize, vk::BufferUsageFlags, MemoryUsage) const;

    VulkanBuffer CreateBuffer(vk::BufferUsageFlags usage, vk::DeviceSize bytes) const {
        return {
            usage,
            bytes,
            // Host buffer: host-visible and coherent staging buffer for CPU writes
            CreateVmaBuffer(bytes, vk::BufferUsageFlagBits::eTransferSrc, MemoryUsage::CpuOnly),
            // Device buffer: device (GPU)-local
            CreateVmaBuffer(bytes, vk::BufferUsageFlagBits::eTransferDst | usage, MemoryUsage::GpuOnly),
        };
    }

private:
    VmaAllocator Allocator{};
    struct AllocatorInfo;
    std::unique_ptr<AllocatorInfo> Info;
};
