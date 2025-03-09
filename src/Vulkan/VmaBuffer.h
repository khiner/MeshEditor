#pragma once

#include <vulkan/vulkan.hpp>

// Forwards to avoid including `vk_mem_alloc.h`.
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T *;

enum class MemoryUsage {
    Unknown,
    GpuOnly,
    CpuOnly,
    CpuToGpu,
    GpuToCpu,
};

struct VmaBuffer {
    VmaBuffer(const VmaAllocator &, vk::DeviceSize, vk::BufferUsageFlags, MemoryUsage);
    VmaBuffer(VmaBuffer &&);
    ~VmaBuffer();

    VmaBuffer &operator=(VmaBuffer &&) noexcept;

    VkBuffer operator*() const;
    const char *GetMappedData() const { return MappedData; }
    vk::DeviceSize GetAllocatedSize() const;

    void WriteRegion(const void *data, vk::DeviceSize offset, vk::DeviceSize bytes);
    void MoveRegion(vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize bytes);

private:
    // Map the memory to `MappedData` the allocation is host visible.
    void MapMemory();
    void UnmapMemory();

    const VmaAllocator &Allocator;
    struct AllocationInfo;
    std::unique_ptr<AllocationInfo> Allocation;

    VkBuffer Buffer{VK_NULL_HANDLE};
    char *MappedData{nullptr};
};

// See https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/quick_start.html
struct VulkanBufferAllocator {
    VulkanBufferAllocator(vk::PhysicalDevice, vk::Device, VkInstance);
    ~VulkanBufferAllocator();

    VmaBuffer CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, MemoryUsage memory_usage) {
        return {Allocator, size, usage, memory_usage};
    }

private:
    VmaAllocator Allocator{};

    struct AllocatorInfo;
    std::unique_ptr<AllocatorInfo> Info;
};
