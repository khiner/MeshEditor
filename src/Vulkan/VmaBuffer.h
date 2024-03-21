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
    VkBuffer Get() const;

    vk::DeviceSize GetAllocatedSize() const;

    void Update(const void *data, vk::DeviceSize offset, vk::DeviceSize bytes);

private:
    const VmaAllocator &Allocator;

    VkBuffer Buffer{VK_NULL_HANDLE};

    struct AllocationInfo;
    std::unique_ptr<AllocationInfo> Allocation;
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
