#pragma once

#include <vulkan/vulkan.hpp>

#include <memory>
#include <span>

// Forwards to avoid including `vk_mem_alloc.h`.
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T *;
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T *;

namespace mvk {
enum class MemoryUsage {
    Unknown,
    CpuOnly,
    CpuToGpu,
    GpuOnly,
    GpuToCpu,
};

struct UniqueBuffer {
    UniqueBuffer(VmaAllocator, vk::DeviceSize, mvk::MemoryUsage, vk::BufferUsageFlags = vk::BufferUsageFlagBits::eTransferSrc);
    UniqueBuffer(VmaAllocator, std::span<const std::byte>, mvk::MemoryUsage, vk::BufferUsageFlags = vk::BufferUsageFlagBits::eTransferSrc);
    UniqueBuffer(UniqueBuffer &&);
    UniqueBuffer(const UniqueBuffer &) = delete;
    ~UniqueBuffer();

    UniqueBuffer &operator=(UniqueBuffer &&);
    UniqueBuffer &operator=(const UniqueBuffer &) = delete;
    vk::Buffer operator*() const { return Get(); }

    vk::Buffer Get() const;
    std::span<const std::byte> GetData() const;
    vk::DeviceSize GetAllocatedSize() const;

    void Write(std::span<const std::byte>, vk::DeviceSize offset = 0) const;
    void Move(vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize size) const;

private:
    std::span<std::byte> GetMappedData() const;

    struct Impl;
    std::unique_ptr<Impl> Imp;
};

struct UniqueVmaAllocator {
    UniqueVmaAllocator(vk::PhysicalDevice, vk::Device, VkInstance);
    ~UniqueVmaAllocator();

    VmaAllocator operator*() const { return Get(); }

    VmaAllocator Get() const;
    std::string DebugHeapUsage() const;

private:
    vk::PhysicalDevice PhysicalDevice;
    vk::Device Device;
    VmaAllocator Vma;
};
} // namespace mvk
