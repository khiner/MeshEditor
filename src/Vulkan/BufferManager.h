#pragma once

#include "BufferAllocator.h"

#include <array>
#include <unordered_set>
#include <vector>

template<typename T>
constexpr std::span<const std::byte> as_bytes(const std::vector<T> &v) { return std::as_bytes(std::span{v}); }
template<typename T, uint32_t N>
constexpr std::span<const std::byte> as_bytes(const std::array<T, N> &v) { return std::as_bytes(std::span{v}); }
template<typename T>
constexpr std::span<const std::byte> as_bytes(const T &v) { return {reinterpret_cast<const std::byte *>(&v), sizeof(T)}; }

namespace mvk {
struct Buffer {
    vk::BufferUsageFlags Usage;
    vk::Buffer HostBuffer; // Host (staging) buffer (CPU)
    vk::Buffer DeviceBuffer; // Device buffer (GPU)
    vk::DeviceSize Size{0}; // Used size (not allocated size)

    vk::DescriptorBufferInfo GetDescriptor() const { return {DeviceBuffer, 0, vk::WholeSize}; }
};

// Wraps an allocator and a transfer command buffer to manage `mvk::Buffer`s.
struct BufferManager {
    BufferManager(vk::PhysicalDevice pd, vk::Device d, VkInstance instance, vk::CommandBuffer cb)
        : Cb(cb), Allocator(pd, d, instance)  {
        Cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    }
    ~BufferManager() {
        Cb.end();
    }

    void Begin() const {
        for (auto stale_buffer : StaleBuffers) Allocator.Destroy(stale_buffer);
        StaleBuffers.clear();
        Cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    }

    Buffer Allocate(vk::DeviceSize size, vk::BufferUsageFlags usage) const {
        return {usage, Allocator.Allocate(size, MemoryUsage::CpuOnly), Allocator.Allocate(size, MemoryUsage::GpuOnly, usage)};
    }

    // Mark the buffer as unused so it can be garbage collected after the command buffer is submitted.
    void Invalidate(vk::Buffer buffer) const { StaleBuffers.insert(static_cast<VkBuffer>(buffer)); }
    void Invalidate(const Buffer &buffer) const {
        Invalidate(buffer.HostBuffer);
        Invalidate(buffer.DeviceBuffer);
    }

    vk::DeviceSize GetAllocatedSize(const Buffer &b) const { return Allocator.GetAllocatedSize(b.DeviceBuffer); }

    Buffer Create(std::span<const std::byte>, vk::BufferUsageFlags) const;
    vk::Buffer CreateStaging(std::span<const std::byte>) const;

    // Updates the buffer with the given data, growing it if necessary.
    void Update(Buffer &, std::span<const std::byte>, vk::DeviceSize offset = 0) const;
    // Grows the buffer if it's not big enough (to the next power of 2).
    void EnsureAllocated(Buffer &, vk::DeviceSize required_size) const;
    template<typename T> void Update(Buffer &buffer, const std::vector<T> &data) const {
        Update(buffer, as_bytes(data));
    }
    // Insert a region of a buffer by moving the data at or after the region to the end of the region and increasing the buffer size.
    // **Does nothing if the buffer doesn't have enough enough space allocated.**
    void Insert(Buffer &, std::span<const std::byte>, vk::DeviceSize offset) const;
    // Erase a region of a buffer by moving the data after the region to the beginning of the region and reducing the buffer size.
    // Doesn't free memory, so the allocated size will be greater than the used size.
    void Erase(Buffer &, vk::DeviceSize offset, vk::DeviceSize size) const;

    vk::CommandBuffer Cb; // Transfer command buffer

private:
    BufferAllocator Allocator;
    // Buffers that are no longer used and can be destroyed after the command buffer is submitted.
    mutable std::unordered_set<VkBuffer> StaleBuffers;
};
} // namespace mvk
