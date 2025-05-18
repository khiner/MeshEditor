#pragma once

#include "BufferAllocator.h"

#include <array>
#include <functional>
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
};

// Wraps an allocator and a transfer command buffer.
struct BufferManager {
    using SubmitTransferFn = std::function<void()>;

    BufferManager(vk::PhysicalDevice pd, vk::Device d, VkInstance instance, vk::CommandBuffer cb, SubmitTransferFn submit_transfer)
        : Allocator(pd, d, instance), Cb(cb), SubmitTransfer(std::move(submit_transfer)) {}
    ~BufferManager() = default;

    Buffer Allocate(vk::DeviceSize size, vk::BufferUsageFlags usage) const {
        return {usage, Allocator.Allocate(size, MemoryUsage::CpuOnly), Allocator.Allocate(size, MemoryUsage::GpuOnly, usage)};
    }
    vk::DeviceSize GetAllocatedSize(const Buffer &b) const { return Allocator.GetAllocatedSize(b.DeviceBuffer); }

    Buffer Create(std::span<const std::byte>, vk::BufferUsageFlags) const;
    vk::Buffer CreateStaging(std::span<const std::byte>) const;

    // Grows the buffer if it's not big enough (to the next power of 2).
    void Update(mvk::Buffer &, std::span<const std::byte>, vk::DeviceSize offset = 0) const;
    // Returns a new buffer if resize is needed.
    std::optional<mvk::Buffer> EnsureAllocated(const Buffer &, vk::DeviceSize required_size) const;
    template<typename T> void Update(mvk::Buffer &buffer, const std::vector<T> &data) const {
        Update(buffer, as_bytes(data));
    }
    // Insert a region of a buffer by moving the data at or after the region to the end of the region and increasing the buffer size.
    // **It does nothing if the buffer doesn't have enough enough space allocated.**
    void InsertRegion(mvk::Buffer &, std::span<const std::byte>, vk::DeviceSize offset) const;
    // Erase a region of a buffer by moving the data after the region to the beginning of the region and reducing the buffer size.
    // It doesn't free memory, so the allocated size will be greater than the used size.
    void EraseRegion(mvk::Buffer &, vk::DeviceSize offset, vk::DeviceSize size) const;

    const vk::CommandBuffer &GetCommandBuffer() const { return Cb; }

private:
    BufferAllocator Allocator;
    vk::CommandBuffer Cb; // Transfer command buffer
    SubmitTransferFn SubmitTransfer; // xxx temporary until decoupled
};
} // namespace mvk
