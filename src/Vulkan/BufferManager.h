#pragma once

#include "BufferAllocator.h"

#include <array>
#include <vector>

template<typename T>
constexpr std::span<const std::byte> as_bytes(const std::vector<T> &v) { return std::as_bytes(std::span{v}); }
template<typename T, uint32_t N>
constexpr std::span<const std::byte> as_bytes(const std::array<T, N> &v) { return std::as_bytes(std::span{v}); }
template<typename T>
constexpr std::span<const std::byte> as_bytes(const T &v) { return {reinterpret_cast<const std::byte *>(&v), sizeof(T)}; }

namespace mvk {
struct BufferManager;

struct UniqueBuffers {
    UniqueBuffers(const BufferManager &, vk::DeviceSize, vk::BufferUsageFlags);
    UniqueBuffers(const BufferManager &, std::span<const std::byte>, vk::BufferUsageFlags);
    UniqueBuffers(const UniqueBuffers &) = delete;
    UniqueBuffers(UniqueBuffers &&);
    UniqueBuffers &operator=(const UniqueBuffers &) = delete;
    ~UniqueBuffers();

    UniqueBuffers &operator=(UniqueBuffers &&);

    // Updates the buffer with the given data, growing it if necessary.
    void Update(std::span<const std::byte>, vk::DeviceSize offset = 0);

    // Grows the buffer if it's not big enough (to the next power of 2).
    void EnsureAllocated(vk::DeviceSize required_size);
    template<typename T> void Update(const std::vector<T> &data) { Update(as_bytes(data)); }
    // Insert a region of a buffer by moving the data at or after the region to the end of the region and increasing the buffer size.
    // **Does nothing if the buffer doesn't have enough enough space allocated.**
    void Insert(std::span<const std::byte>, vk::DeviceSize offset);
    // Erase a region of a buffer by moving the data after the region to the beginning of the region and reducing the buffer size.
    // Doesn't free memory, so the allocated size will be greater than the used size.
    void Erase(vk::DeviceSize offset, vk::DeviceSize size);

    const BufferManager &Manager;
    vk::BufferUsageFlags Usage;
    UniqueBuffer HostBuffer; // Host (staging) buffer (CPU)
    UniqueBuffer DeviceBuffer; // Device buffer (GPU)
    vk::DeviceSize Size{0}; // Used size (not allocated size)

    vk::DescriptorBufferInfo GetDescriptor() const { return {*DeviceBuffer, 0, vk::WholeSize}; }
};

// Wraps an allocator and a transfer command buffer to manage `mvk::UniqueBuffer`s.
struct BufferManager {
    BufferManager(vk::PhysicalDevice pd, vk::Device d, VkInstance instance, vk::CommandBuffer transfer_cb)
        : TransferCb(transfer_cb), Allocator(pd, d, instance) {
        TransferCb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    }
    ~BufferManager() {
        TransferCb.end();
    }

    VmaAllocator operator*() const { return *GetAllocator(); }
    const UniqueVmaAllocator &GetAllocator() const { return Allocator; }

    void Begin() const {
        StaleBuffers.clear();
        TransferCb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    }

    // Mark the buffer as unused so it can be garbage collected after the command buffer is submitted.
    void MarkStale(UniqueBuffer &&buffer) const { StaleBuffers.emplace_back(std::move(buffer)); }

    vk::CommandBuffer TransferCb;

private:
    UniqueVmaAllocator Allocator;
    // Buffers that are no longer used and can be destroyed after the command buffer is submitted.
    mutable std::vector<UniqueBuffer> StaleBuffers;
};
} // namespace mvk
