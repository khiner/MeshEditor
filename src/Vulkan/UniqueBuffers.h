#pragma once

#include "UniqueBuffer.h"

#include <array>
#include <vector>

template<typename T>
constexpr std::span<const std::byte> as_bytes(const std::vector<T> &v) { return std::as_bytes(std::span{v}); }
template<typename T, uint32_t N>
constexpr std::span<const std::byte> as_bytes(const std::array<T, N> &v) { return std::as_bytes(std::span{v}); }
template<typename T>
constexpr std::span<const std::byte> as_bytes(const T &v) { return {reinterpret_cast<const std::byte *>(&v), sizeof(T)}; }

namespace mvk {
struct DeferredBufferReclaimer {
    void Retire(UniqueBuffer &&buffer) { Retired.emplace_back(std::move(buffer)); }
    void Reclaim() { Retired.clear(); }

private:
    mutable std::vector<UniqueBuffer> Retired;
};

struct BufferContext {
    BufferContext(vk::PhysicalDevice pd, vk::Device d, VkInstance instance, vk::CommandPool command_pool)
        : Allocator(pd, d, instance),
          TransferCb(std::move(d.allocateCommandBuffersUnique({command_pool, vk::CommandBufferLevel::ePrimary, 1}).front())) {
        TransferCb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    }
    ~BufferContext() {
        TransferCb->end();
    }

    UniqueVmaAllocator Allocator;
    vk::UniqueCommandBuffer TransferCb;
    mutable DeferredBufferReclaimer Reclaimer{};
};

struct UniqueBuffers {
    UniqueBuffers(const BufferContext &, vk::DeviceSize, vk::BufferUsageFlags);
    UniqueBuffers(const BufferContext &, std::span<const std::byte>, vk::BufferUsageFlags);
    UniqueBuffers(const UniqueBuffers &) = delete;
    UniqueBuffers(UniqueBuffers &&);
    UniqueBuffers &operator=(const UniqueBuffers &) = delete;
    ~UniqueBuffers();

    UniqueBuffers &operator=(UniqueBuffers &&);

    // Updates the buffer with the given data, growing it if necessary.
    void Update(std::span<const std::byte>, vk::DeviceSize offset = 0);

    // Grows the buffer if it's not big enough (to the next power of 2).
    void Reserve(vk::DeviceSize);
    template<typename T> void Update(const std::vector<T> &data) { Update(as_bytes(data)); }
    // Insert a region of a buffer by moving the data at or after the region to the end of the region and increasing the buffer size.
    // **Does nothing if the buffer doesn't have enough enough space allocated.**
    void Insert(std::span<const std::byte>, vk::DeviceSize offset);
    // Erase a region of a buffer by moving the data after the region to the beginning of the region and reducing the buffer size.
    // Doesn't free memory, so the allocated size will be greater than the used size.
    void Erase(vk::DeviceSize offset, vk::DeviceSize size);

    const BufferContext &Ctx;
    vk::DeviceSize UsedSize{0}; // Used (not allocated) bytes
    vk::BufferUsageFlags Usage;
    UniqueBuffer HostBuffer; // Host (staging) buffer (CPU)
    UniqueBuffer DeviceBuffer; // Device buffer (GPU)

    vk::DescriptorBufferInfo GetDescriptor() const { return {*DeviceBuffer, 0, vk::WholeSize}; }

private:
    void Retire();
};
} // namespace mvk
