#pragma once

#include "../Slots.h"
#include "UniqueBuffer.h"

#include <array>
#include <optional>
#include <vector>

template<typename T>
constexpr std::span<const std::byte> as_bytes(const std::vector<T> &v) { return std::as_bytes(std::span{v}); }
template<typename T, uint32_t N>
constexpr std::span<const std::byte> as_bytes(const std::array<T, N> &v) { return std::as_bytes(std::span{v}); }
template<typename T>
constexpr std::span<const std::byte> as_bytes(const T &v) { return {reinterpret_cast<const std::byte *>(&v), sizeof(T)}; }

struct DescriptorSlots;

namespace mvk {
struct DeferredBufferReclaimer {
    void Retire(UniqueBuffer &&buffer) { Retired.emplace_back(std::move(buffer)); }
    void Reclaim() { Retired.clear(); }

private:
    mutable std::vector<UniqueBuffer> Retired;
};

struct BufferContext {
    BufferContext(vk::PhysicalDevice pd, vk::Device d, VkInstance instance, vk::CommandPool command_pool, DescriptorSlots &slots)
        : Device(d),
          Allocator(pd, d, instance),
          TransferCb(std::move(d.allocateCommandBuffersUnique({command_pool, vk::CommandBufferLevel::ePrimary, 1}).front())),
          Slots(slots) {
        TransferCb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    }
    ~BufferContext() {
        TransferCb->end();
    }

    vk::Device Device;
    UniqueVmaAllocator Allocator;
    vk::UniqueCommandBuffer TransferCb;
    mutable DeferredBufferReclaimer Reclaimer{};
    DescriptorSlots &Slots;
};

// Wraps either a staging+device pair or a single host-visible device buffer.
// Manages its own bindless descriptor slot.
struct UniqueBuffers {
    UniqueBuffers(BufferContext &, vk::DeviceSize, vk::BufferUsageFlags, SlotType);
    UniqueBuffers(BufferContext &, std::span<const std::byte>, vk::BufferUsageFlags, SlotType);
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

    BufferContext &Ctx;
    SlotType Type;
    uint32_t Slot;
    vk::DeviceSize UsedSize{0}; // Used (not allocated) bytes
    vk::BufferUsageFlags Usage;
    std::optional<UniqueBuffer> HostBuffer; // Host (staging) buffer (CPU), if needed.
    UniqueBuffer DeviceBuffer; // Device buffer (GPU)

    vk::DescriptorBufferInfo GetDescriptor() const { return {*DeviceBuffer, 0, vk::WholeSize}; }

private:
    void Retire();
    void UpdateDescriptor();
    using UpdateFn = bool (*)(UniqueBuffers &, std::span<const std::byte>, vk::DeviceSize);
    using ReserveFn = bool (*)(UniqueBuffers &, vk::DeviceSize);
    using InsertFn = void (*)(UniqueBuffers &, std::span<const std::byte>, vk::DeviceSize);
    using EraseFn = void (*)(UniqueBuffers &, vk::DeviceSize, vk::DeviceSize);

    struct Impl {
        UpdateFn Update;
        ReserveFn Reserve;
        InsertFn Insert;
        EraseFn Erase;
    };

    Impl ImplOps{};
};
} // namespace mvk
