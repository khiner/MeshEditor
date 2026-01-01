#pragma once

#include "Slots.h"

#include <vulkan/vulkan.hpp>

#include <array>
#include <memory>
#include <span>
#include <unordered_set>
#include <vector>

// Forwards to avoid including `vk_mem_alloc.h`.
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T *;
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T *;

template<typename T>
constexpr std::span<const std::byte> as_bytes(const std::vector<T> &v) { return std::as_bytes(std::span{v}); }
template<typename T, uint32_t N>
constexpr std::span<const std::byte> as_bytes(const std::array<T, N> &v) { return std::as_bytes(std::span{v}); }
template<typename T>
constexpr std::span<const std::byte> as_bytes(const T &v) { return {reinterpret_cast<const std::byte *>(&v), sizeof(T)}; }

struct DescriptorSlots;

namespace mvk {
struct DeferredBufferReclaimer;

enum class MemoryUsage {
    Unknown,
    CpuOnly,
    CpuToGpu,
    GpuOnly,
    GpuToCpu
};

struct BufferContext {
    BufferContext(vk::PhysicalDevice, vk::Device, vk::Instance, vk::CommandPool, DescriptorSlots &);
    ~BufferContext();

    void ReclaimRetiredBuffers();

    std::vector<vk::WriteDescriptorSet> GetPendingDescriptorUpdates();
    void AddPendingDescriptorUpdate(SlotType type, uint32_t slot, const vk::DescriptorBufferInfo &info) {
        CancelDescriptorUpdate(type, slot);
        PendingDescriptorUpdates.emplace(type, slot, info);
    }
    void CancelDescriptorUpdate(SlotType, uint32_t);
    void ClearPendingDescriptorUpdates() { PendingDescriptorUpdates.clear(); }

    std::string DebugHeapUsage() const;

    vk::PhysicalDevice PhysicalDevice;
    vk::Device Device;
    VmaAllocator Vma;
    vk::UniqueCommandBuffer TransferCb;
    std::unique_ptr<DeferredBufferReclaimer> Reclaimer;
    DescriptorSlots &Slots;

private:
    struct PendingDescriptorUpdate {
        SlotType Type;
        uint32_t Slot;
        vk::DescriptorBufferInfo Info;

        bool operator==(const auto &other) const noexcept { return Type == other.Type && Slot == other.Slot; }
    };
    struct PendingDescriptorUpdateHash {
        size_t operator()(const PendingDescriptorUpdate &pending) const noexcept {
            const auto type = static_cast<uint64_t>(std::to_underlying(pending.Type));
            return std::hash<uint64_t>{}((type << 32) | static_cast<uint64_t>(pending.Slot));
        }
    };
    std::unordered_set<PendingDescriptorUpdate, PendingDescriptorUpdateHash> PendingDescriptorUpdates;
};

// Vulkan buffer with optional host staging and descriptor slot management.
struct Buffer {
    // Slotted buffer (GpuOnly + staging, with descriptor slot management)
    Buffer(BufferContext &, vk::DeviceSize, vk::BufferUsageFlags, SlotType);
    Buffer(BufferContext &, std::span<const std::byte>, vk::BufferUsageFlags, SlotType);
    // Raw buffer (no slot, direct memory access)
    Buffer(BufferContext &, vk::DeviceSize, MemoryUsage, vk::BufferUsageFlags = {});
    Buffer(BufferContext &, std::span<const std::byte>, MemoryUsage, vk::BufferUsageFlags = {});

    Buffer(const Buffer &) = delete;
    Buffer(Buffer &&);
    Buffer &operator=(const Buffer &) = delete;
    Buffer &operator=(Buffer &&);
    ~Buffer();

    void Update(std::span<const std::byte>, vk::DeviceSize offset = 0);
    void Reserve(vk::DeviceSize);
    template<typename T> void Update(const std::vector<T> &data) { Update(as_bytes(data)); }
    void Insert(std::span<const std::byte>, vk::DeviceSize offset);
    void Erase(vk::DeviceSize offset, vk::DeviceSize size);

    vk::Buffer operator*() const;
    std::span<const std::byte> GetData() const;
    std::span<const std::byte> GetMappedData() const;
    std::span<std::byte> GetMappedData();
    vk::DeviceSize GetAllocatedSize() const;
    void Write(std::span<const std::byte>, vk::DeviceSize offset = 0) const;
    void Move(vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize size) const;
    void Flush(vk::DeviceSize offset, vk::DeviceSize size) const;
    vk::DescriptorBufferInfo GetDescriptor() const { return {operator*(), 0, vk::WholeSize}; }

    BufferContext &Ctx;
    uint32_t Slot{InvalidSlot};
    vk::DeviceSize UsedSize{0};
    vk::BufferUsageFlags Usage{};

    struct VmaBuffer;
    std::unique_ptr<VmaBuffer> DeviceBuffer;
    std::unique_ptr<VmaBuffer> HostBuffer;

private:
    void Retire();
    void UpdateDescriptor();
    bool IsMapped() const;

    using UpdateFn = bool (*)(Buffer &, std::span<const std::byte>, vk::DeviceSize);
    using ReserveFn = bool (*)(Buffer &, vk::DeviceSize);
    using InsertFn = void (*)(Buffer &, std::span<const std::byte>, vk::DeviceSize);
    using EraseFn = void (*)(Buffer &, vk::DeviceSize, vk::DeviceSize);
    struct Impl {
        UpdateFn Update;
        ReserveFn Reserve;
        InsertFn Insert;
        EraseFn Erase;
    };

    SlotType Type;
    Impl ImplOps;
};
} // namespace mvk
