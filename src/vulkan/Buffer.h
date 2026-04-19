#pragma once

#include "Slots.h"

#include <vulkan/vulkan.hpp>

#include <array>
#include <map>
#include <memory>
#include <span>
#include <unordered_map>
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
enum class MemoryUsage {
    Unknown,
    CpuOnly,
    CpuToGpu,
    GpuOnly,
    GpuToCpu
};

#ifdef MVK_FORCE_STAGED_TRANSFERS
struct BufferPair {
    vk::Buffer Src, Dst;
    bool operator==(const BufferPair &) const = default;
};
struct BufferPairHash {
    size_t operator()(const BufferPair &p) const noexcept {
        return std::hash<void *>{}(static_cast<VkBuffer>(p.Src)) ^
            (std::hash<void *>{}(static_cast<VkBuffer>(p.Dst)) << 1);
    }
};
// Maps start offset -> end offset, automatically sorted and merged on insert
using CopyRanges = std::map<vk::DeviceSize, vk::DeviceSize>;
#endif

struct VmaBuffer;
struct BufferContext {
    BufferContext(vk::PhysicalDevice, vk::Device, vk::Instance, DescriptorSlots &);
    ~BufferContext();

    void ReclaimRetiredBuffers();

    std::vector<vk::WriteDescriptorSet> GetDeferredDescriptorUpdates();
    void DeferDescriptorUpdate(TypedSlot slot, const vk::DescriptorBufferInfo &info) { DeferredDescriptorUpdates.insert_or_assign(slot, info); }
    void CancelDeferredDescriptorUpdate(TypedSlot slot) { DeferredDescriptorUpdates.erase(slot); }
    void ClearDeferredDescriptorUpdates() { DeferredDescriptorUpdates.clear(); }

#ifdef MVK_FORCE_STAGED_TRANSFERS
    void DeferCopy(vk::Buffer src, vk::Buffer dst, vk::DeviceSize offset, vk::DeviceSize size);
    void CancelDeferredCopies(vk::Buffer src, vk::Buffer dst);
    void RecordDeferredCopies(vk::CommandBuffer cb);
    auto TakeDeferredCopies() { return std::exchange(DeferredBufferCopies, {}); }
#endif

    std::string DebugHeapUsage() const;

    vk::PhysicalDevice PhysicalDevice;
    VmaAllocator Vma;
    std::vector<std::unique_ptr<VmaBuffer>> Retired;
    DescriptorSlots &Slots;

private:
    struct TypedSlotHash {
        size_t operator()(const TypedSlot &key) const noexcept {
            return std::hash<uint64_t>{}((static_cast<uint64_t>(std::to_underlying(key.Type)) << 32) | static_cast<uint64_t>(key.Slot));
        }
    };
    std::unordered_map<TypedSlot, vk::DescriptorBufferInfo, TypedSlotHash> DeferredDescriptorUpdates;
#ifdef MVK_FORCE_STAGED_TRANSFERS
    std::unordered_map<BufferPair, CopyRanges, BufferPairHash> DeferredBufferCopies;
#endif
};

// Vulkan buffer with descriptor slot management.
// By default, assumes direct-mapped memory (unified memory architectures like Apple Silicon).
// Define MVK_FORCE_STAGED_TRANSFERS to use explicit staging buffers (for discrete GPUs or testing).
struct Buffer {
    // Slotted buffer (GpuOnly, with descriptor slot management)
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
    std::span<std::byte> GetMutableRange(vk::DeviceSize offset, vk::DeviceSize size);
    vk::DescriptorBufferInfo GetDescriptor() const { return {operator*(), 0, vk::WholeSize}; }

    BufferContext &Ctx;
    uint32_t Slot{InvalidSlot};
    vk::DeviceSize UsedSize{0};
    vk::BufferUsageFlags Usage{};
    MemoryUsage Memory{MemoryUsage::Unknown};
    std::unique_ptr<VmaBuffer> DeviceBuffer;
#ifdef MVK_FORCE_STAGED_TRANSFERS
    std::unique_ptr<VmaBuffer> HostBuffer;
#endif

private:
    void Retire();
    void UpdateDescriptor();

    SlotType Type;
};
} // namespace mvk

// Typed wrapper over mvk::Buffer for fixed-element-count buffers (no arena/free-list).
template<typename T>
struct TypedBuffer {
    TypedBuffer(mvk::BufferContext &ctx, vk::DeviceSize bytes, vk::BufferUsageFlags usage, SlotType slot)
        : Buffer(ctx, bytes, usage, slot) {}
    TypedBuffer(mvk::BufferContext &ctx, vk::DeviceSize bytes, mvk::MemoryUsage mem, vk::BufferUsageFlags usage = {})
        : Buffer(ctx, bytes, mem, usage) {}

    vk::Buffer operator*() const { return *Buffer; }
    uint32_t Slot() const { return Buffer.Slot; }

    uint32_t Count() const { return uint32_t(Buffer.UsedSize / sizeof(T)); }
    void SetCount(uint32_t n) {
        auto s = vk::DeviceSize(n) * sizeof(T);
        Buffer.Reserve(s);
        Buffer.UsedSize = s;
    }
    void Reserve(uint32_t n) { Buffer.Reserve(vk::DeviceSize(n) * sizeof(T)); }

    T *Data() { return reinterpret_cast<T *>(Buffer.GetMappedData().data()); }
    const T *Data() const { return reinterpret_cast<const T *>(Buffer.GetData().data()); }

    T &Get(uint32_t i) { return Data()[i]; }
    const T &Get(uint32_t i) const { return Data()[i]; }

    void Set(uint32_t i, const T &v) { Buffer.Update(as_bytes(v), vk::DeviceSize(i) * sizeof(T)); }
    uint32_t Append(const T &v) {
        auto i = Count();
        Set(i, v);
        return i;
    }

    vk::DescriptorBufferInfo GetDescriptor() const { return Buffer.GetDescriptor(); }
    vk::DescriptorBufferInfo GetDescriptor(uint32_t count) const { return {*Buffer, 0, vk::DeviceSize(count) * sizeof(T)}; }

    mvk::Buffer Buffer;
};
