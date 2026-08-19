#pragma once

#include "Range.h"
#include "metal/Bindless.h"

#include <span>
#include <string>
#include <vector>

template<typename T>
constexpr std::span<const std::byte> as_bytes(const std::vector<T> &v) { return std::as_bytes(std::span{v}); }
template<typename T, uint32_t N>
constexpr std::span<const std::byte> as_bytes(const std::array<T, N> &v) { return std::as_bytes(std::span{v}); }
template<typename T>
constexpr std::span<const std::byte> as_bytes(const T &v) { return {reinterpret_cast<const std::byte *>(&v), sizeof(T)}; }

namespace mtl {
// Retired buffers outlive the submit that still references them.
struct BufferContext {
    BufferContext(const Context &ctx, BindlessSet &slots) : Ctx(ctx), Slots(slots) {}

    void ReclaimRetiredBuffers() { Retired.clear(); }

    std::string DebugHeapUsage() const;

    const Context &Ctx;
    BindlessSet &Slots;
    std::vector<Owned<MTL::Buffer>> Retired;
};

// A zero size defers allocation.
Owned<MTL::Buffer> NewBuffer(const Context &, uint64_t size);

struct Buffer {
    Buffer(BufferContext &, uint64_t size, SlotType);
    Buffer(BufferContext &, std::span<const std::byte>, SlotType);
    Buffer(BufferContext &, uint64_t size);
    Buffer(BufferContext &, std::span<const std::byte>);

    Buffer(const Buffer &) = delete;
    Buffer(Buffer &&) noexcept;
    Buffer &operator=(const Buffer &) = delete;
    Buffer &operator=(Buffer &&) noexcept;
    ~Buffer();

    void Update(std::span<const std::byte>, uint64_t offset = 0);
    void Reserve(uint64_t);
    template<typename T> void Update(const std::vector<T> &data) { Update(as_bytes(data)); }

    MTL::Buffer *operator*() const { return *DeviceBuffer; }
    std::span<const std::byte> GetData() const;
    std::span<std::byte> GetMappedData() const;
    uint64_t GetAllocatedSize() const;
    void Write(std::span<const std::byte>, uint64_t offset = 0) const;
    void Move(uint64_t from, uint64_t to, uint64_t size) const;
    std::span<std::byte> GetMutableRange(uint64_t offset, uint64_t size) const;
    template<typename T> std::span<T> SetCount(uint32_t count) {
        const auto size = uint64_t(count) * sizeof(T);
        Reserve(size);
        UsedSize = size;
        if (count == 0) return {};
        return {reinterpret_cast<T *>(GetMutableRange(0, size).data()), count};
    }
    template<typename T> uint32_t Count() const { return uint32_t(UsedSize / sizeof(T)); }
    template<typename T> std::span<const T> GetSpan(Range range) const {
        if (range.Count == 0) return {};
        return {reinterpret_cast<const T *>(GetData().data()) + range.Offset, range.Count};
    }
    template<typename T> std::span<T> GetMutableSpan(Range range) const {
        if (range.Count == 0) return {};
        return {reinterpret_cast<T *>(GetMutableRange(uint64_t(range.Offset) * sizeof(T), uint64_t(range.Count) * sizeof(T)).data()), range.Count};
    }

    BufferContext &Ctx;
    uint32_t Slot{InvalidSlot};
    uint64_t UsedSize{0};
    Owned<MTL::Buffer> DeviceBuffer;

    void UpdateSlot();

private:
    void Retire();

    SlotType Type{};
};
} // namespace mtl

template<typename T>
struct TypedBuffer {
    TypedBuffer(mtl::BufferContext &ctx, uint64_t bytes, SlotType slot) : Buffer(ctx, bytes, slot) {}
    TypedBuffer(mtl::BufferContext &ctx, uint64_t bytes) : Buffer(ctx, bytes) {}

    MTL::Buffer *operator*() const { return *Buffer; }
    uint32_t Slot() const { return Buffer.Slot; }

    uint32_t Count() const { return uint32_t(Buffer.UsedSize / sizeof(T)); }
    void SetCount(uint32_t n) {
        const auto s = uint64_t(n) * sizeof(T);
        Buffer.Reserve(s);
        Buffer.UsedSize = s;
    }
    void Reserve(uint32_t n) { Buffer.Reserve(uint64_t(n) * sizeof(T)); }

    T *Data() { return reinterpret_cast<T *>(Buffer.GetMappedData().data()); }
    const T *Data() const { return reinterpret_cast<const T *>(Buffer.GetData().data()); }

    T &Get(uint32_t i) { return Data()[i]; }
    const T &Get(uint32_t i) const { return Data()[i]; }

    void Set(uint32_t i, const T &v) { Buffer.Update(as_bytes(v), uint64_t(i) * sizeof(T)); }
    uint32_t Append(const T &v) {
        const auto i = Count();
        Set(i, v);
        return i;
    }

    mtl::Buffer Buffer;
};
