#pragma once

#include "Buffer.h"
#include "RangeAllocator.h"
#include "SlottedRange.h"

struct ArenaState {
    std::vector<std::byte> Bytes;
    RangeAllocator::State Allocator;
};

template<typename T>
struct BufferArena {
    BufferArena(mvk::BufferContext &ctx, vk::BufferUsageFlags usage, SlotType slot_type)
        : Buffer(ctx, 0, usage, slot_type) {}
    BufferArena(mvk::BufferContext &ctx, mvk::MemoryUsage mem, vk::BufferUsageFlags usage = {})
        : Buffer(ctx, 0, mem, usage) {}

    void ReserveAdditional(uint32_t count) {
        if (count == 0) return;
        Buffer.Reserve(Buffer.UsedSize + vk::DeviceSize(count) * sizeof(T));
    }

    Range Allocate(uint32_t count) {
        const auto range = Allocator.Allocate(count);
        if (range.Count == 0) return range;

        const vk::DeviceSize required_size = (range.Offset + range.Count) * sizeof(T);
        Buffer.Reserve(required_size);
        Buffer.UsedSize = std::max(Buffer.UsedSize, required_size);
        return range;
    }

    Range Allocate(std::span<const T> values) {
        const auto range = Allocate(values.size());
        WriteRange(range.Offset, values);
        return range;
    }

    void Update(Range &range, std::span<const T> values) {
        if (values.size() == range.Count) {
            WriteRange(range.Offset, values);
            return;
        }
        auto new_range = Allocator.Allocate(values.size());
        WriteRange(new_range.Offset, values);
        Allocator.Free(range);
        range = new_range;
    }

    void Release(Range range) { Allocator.Free(range); }

    std::span<const T> Get(Range range) const { return SpanFromBytes<const T>(range); }
    std::span<T> GetMutable(Range range) {
        auto bytes = Buffer.GetMutableRange(range.Offset * sizeof(T), range.Count * sizeof(T));
        return {reinterpret_cast<T *>(bytes.data()), range.Count};
    }

    // Deep-copy an existing range into a new allocation.
    Range Clone(Range src) { return src.Count > 0 ? Allocate(Get(src)) : Range{}; }

    SlottedRange Slotted(Range r) const { return {r, Buffer.Slot}; }

    // Reset to empty: used size and allocator go to zero, keeping the GPU allocation for reuse.
    void Reset() {
        Buffer.UsedSize = 0;
        Allocator = {};
    }

    // Capture/restore the whole arena (see ArenaState).
    ArenaState Save() const {
        const auto used = std::size_t(Buffer.UsedSize);
        const auto mapped = Buffer.GetMappedData();
        const auto count = std::min(used, mapped.size());
        return {{mapped.begin(), mapped.begin() + count}, Allocator.Save()};
    }
    void Restore(ArenaState state) {
        Buffer.Reserve(state.Bytes.size());
        Buffer.Update(state.Bytes, 0);
        Buffer.UsedSize = state.Bytes.size();
        Allocator.Restore(std::move(state.Allocator));
    }

    mvk::Buffer Buffer;

private:
    void WriteRange(uint32_t offset, std::span<const T> values) { Buffer.Update(as_bytes(values), offset * sizeof(T)); }

    static auto RangeBytes(auto bytes, Range range) {
        const auto start = range.Offset * sizeof(T);
        const auto count_bytes = range.Count * sizeof(T);
        return start + count_bytes > bytes.size() ? bytes.subspan(0, 0) : bytes.subspan(start, count_bytes);
    }

    template<typename U>
    std::span<const U> SpanFromBytes(Range range) const {
        if (const auto bytes = RangeBytes(Buffer.GetMappedData(), range); !bytes.empty()) {
            return {reinterpret_cast<const U *>(bytes.data()), range.Count};
        }
        return {};
    }

    RangeAllocator Allocator;
};
