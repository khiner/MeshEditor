#pragma once

#include "RangeAllocator.h"
#include "SlottedRange.h"
#include "metal/Buffer.h"
#include "project/store/History.h"
#include "project/store/Records.h"

template<typename T>
struct BufferArena {
    BufferArena(mtl::BufferContext &ctx, SlotType slot_type) : Buffer(ctx, 0, slot_type) {}
    explicit BufferArena(mtl::BufferContext &ctx) : Buffer(ctx, 0) {}

    void Track(store::History &history, const std::string &name, uint32_t page_bytes = 4096) {
        Buffer.Track(history, name + ".bytes", page_bytes);
        Tracked = std::make_unique<store::Records>(&Allocator, AllocatorCodec, 1);
        Allocator.History = Tracked.get();
        history.Track(*Tracked, name + ".alloc", 0);
    }

    void ReserveAdditional(uint32_t count) {
        if (count == 0) return;
        Buffer.Reserve(Buffer.UsedSize + uint64_t(count) * sizeof(T));
    }

    Range Allocate(uint32_t count) {
        const auto range = Allocator.Allocate(count);
        if (range.Count == 0) return range;

        const uint64_t required_size = (range.Offset + range.Count) * sizeof(T);
        Buffer.SetUsedSize(std::max(Buffer.UsedSize, required_size));
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

    void Shrink(Range &range, uint32_t used) {
        if (used >= range.Count) return;
        const Range tail{range.Offset + used, range.Count - used};
        Allocator.Free(tail);
        // Keep UsedSize equal to the highest allocated byte because reporting and binding use it.
        if (const uint64_t tail_end = uint64_t(tail.Offset + tail.Count) * sizeof(T); Buffer.UsedSize == tail_end) {
            Buffer.SetUsedSize(uint64_t(range.Offset + used) * sizeof(T));
        }
        range.Count = used;
    }

    std::span<const T> Get(Range range) const { return Buffer.GetSpan<T>(range); }
    std::span<T> GetMutable(Range range) { return Buffer.GetMutableSpan<T>(range); }

    Range Clone(Range src) { return src.Count > 0 ? Allocate(Get(src)) : Range{}; }

    SlottedRange Slotted(Range r) const { return {r, Buffer.Slot}; }

    void Reset() {
        Buffer.SetUsedSize(0);
        Allocator.Reset();
    }

    mtl::Buffer Buffer;

private:
    void WriteRange(uint32_t offset, std::span<const T> values) { Buffer.Update(as_bytes(values), offset * sizeof(T)); }

    RangeAllocator Allocator;
    std::unique_ptr<store::Records> Tracked;
};
