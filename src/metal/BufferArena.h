#pragma once

#include "RangeAllocator.h"
#include "SlottedRange.h"
#include "metal/Buffer.h"
#include "project/store/History.h"
#include "project/store/Records.h"

#include <utility>

template<typename T>
struct BufferArena {
    BufferArena(mtl::BufferContext &ctx, SlotType slot_type, mtl::BufferLifetime lifetime = mtl::BufferLifetime::Canonical) : Buffer(ctx, 0, slot_type, lifetime) {}
    explicit BufferArena(mtl::BufferContext &ctx) : Buffer(ctx, 0) {}

    void Track(store::History &history, const std::string &name) {
        Buffer.Track(history, name + ".bytes");
        TrackAllocator(history, name);
    }
    // Runtime binding records restore allocation identity and rebuild their small descriptor from canonical addresses.
    void TrackAllocator(store::History &history, const std::string &name) {
        Tracked = std::make_unique<store::Records>(&Allocator, AllocatorCodec, RangeAllocator::HistoryLevels);
        Allocator.History = Tracked.get();
        history.Track(*Tracked, name + ".alloc", 0);
    }

    void ReserveAdditional(uint64_t count) {
        if (count == 0) return;
        Buffer.Reserve(Buffer.UsedSize + count * sizeof(T));
    }
    // Accumulate a coming allocation so one CommitPlanned grows the buffer once for a batch.
    void PlanAdditional(uint32_t count) { Planned += count; }
    void CommitPlanned() { ReserveAdditional(std::exchange(Planned, 0u)); }

    // Size the buffer to cover `range`, which a master arena this one mirrors element for element allocated.
    void Mirror(Range range) {
        if (range.Count == 0) return;
        Buffer.SetUsedSize(std::max(Buffer.UsedSize, uint64_t(range.Offset + range.Count) * sizeof(T)));
    }

    Range Allocate(uint32_t count) {
        auto transaction = BeginAllocation();
        const auto range = Allocator.Allocate(count);
        if (range.Count == 0) return range;

        const uint64_t required_size = (uint64_t(range.Offset) + range.Count) * sizeof(T);
        Buffer.SetUsedSize(std::max(Buffer.UsedSize, required_size));
        transaction.Commit();
        return range;
    }

    Range Allocate(std::span<const T> values) {
        auto transaction = BeginAllocation();
        const auto range = Allocate(values.size());
        WriteRange(range.Offset, values);
        transaction.Commit();
        return range;
    }

    void Update(Range &range, std::span<const T> values) {
        if (values.size() == range.Count) {
            WriteRange(range.Offset, values);
            return;
        }
        auto transaction = BeginAllocation();
        auto new_range = Allocator.Allocate(values.size());
        WriteRange(new_range.Offset, values);
        Allocator.Free(range);
        range = new_range;
        transaction.Commit();
    }

    // Backing capacity may remain enlarged after cancellation.
    // No range stays allocated.
    // The journal contains allocator metadata only.
    auto BeginAllocation() { return RangeAllocator::Transaction{Allocator}; }

    void Release(Range range) { Allocator.Free(range); }
    // Releases every range, freeing each run of adjacent ranges at once.
    void Release(std::vector<Range> ranges) {
        Allocator.Free(std::move(ranges));
    }
    uint32_t HighWaterMark() const { return Allocator.HighWaterMark(); }

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
    // Captures the range's pages before a GPU write, which a new range needs where it reuses pages an older edit view pins.
    void CaptureWrite(Range range) const {
        if (range.Count) Buffer.CaptureWrite(uint64_t(range.Offset) * sizeof(T), uint64_t(range.Count) * sizeof(T));
    }

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
    uint32_t Planned{0};
};
