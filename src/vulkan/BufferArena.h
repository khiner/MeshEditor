#pragma once

#include "Buffer.h"

#include <algorithm>
#include <limits>
#include <ranges>
#include <span>
#include <vector>

struct BufferRange {
    uint32_t Offset{0}, Count{0};
};

struct SlottedBufferRange {
    BufferRange Range;
    uint32_t Slot{InvalidSlot};
};

struct RangeAllocator {
    BufferRange Allocate(uint32_t count) {
        if (count == 0) return {};

        auto it = std::ranges::min_element(FreeBlocks, {}, [count](const auto &b) {
            return b.Count >= count ? b.Count : std::numeric_limits<uint32_t>::max();
        });
        if (it != FreeBlocks.end() && it->Count >= count) {
            uint32_t offset = it->Offset;
            if (it->Count == count) FreeBlocks.erase(it);
            else *it = {it->Offset + count, it->Count - count};
            return {offset, count};
        }
        return {std::exchange(EndOffset, EndOffset + count), count};
    }

    void Free(BufferRange range) {
        if (range.Count == 0) return;

        auto it = std::ranges::lower_bound(FreeBlocks, range.Offset, {}, &BufferRange::Offset);
        auto start = range.Offset, end = start + range.Count;
        if (it != FreeBlocks.begin()) {
            if (auto prev = std::prev(it); prev->Offset + prev->Count == start) {
                start = prev->Offset;
                it = FreeBlocks.erase(prev);
            }
        }
        if (it != FreeBlocks.end() && end == it->Offset) {
            end = it->Offset + it->Count;
            it = FreeBlocks.erase(it);
        }
        FreeBlocks.insert(it, {start, end - start});
    }

private:
    std::vector<BufferRange> FreeBlocks;
    uint32_t EndOffset{0};
};

template<typename T>
struct BufferArena {
    BufferArena(mvk::BufferContext &ctx, vk::BufferUsageFlags usage, SlotType slot_type)
        : Buffer(ctx, 1, usage, slot_type) {}
    BufferArena(mvk::BufferContext &ctx, mvk::MemoryUsage mem, vk::BufferUsageFlags usage = {})
        : Buffer(ctx, 1, mem, usage) {}

    BufferRange Allocate(uint32_t count) {
        const auto range = Allocator.Allocate(count);
        if (range.Count == 0) return range;

        const vk::DeviceSize required_size = (range.Offset + range.Count) * sizeof(T);
        Buffer.Reserve(required_size);
        Buffer.UsedSize = std::max(Buffer.UsedSize, required_size);
        return range;
    }

    BufferRange Allocate(std::span<const T> values) {
        const auto range = Allocate(values.size());
        WriteRange(range.Offset, values);
        return range;
    }

    void Update(BufferRange &range, std::span<const T> values) {
        if (values.size() == range.Count) {
            WriteRange(range.Offset, values);
            return;
        }
        auto new_range = Allocator.Allocate(values.size());
        WriteRange(new_range.Offset, values);
        Allocator.Free(range);
        range = new_range;
    }

    void Release(BufferRange range) { Allocator.Free(range); }

    std::span<const T> Get(BufferRange range) const { return SpanFromBytes<const T>(range); }
    std::span<T> GetMutable(BufferRange range) {
        auto bytes = Buffer.GetMutableRange(range.Offset * sizeof(T), range.Count * sizeof(T));
        return {reinterpret_cast<T *>(bytes.data()), range.Count};
    }

    mvk::Buffer Buffer;

private:
    void WriteRange(uint32_t offset, std::span<const T> values) { Buffer.Update(as_bytes(values), offset * sizeof(T)); }

    static auto RangeBytes(auto bytes, BufferRange range) {
        const auto start = range.Offset * sizeof(T);
        const auto count_bytes = range.Count * sizeof(T);
        return start + count_bytes > bytes.size() ? bytes.subspan(0, 0) : bytes.subspan(start, count_bytes);
    }

    template<typename U>
    std::span<const U> SpanFromBytes(BufferRange range) const {
        if (const auto bytes = RangeBytes(Buffer.GetMappedData(), range); !bytes.empty()) {
            return {reinterpret_cast<const U *>(bytes.data()), range.Count};
        }
        return {};
    }

    RangeAllocator Allocator;
};
