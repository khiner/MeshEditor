#pragma once

#include "Buffer.h"

#include <algorithm>
#include <limits>
#include <span>
#include <vector>

struct BufferRange {
    uint32_t Offset{0}, Count{0};
};

struct RangeAllocator {
    BufferRange Allocate(uint32_t count) {
        if (count == 0) return {};

        auto it = std::ranges::min_element(
            FreeBlocks,
            {},
            [count](const BufferRange &block) {
                return block.Count < count ? std::numeric_limits<uint32_t>::max() : block.Count;
            }
        );
        if (it != FreeBlocks.end() && it->Count >= count) {
            auto &block = *it;
            const uint32_t offset = block.Offset;
            if (block.Count == count) {
                FreeBlocks.erase(it);
            } else {
                block.Offset += count;
                block.Count -= count;
            }
            return {offset, count};
        }

        const uint32_t offset = EndOffset;
        EndOffset += count;
        return {offset, count};
    }

    void Free(BufferRange range) {
        if (range.Count == 0) return;
        auto it = std::lower_bound(
            FreeBlocks.begin(),
            FreeBlocks.end(),
            range.Offset,
            [](const BufferRange &block, uint32_t offset) { return block.Offset < offset; }
        );
        uint32_t start = range.Offset;
        uint32_t end = range.Offset + range.Count;
        if (it != FreeBlocks.begin()) {
            auto &prev = *(it - 1);
            if (prev.Offset + prev.Count == start) {
                start = prev.Offset;
                it = FreeBlocks.erase(it - 1);
            }
        }
        while (it != FreeBlocks.end() && end == it->Offset) {
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
struct Megabuffer {
    explicit Megabuffer(mvk::BufferContext &ctx, vk::BufferUsageFlags usage, SlotType slot_type)
        : Buffer(ctx, 1, usage, slot_type) {}

    explicit Megabuffer(mvk::BufferContext &ctx, mvk::MemoryUsage mem, vk::BufferUsageFlags usage = {})
        : Buffer(ctx, 1, mem, usage) {}

    BufferRange Allocate(std::span<const T> values) {
        const auto range = Allocator.Allocate(static_cast<uint32_t>(values.size()));
        WriteRange(range.Offset, values);
        return range;
    }

    void Update(BufferRange &range, std::span<const T> values) {
        if (values.size() == range.Count) {
            WriteRange(range.Offset, values);
            return;
        }
        auto new_range = Allocator.Allocate(static_cast<uint32_t>(values.size()));
        WriteRange(new_range.Offset, values);
        Allocator.Free(range);
        range = new_range;
    }

    void Release(BufferRange range) { Allocator.Free(range); }

    std::span<const T> Get(BufferRange range) const { return SpanFromBytes<const T>(range); }
    std::span<T> GetMutable(BufferRange range) { return SpanFromBytes<T>(range); }

    mvk::Buffer Buffer;

private:
    RangeAllocator Allocator;

    static vk::DeviceSize ByteOffset(uint32_t offset) {
        return static_cast<vk::DeviceSize>(offset) * sizeof(T);
    }

    void WriteRange(uint32_t offset, std::span<const T> values) {
        if (values.empty()) return;
        Buffer.Update(as_bytes(values), ByteOffset(offset));
    }

    static auto RangeBytes(auto bytes, BufferRange range) {
        const auto start = ByteOffset(range.Offset);
        const auto count_bytes = static_cast<vk::DeviceSize>(range.Count) * sizeof(T);
        if (start + count_bytes > bytes.size()) return bytes.subspan(0, 0);
        return bytes.subspan(start, count_bytes);
    }

    template<typename U>
    std::span<U> SpanFromBytes(BufferRange range) {
        if (auto bytes = RangeBytes(Buffer.GetMappedData(), range); !bytes.empty()) {
            return {reinterpret_cast<U *>(bytes.data()), range.Count};
        }
        return {};
    }

    template<typename U>
    std::span<const U> SpanFromBytes(BufferRange range) const {
        if (const auto bytes = RangeBytes(Buffer.GetMappedData(), range); !bytes.empty()) {
            return {reinterpret_cast<const U *>(bytes.data()), range.Count};
        }
        return {};
    }
};
