#pragma once

#include "Range.h"
#include "project/store/Records.h"

#include <algorithm>
#include <limits>
#include <map>

// Order-independent allocator over a linear index/offset space: a best-fit, coalesced free list plus a high-water mark.
// History tracks the free list and high-water mark as one serialized record through AllocatorCodec.
struct RangeAllocator {
    store::Records *History{};

    Range Allocate(uint32_t count) {
        if (count == 0) return {};
        if (History) History->Write(0, 1);

        auto it = std::ranges::min_element(FreeBlocks, {}, [count](const auto &b) {
            return b.second >= count ? b.second : std::numeric_limits<uint32_t>::max();
        });
        if (it != FreeBlocks.end() && it->second >= count) {
            const auto [offset, block_count] = *it;
            FreeBlocks.erase(it);
            if (block_count > count) FreeBlocks.emplace(offset + count, block_count - count);
            return {offset, count};
        }
        return {std::exchange(EndOffset, EndOffset + count), count};
    }

    void Free(Range range) {
        if (range.Count == 0) return;
        if (History) History->Write(0, 1);

        auto it = FreeBlocks.lower_bound(range.Offset);
        auto start = range.Offset, end = start + range.Count;
        if (it != FreeBlocks.begin()) {
            if (auto prev = std::prev(it); prev->first + prev->second == start) {
                start = prev->first;
                it = FreeBlocks.erase(prev);
            }
        }
        if (it != FreeBlocks.end() && end == it->first) {
            end = it->first + it->second;
            it = FreeBlocks.erase(it);
        }
        FreeBlocks.emplace_hint(it, start, end - start);
    }

    // Reserve a specific free range and return false if any part is allocated.
    bool Reserve(Range r) {
        if (r.Count == 0) return true;
        if (History) History->Write(0, 1);
        const auto r_end = r.Offset + r.Count;
        if (r.Offset >= EndOffset) {
            // Extend the high-water mark and add any skipped indices to the free list.
            const auto old_end = EndOffset;
            EndOffset = r_end;
            if (r.Offset > old_end) Free({old_end, r.Offset - old_end});
            return true;
        }
        auto it = FreeBlocks.upper_bound(r.Offset);
        if (it == FreeBlocks.begin()) return false;
        --it;
        if (r_end > it->first + it->second) return false;
        const Range left{it->first, r.Offset - it->first}, right{r_end, it->first + it->second - r_end};
        FreeBlocks.erase(it);
        if (left.Count) FreeBlocks.emplace(left.Offset, left.Count);
        if (right.Count) FreeBlocks.emplace(right.Offset, right.Count);
        return true;
    }

    uint32_t HighWaterMark() const { return EndOffset; }

    void Reset() {
        if (History) History->Write(0, 1);
        FreeBlocks.clear();
        EndOffset = 0;
    }

    // Free blocks by offset, each holding its count.
    std::map<uint32_t, uint32_t> FreeBlocks;
    uint32_t EndOffset{0};
};

// The one-record codec history tracks an allocator with.
inline constexpr store::Records::Codec AllocatorCodec{
    [](const void *) { return uint64_t{1}; },
    [](void *, uint64_t) {},
    [](const void *v, uint64_t, std::vector<std::byte> &out) {
        const auto &a = *static_cast<const RangeAllocator *>(v);
        zpp::bits::out archive{out};
        archive(a.FreeBlocks, a.EndOffset).or_throw();
        out.resize(archive.position());
    },
    [](void *v, uint64_t, std::span<const std::byte> bytes) {
        auto &a = *static_cast<RangeAllocator *>(v);
        zpp::bits::in{bytes}(a.FreeBlocks, a.EndOffset).or_throw();
    },
    [](void *v, uint64_t) {
        auto &a = *static_cast<RangeAllocator *>(v);
        a.FreeBlocks.clear();
        a.EndOffset = 0;
    },
};
