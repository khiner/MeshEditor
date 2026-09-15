#pragma once

#include "Range.h"
#include "project/store/Records.h"

#include <algorithm>
#include <limits>
#include <vector>

// Order-independent allocator over a linear index/offset space: a best-fit, coalesced free list plus a high-water mark.
// History tracks the free list and high-water mark as one serialized record through AllocatorCodec.
struct RangeAllocator {
    store::Records *History{};

    Range Allocate(uint32_t count) {
        if (count == 0) return {};
        if (History) History->Write(0, 1);

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

    void Free(Range range) {
        if (range.Count == 0) return;
        if (History) History->Write(0, 1);

        auto it = std::ranges::lower_bound(FreeBlocks, range.Offset, {}, &Range::Offset);
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
        const auto it = std::ranges::find_if(FreeBlocks, [&](const Range &b) {
            return b.Offset <= r.Offset && r_end <= b.Offset + b.Count;
        });
        if (it == FreeBlocks.end()) return false;
        const Range left{it->Offset, r.Offset - it->Offset}, right{r_end, it->Offset + it->Count - r_end};
        if (left.Count && right.Count) {
            *it = left;
            FreeBlocks.insert(it + 1, right);
        } else if (left.Count) {
            *it = left;
        } else if (right.Count) {
            *it = right;
        } else {
            FreeBlocks.erase(it);
        }
        return true;
    }

    uint32_t HighWaterMark() const { return EndOffset; }

    void Reset() {
        if (History) History->Write(0, 1);
        FreeBlocks.clear();
        EndOffset = 0;
    }

    std::vector<Range> FreeBlocks;
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
