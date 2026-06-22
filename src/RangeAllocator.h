#pragma once

#include "Range.h"

#include <algorithm>
#include <limits>
#include <vector>

// Order-independent allocator over a linear index/offset space: a best-fit, coalesced free list plus a high-water mark.
struct RangeAllocator {
    Range Allocate(uint32_t count) {
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

    void Free(Range range) {
        if (range.Count == 0) return;

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

    // Carve a specific range out of the free space (inverse of Free for a known offset). False if any of it is already allocated.
    bool Reserve(Range r) {
        if (r.Count == 0) return true;
        const auto r_end = r.Offset + r.Count;
        if (r.Offset >= EndOffset) {
            // Beyond the high-water: extend, leaving any skipped gap free.
            const auto old_end = EndOffset;
            EndOffset = r_end;
            if (r.Offset > old_end) Free({old_end, r.Offset - old_end});
            return true;
        }
        const auto it = std::ranges::find_if(FreeBlocks, [&](const Range &b) {
            return b.Offset <= r.Offset && r_end <= b.Offset + b.Count;
        });
        if (it == FreeBlocks.end()) return false; // (partially) allocated already
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

    // Serializable allocator state for save/restore.
    struct State {
        std::vector<Range> FreeBlocks;
        uint32_t EndOffset{0};
    };
    State Save() const { return {FreeBlocks, EndOffset}; }
    void Restore(State state) {
        FreeBlocks = std::move(state.FreeBlocks);
        EndOffset = state.EndOffset;
    }

private:
    std::vector<Range> FreeBlocks;
    uint32_t EndOffset{0};
};
