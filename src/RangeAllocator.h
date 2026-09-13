#pragma once

#include "Range.h"
#include "project/store/LiveTrie.h"

#include <algorithm>
#include <limits>
#include <vector>

// Order-independent allocator over a linear index/offset space: a best-fit, coalesced free list plus a high-water mark.
struct RangeAllocator {
    store::LiveTrie *FreeHistory{}, *EndHistory{};

    Range Allocate(uint32_t count) {
        if (count == 0) return {};

        auto it = std::ranges::min_element(FreeBlocks, {}, [count](const auto &b) {
            return b.Count >= count ? b.Count : std::numeric_limits<uint32_t>::max();
        });
        if (it != FreeBlocks.end() && it->Count >= count) {
            if (FreeHistory) FreeHistory->Write(size_t(it - FreeBlocks.begin()), it->Count == count ? size_t(FreeBlocks.end() - it) : 1);
            uint32_t offset = it->Offset;
            if (it->Count == count) FreeBlocks.erase(it);
            else *it = {it->Offset + count, it->Count - count};
            return {offset, count};
        }
        if (EndHistory) EndHistory->Write(0, 1);
        return {std::exchange(EndOffset, EndOffset + count), count};
    }

    void Free(Range range) {
        if (range.Count == 0) return;

        auto it = std::ranges::lower_bound(FreeBlocks, range.Offset, {}, &Range::Offset);
        if (FreeHistory) {
            const auto first = it == FreeBlocks.begin() ? 0 : size_t(it - FreeBlocks.begin() - 1);
            FreeHistory->Write(first, FreeBlocks.size() - first + 1);
        }
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
        const auto r_end = r.Offset + r.Count;
        if (r.Offset >= EndOffset) {
            if (EndHistory) EndHistory->Write(0, 1);
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
        if (FreeHistory) FreeHistory->Write(size_t(it - FreeBlocks.begin()), size_t(FreeBlocks.end() - it) + 1);
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
        if (FreeHistory) FreeHistory->Write(0, std::max(FreeBlocks.size(), state.FreeBlocks.size()));
        if (EndHistory) EndHistory->Write(0, 1);
        FreeBlocks = std::move(state.FreeBlocks);
        EndOffset = state.EndOffset;
    }
    void Reset() { Restore({}); }

    std::vector<Range> FreeBlocks;
    uint32_t EndOffset{0};
};
