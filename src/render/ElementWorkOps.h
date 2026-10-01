#pragma once

#include "Range.h"
#include "gpu/ElementWork.h"
#include "metal/BufferArena.h"

#include <bit>
#include <cassert>
#include <stdexcept>

inline Range WorkStorageRange(ElementWork work) { return {work.Storage.Offset, WorkHeaderWords + work.Capacity * (WorkBlockWords + 2u)}; }
inline uint64_t WorkArgsOffset(ElementWork work, bool groups = false) { return uint64_t(work.Storage.Offset + (groups ? 5u : 2u)) * sizeof(uint32_t); }
inline uint32_t WorkDomainBlocks(uint32_t count) { return uint32_t((uint64_t(count) + 255u) / 256u); }
inline uint32_t WorkCapacity(uint32_t count, uint64_t blocks) {
    return uint32_t(std::bit_ceil(std::max(uint64_t{16}, 2u * std::min(uint64_t(WorkDomainBlocks(count)), blocks))));
}
// The storage words AllocateElementWork takes for `count` elements in at most `blocks` blocks.
inline uint32_t ElementWorkWords(uint32_t count, uint64_t blocks) { return WorkHeaderWords + WorkCapacity(count, blocks) * (WorkBlockWords + 2u); }
inline void FinishElementWork(std::span<uint32_t> data, uint32_t capacity) {
    auto slots = data.subspan(WorkHeaderWords + capacity * WorkBlockWords, data[0]);
    const auto key = [&](uint32_t slot) { return data[WorkHeaderWords + slot * WorkBlockWords]; };
    if (!std::ranges::is_sorted(slots, {}, key)) std::ranges::sort(slots, {}, key);
    uint32_t elements = 0;
    for (uint32_t i = 0; i < data[0]; ++i) {
        const auto slot = data[WorkHeaderWords + capacity * WorkBlockWords + i];
        for (uint32_t word = 1; word <= 8u; ++word) elements += std::popcount(data[WorkHeaderWords + slot * WorkBlockWords + word]);
        data[WorkHeaderWords + capacity * (WorkBlockWords + 1u) + slot] = elements;
    }
    data[2] = data[0];
    data[3] = data[4] = data[6] = data[7] = 1u;
    data[5] = elements;
}
inline ElementWork AllocateElementWork(BufferArena<uint32_t> &arena, uint32_t count, uint64_t blocks = 0) {
    assert(arena.Buffer.Slot != InvalidSlot);
    const auto capacity = WorkCapacity(count, blocks);
    const auto range = arena.Allocate(ElementWorkWords(count, blocks));
    auto data = arena.GetMutable(range);
    std::ranges::fill(data, 0u);
    FinishElementWork(data, capacity);
    return {{arena.Buffer.Slot, range.Offset}, count, capacity};
}
inline uint32_t WorkBlockCount(const BufferArena<uint32_t> &arena, ElementWork work) { return arena.Get({work.Storage.Offset, 1})[0]; }
inline void CheckElementWork(const BufferArena<uint32_t> &arena, ElementWork work) {
    if (arena.Get({work.Storage.Offset + 1u, 1})[0]) throw std::runtime_error("Sparse element work overflowed its prepared capacity.");
}
// The element count of finished work whose producing submit has completed, or of an implicit range.
inline uint32_t ElementWorkCount(const BufferArena<uint32_t> &arena, ElementWork work) {
    if (work.Storage.Slot == InvalidSlot) return work.Count;
    CheckElementWork(arena, work);
    return arena.Get({work.Storage.Offset + 5u, 1})[0];
}
inline uint32_t ElementWorkRank(std::span<const uint32_t> data, ElementWork work, uint32_t element) {
    if (element >= work.Count) return InvalidOffset;
    if (work.Storage.Slot == InvalidSlot) return element;
    const auto key=element/256u+1u;
    auto slot=WorkHash(key-1u,work.Capacity);
    for (uint32_t probe=0u;probe<work.Capacity;++probe,slot=(slot+1u)&(work.Capacity-1u)) {
        const auto block=data.subspan(WorkHeaderWords+slot*WorkBlockWords,WorkBlockWords);
        if (!block[0]) return InvalidOffset;
        if (block[0]!=key) continue;
        const auto word=(element%256u)/32u,bit=element%32u;
        if (!(block[word+1u]&(1u<<bit))) return InvalidOffset;
        uint32_t rank=data[WorkHeaderWords+work.Capacity*(WorkBlockWords+1u)+slot];
        for (uint32_t w=word+1u;w<8u;++w) rank-=std::popcount(block[w+1u]);
        return rank-std::popcount(block[word+1u]&(~0u<<bit));
    }
    return InvalidOffset;
}
// Grows a table's hash capacity to hold `blocks` occupied blocks, moving each occupied block's key and masks into the new table.
inline void ReserveElementWork(BufferArena<uint32_t> &arena, ElementWork &work, uint64_t blocks) {
    if (WorkCapacity(work.Count, blocks) <= work.Capacity) return;
    auto allocation = arena.BeginAllocation();
    const auto replacement = AllocateElementWork(arena, work.Count, blocks);
    const auto source = arena.Get(WorkStorageRange(work));
    auto target = arena.GetMutable(WorkStorageRange(replacement));
    for (uint32_t i = 0; i < source[0]; ++i) {
        const auto block = source.subspan(WorkHeaderWords + source[WorkHeaderWords + work.Capacity * WorkBlockWords + i] * WorkBlockWords, WorkBlockWords);
        auto slot = WorkHash(block[0] - 1u, replacement.Capacity);
        while (target[WorkHeaderWords + slot * WorkBlockWords]) slot = (slot + 1u) & (replacement.Capacity - 1u);
        std::ranges::copy(block, target.begin() + WorkHeaderWords + slot * WorkBlockWords);
        target[WorkHeaderWords + replacement.Capacity * WorkBlockWords + target[0]++] = slot;
    }
    FinishElementWork(target, replacement.Capacity);
    arena.Release(WorkStorageRange(work));
    allocation.Commit();
    work = replacement;
}
inline void ClearElementWork(BufferArena<uint32_t> &arena, ElementWork work) {
    auto data = arena.GetMutable(WorkStorageRange(work));
    for (uint32_t i = 0; i < data[0]; ++i) {
        const auto slot = data[WorkHeaderWords + work.Capacity * WorkBlockWords + i];
        std::ranges::fill(data.subspan(WorkHeaderWords + slot * WorkBlockWords, WorkBlockWords), 0u);
    }
    data[0] = data[1] = 0u;
    FinishElementWork(data, work.Capacity);
}
inline void MarkElementWorkWord(std::span<uint32_t> data, ElementWork work, uint32_t word, uint32_t mask) {
    if (!mask) return;
    const auto key = word / 8u + 1u;
    auto slot = WorkHash(key - 1u, work.Capacity);
    for (uint32_t probe = 0; probe < work.Capacity; ++probe, slot = (slot + 1u) & (work.Capacity - 1u)) {
        auto block = data.subspan(WorkHeaderWords + slot * WorkBlockWords, WorkBlockWords);
        if (!block[0]) {
            block[0] = key;
            data[WorkHeaderWords + work.Capacity * WorkBlockWords + data[0]++] = slot;
        }
        if (block[0] != key) continue;
        block[1u + word % 8u] |= mask;
        return;
    }
    throw std::length_error("Sparse element work exceeds its reserved block count.");
}

inline ElementWork SeedElementWorkHandles(BufferArena<uint32_t> &arena, uint32_t domain_count,
                                         std::span<const uint32_t> handles, uint64_t block_bound = UINT64_MAX) {
    auto work = AllocateElementWork(arena, domain_count, std::min<uint64_t>(handles.size(), block_bound));
    auto data = arena.GetMutable(WorkStorageRange(work));
    for (const auto handle : handles) {
        if (handle >= domain_count) throw std::out_of_range("Element work handle exceeds its domain.");
        MarkElementWorkWord(data, work, handle / 32u, 1u << (handle % 32u));
    }
    FinishElementWork(data, work.Capacity);
    return work;
}

// Restored address ranges remain metadata.
// No vertex values are copied.
inline void SeedElementWorkRanges(BufferArena<uint32_t> &arena, ElementWork &work, std::span<const Range> ranges, uint32_t origin, bool accumulate) {
    if (!accumulate) ClearElementWork(arena, work);
    uint64_t bound = WorkBlockCount(arena, work);
    for (const auto range : ranges) bound += (uint64_t(range.Count) + 510u) / 256u;
    ReserveElementWork(arena, work, bound);
    auto data = arena.GetMutable(WorkStorageRange(work));
    for (const auto range : ranges) {
        const auto start = std::max(uint64_t(range.Offset), uint64_t(origin));
        const auto stop = std::min(uint64_t(range.Offset) + range.Count, uint64_t(origin) + work.Count);
        if (start >= stop) continue;
        const auto end = uint32_t(stop - origin);
        for (uint32_t first = uint32_t(start - origin); first < end;) {
            const auto last = uint32_t(std::min(uint64_t(end), (uint64_t(first) / 32u + 1u) * 32u));
            MarkElementWorkWord(data, work, first / 32u, (~0u >> (32u - (last - first))) << (first % 32u));
            first = last;
        }
    }
    FinishElementWork(data, work.Capacity);
}
inline void ForEachWorkBlock(const BufferArena<uint32_t> &arena, ElementWork work, auto &&fn) {
    if (work.Storage.Slot != arena.Buffer.Slot) throw std::invalid_argument("Element work belongs to a different scratch arena.");
    const auto data = arena.Get(WorkStorageRange(work));
    for (uint32_t i = 0; i < data[0]; ++i) {
        const auto slot = data[WorkHeaderWords + work.Capacity * WorkBlockWords + i];
        const auto block = data.subspan(WorkHeaderWords + slot * WorkBlockWords, WorkBlockWords);
        fn(block[0] - 1u, block.subspan(1u));
    }
}
inline void ForEachWorkElement(const BufferArena<uint32_t> &arena, ElementWork work, auto &&fn) {
    ForEachWorkBlock(arena, work, [&](uint32_t block, auto masks) {
        for (uint32_t word = 0; word < 8u; ++word)
            for (uint32_t bits = masks[word]; bits; bits &= bits - 1u) fn(block * 256u + word * 32u + std::countr_zero(bits));
    });
}
inline bool ElementWorkEmpty(const BufferArena<uint32_t> &arena, ElementWork work) { return WorkBlockCount(arena, work) == 0u; }
