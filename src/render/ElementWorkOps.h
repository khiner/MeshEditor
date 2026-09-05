#pragma once

#include "gpu/ElementWork.h"
#include "metal/BufferArena.h"

#include <bit>
#include <cassert>

// UMA storage: masks, active-word indices, word count, then two uint3 dispatch arguments.
// The first dispatch uses one lane per bit; the second uses one threadgroup per bit.
inline uint32_t WorkWordCount(ElementWork work) { return (work.Count + 31u) / 32u; }
inline Range WorkStorageRange(ElementWork work) { return {work.Storage.Offset, WorkWordCount(work) * 2u + 7u}; }
inline uint64_t WorkArgsOffset(ElementWork work, bool groups = false) { return uint64_t(work.Storage.Offset + WorkWordCount(work) * 2u + (groups ? 4u : 1u)) * sizeof(uint32_t); }

inline ElementWork AllocateElementWork(BufferArena<uint32_t> &arena, uint32_t count) {
    assert(arena.Buffer.Slot != InvalidSlot);
    const auto range = arena.Allocate(((count + 31u) / 32u) * 2u + 7u);
    std::ranges::fill(arena.GetMutable(range), 0u);
    return {{arena.Buffer.Slot, range.Offset}, count};
}

inline void ClearElementWork(BufferArena<uint32_t> &arena, ElementWork work) {
    auto data = arena.GetMutable(WorkStorageRange(work));
    const auto words = WorkWordCount(work);
    for (uint32_t i = 0; i < data[words * 2u]; ++i) data[data[words + i]] = 0;
    data[words * 2u] = data[words * 2u + 1u] = data[words * 2u + 4u] = 0;
    data[words * 2u + 5u] = data[words * 2u + 6u] = 1;
    data[words * 2u + 2u] = data[words * 2u + 3u] = 1;
}

inline void SeedElementWork(BufferArena<uint32_t> &arena, ElementWork work, std::span<const uint32_t> masks, bool accumulate = false) {
    if (!accumulate) ClearElementWork(arena, work);
    auto data = arena.GetMutable(WorkStorageRange(work));
    const auto words = WorkWordCount(work);
    for (uint32_t i = 0; i < std::min(words, uint32_t(masks.size())); ++i) {
        uint32_t mask = masks[i];
        if (i + 1 == words && work.Count % 32u) mask &= (1u << (work.Count % 32u)) - 1u;
        if (mask && data[i] == 0u) data[words + data[words * 2u]++] = i;
        data[i] |= mask;
    }
    data[words * 2u + 1u] = (data[words * 2u] + 7u) / 8u;
    data[words * 2u + 4u] = data[words * 2u] * 32u;
}

// Drop old preview vertices after reset, touching only words already present in the footprint.
inline void IntersectElementWork(BufferArena<uint32_t> &arena, ElementWork work, std::span<const uint32_t> masks) {
    auto data = arena.GetMutable(WorkStorageRange(work));
    const auto words = WorkWordCount(work);
    uint32_t count = 0;
    for (uint32_t i = 0; i < data[words * 2u]; ++i) {
        const auto word = data[words + i];
        data[word] &= word < masks.size() ? masks[word] : 0u;
        if (data[word]) data[words + count++] = word;
    }
    data[words * 2u] = count;
    data[words * 2u + 1u] = (count + 7u) / 8u;
    data[words * 2u + 4u] = count * 32u;
}

inline void ForEachWorkElement(const BufferArena<uint32_t> &arena, ElementWork work, auto &&fn) {
    const auto data = arena.Get(WorkStorageRange(work));
    const auto words = WorkWordCount(work);
    for (uint32_t i = 0; i < data[words * 2u]; ++i) {
        const auto word = data[words + i];
        for (uint32_t bits = data[word]; bits; bits &= bits - 1u) fn(word * 32u + std::countr_zero(bits));
    }
}

inline bool ElementWorkEmpty(const BufferArena<uint32_t> &arena, ElementWork work) {
    return arena.Get(WorkStorageRange(work))[WorkWordCount(work) * 2u] == 0;
}
