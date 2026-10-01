#ifndef RADIXSORT_MSL
#define RADIXSORT_MSL

#include "BlockScan.metal"

// Stable four-bit passes over 256-item tiles. Keys and order are separate so
// sorting never moves geometry or attribute payloads.
struct RadixSortView {
    device const uint *Keys;
    device uint *Order, *Temporary, *Histogram, *Totals;
    uint Count, Blocks, KeyStride, KeyWord, Shift;
    bool Odd;
    uint Source(uint i) const { return (Odd ? Temporary : Order)[i]; }
    uint Key(uint source) const { return Keys ? Keys[source * KeyStride + KeyWord] : source; }
};

inline void RadixHistogram(RadixSortView sort, uint lane, uint group, threadgroup atomic_uint *counts) {
    if (lane < 16u) atomic_store_explicit(&counts[lane], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint i = group * 256u + lane;
    if (i < sort.Count) atomic_fetch_add_explicit(&counts[(sort.Key(sort.Source(i)) >> sort.Shift) & 15u], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 16u) sort.Histogram[lane * sort.Blocks + group] = atomic_load_explicit(&counts[lane], memory_order_relaxed);
}

inline void RadixPrefix(RadixSortView sort, uint lane, uint digit, uint sl, uint sg, threadgroup uint *sums) {
    uint carry = 0u;
    for (uint base = 0u; base < sort.Blocks; base += 256u) {
        const uint i = base + lane;
        const uint value = i < sort.Blocks ? sort.Histogram[digit * sort.Blocks + i] : 0u;
        const uint rank = ThreadgroupExclusiveScan(value, lane, sl, sg, sums);
        if (i < sort.Blocks) sort.Histogram[digit * sort.Blocks + i] = carry + rank;
        carry += sums[8];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) sort.Totals[digit] = carry;
}

inline void RadixScatter(RadixSortView sort, uint lane, uint group, uint sl, uint sg, threadgroup uint *groups) {
    const uint i = group * 256u + lane;
    const bool present = i < sort.Count;
    const uint source = present ? sort.Source(i) : 0u;
    const uint digit = present ? (sort.Key(source) >> sort.Shift) & 15u : 0u;
    uint rank = 0u;
    for (uint d = 0u; d < 16u; ++d) {
        const uint ballot = uint((simd_vote::vote_t)simd_ballot(present && digit == d));
        if (sl == 0u) groups[d * 8u + sg] = popcount(ballot);
        if (d == digit) rank = popcount(ballot & ((1u << sl) - 1u));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!present) return;
    for (uint g = 0u; g < sg; ++g) rank += groups[digit * 8u + g];
    uint base = sort.Histogram[digit * sort.Blocks + group];
    for (uint d = 0u; d < digit; ++d) base += sort.Totals[d];
    (sort.Odd ? sort.Order : sort.Temporary)[base + rank] = source;
}

#endif
