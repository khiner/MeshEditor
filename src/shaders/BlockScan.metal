#ifndef BLOCKSCAN_MSL
#define BLOCKSCAN_MSL

// Computes an exclusive prefix sum over a uint range.
// Every kernel using these declares ScanSimdGroups + 1 threadgroup words for the scratch.

#include "MslPrelude.metal"

constant uint ScanTileSize = 256u;
constant uint ScanPerThread = 4u;
constant uint ScanBlockElements = ScanTileSize * ScanPerThread;
constant uint ScanSimdGroups = ScanTileSize / 32u;

// Returns the exclusive prefix of `value` and writes the group total to sums[ScanSimdGroups].
// The caller may reuse `sums` after another threadgroup barrier.
inline uint ThreadgroupExclusiveScan(uint value, uint lane, uint simd_lane, uint simd_group, threadgroup uint *sums) {
    const uint rank = simd_prefix_exclusive_sum(value);
    const uint simd_total = simd_sum(value);
    if (simd_lane == 0u) sums[simd_group] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        uint acc = 0u;
        for (uint group = 0u; group < ScanSimdGroups; ++group) {
            const uint total = sums[group];
            sums[group] = acc;
            acc += total;
        }
        sums[ScanSimdGroups] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sums[simd_group] + rank;
}

inline void ScanBlockLoad(device const uint *values, uint count, uint block, uint lane, thread uint *local) {
    const uint base = block * ScanBlockElements + lane * ScanPerThread;
    for (uint k = 0u; k < ScanPerThread; ++k) {
        const uint i = base + k;
        local[k] = i < count ? values[i] : 0u;
    }
}

inline void ScanBlockSum(
    device const uint *values, uint count, uint block, device uint *blocks,
    uint lane, uint simd_lane, uint simd_group, threadgroup uint *sums
) {
    uint local[ScanPerThread];
    ScanBlockLoad(values, count, block, lane, local);
    uint total = 0u;
    for (uint k = 0u; k < ScanPerThread; ++k) total += local[k];
    ThreadgroupExclusiveScan(total, lane, simd_lane, simd_group, sums);
    if (lane == 0u) blocks[block] = sums[ScanSimdGroups];
}

inline void ScanBlockPrefix(
    device uint *blocks, uint block_count, uint lane, uint simd_lane, uint simd_group, threadgroup uint *sums
) {
    uint running = 0u;
    for (uint base = 0u; base < block_count; base += ScanTileSize) {
        const uint block = base + lane;
        const uint value = block < block_count ? blocks[block] : 0u;
        const uint rank = ThreadgroupExclusiveScan(value, lane, simd_lane, simd_group, sums);
        const uint block_total = sums[ScanSimdGroups];
        if (block < block_count) blocks[block] = running + rank;
        running += block_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Returns the prefix for this thread's first value and stores its local prefixes in `local`.
inline uint ScanBlockStart(
    device const uint *values, uint count, uint block, device const uint *blocks,
    uint lane, uint simd_lane, uint simd_group, threadgroup uint *sums, thread uint *local
) {
    ScanBlockLoad(values, count, block, lane, local);
    uint total = 0u;
    for (uint k = 0u; k < ScanPerThread; ++k) total += local[k];
    const uint rank = ThreadgroupExclusiveScan(total, lane, simd_lane, simd_group, sums);
    return blocks[block] + rank;
}

#endif
