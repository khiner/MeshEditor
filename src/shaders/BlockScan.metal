#ifndef BLOCKSCAN_MSL
#define BLOCKSCAN_MSL

// Exclusive prefix sum over a run of uints, in three passes: sum each block, scan the block sums,
// then start each block from its scanned sum.
// A pass covers one block per threadgroup, except the block-sum scan, which covers one run per threadgroup.
// Every kernel using these declares ScanSimdGroups + 1 threadgroup words for the scratch.

#include "MslPrelude.metal"

constant uint ScanTileSize = 256u; // Threads per threadgroup.
constant uint ScanPerThread = 4u; // Contiguous values one thread scans.
constant uint ScanBlockElements = ScanTileSize * ScanPerThread;
constant uint ScanSimdGroups = ScanTileSize / 32u;

// Exclusive scan of `value` across the threadgroup, leaving the threadgroup total in sums[ScanSimdGroups].
// `sums` is free to reuse once the caller barriers again.
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

// Entries past `count` read as zero, so a partial block sums to the same total.
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

// Where this thread's first value starts in the scan, with its values left in `local`.
// The caller writes the running offsets out, since each scan puts them somewhere different.
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
