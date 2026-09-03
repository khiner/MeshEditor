#ifndef COMPACTPRESENT_MSL
#define COMPACTPRESENT_MSL

// Returns a predicate's stable rank and the total set predicates across the threadgroup.
inline uint2 CompactPresent(
    uint present, uint thread_index, uint lane, threadgroup uint *simd_counts, uint simd_group_count
) {
    const uint simd_rank = metal::simd_prefix_exclusive_sum(present);
    const uint simd_count = metal::simd_sum(present);
    const uint simdgroup = thread_index / 32u;
    if (lane == 0u) simd_counts[simdgroup] = simd_count;
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint rank = simd_rank;
    for (uint i = 0u; i < simdgroup; ++i) rank += simd_counts[i];
    uint count = 0u;
    for (uint i = 0u; i < simd_group_count; ++i) count += simd_counts[i];
    return uint2(rank, count);
}

#endif
