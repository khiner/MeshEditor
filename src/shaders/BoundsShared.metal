#ifndef BOUNDS_SHARED_MSL
#define BOUNDS_SHARED_MSL

// Reduces lane AABBs into shared_min[0] and shared_max[0].
// Min > Max represents an empty result.
#include <metal_stdlib>
using namespace metal;

constant float3 AabbEmptyMin = float3(3.402823466e38f);
constant float3 AabbEmptyMax = float3(-3.402823466e38f);

constant uint BoundsFoldLanes = 256;

// Kernels provide the arrays because MSL prohibits threadgroup memory at namespace scope.
inline void FoldSharedAabb(
    threadgroup float3 *shared_min, threadgroup float3 *shared_max,
    uint lanes, uint tid, float3 lo, float3 hi
) {
    // Apple GPU SIMD groups have 32 lanes. Only their partials need shared memory.
    lo = simd_min(lo);
    hi = simd_max(hi);
    if ((tid & 31u) == 0u) {
        shared_min[tid >> 5u] = lo;
        shared_max[tid >> 5u] = hi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) {
        lo = tid < lanes / 32u ? shared_min[tid] : AabbEmptyMin;
        hi = tid < lanes / 32u ? shared_max[tid] : AabbEmptyMax;
        lo = simd_min(lo);
        hi = simd_max(hi);
        if (tid == 0u) {
            shared_min[0] = lo;
            shared_max[0] = hi;
        }
    }
    // BoundsCombine consumes the result from every lane.
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#endif
