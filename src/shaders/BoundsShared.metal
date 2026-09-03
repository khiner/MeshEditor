#ifndef BOUNDS_SHARED_MSL
#define BOUNDS_SHARED_MSL

// Reduces lane AABBs into shared_min[0] and shared_max[0].
// Min > Max represents an empty result.
#include <metal_stdlib>
using namespace metal;

constant float3 AabbEmptyMin = float3(3.402823466e38f);
constant float3 AabbEmptyMax = float3(-3.402823466e38f);

constant uint BoundsFoldLanes = 256;
constant uint MeshletBoundsFoldLanes = 64;

// Kernels provide the arrays because MSL prohibits threadgroup memory at namespace scope.
inline void FoldSharedAabb(
    threadgroup float3 *shared_min, threadgroup float3 *shared_max,
    uint lanes, uint tid, float3 lo, float3 hi
) {
    shared_min[tid] = lo;
    shared_max[tid] = hi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes / 2; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#endif
