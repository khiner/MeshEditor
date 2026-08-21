#ifndef BOUNDS_SHARED_MSL
#define BOUNDS_SHARED_MSL

// Threadgroup AABB fold over the 256 lanes.
// Each lane seeds its slot with its partial min/max, the empty constants when it covers nothing.
// The fold leaves the threadgroup's bounds in shared_min[0]/shared_max[0].
// An empty threadgroup leaves Min > Max, the AABB empty state.
#include <metal_stdlib>
using namespace metal;

constant float3 AabbEmptyMin = float3(3.402823466e38f);
constant float3 AabbEmptyMax = float3(-3.402823466e38f);

constant uint BoundsFoldLanes = 256;
constant uint MeshletBoundsFoldLanes = 64;

// Threadgroup memory cannot live at namespace scope in MSL, so the kernel owns the arrays and
// passes them in. `lanes` is the caller's threadgroup width.
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
