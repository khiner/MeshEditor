#ifndef BOUNDS_TREE_MSL
#define BOUNDS_TREE_MSL
#include "Bindless.metal"
#include "AABB.metal"
#include "BoundsShared.metal"
#include "BoundsTreePushConstants.metal"
#include "ElementWorkShared.metal"

kernel void BoundsTreeKernel(
    uint tid [[thread_position_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
    threadgroup float3 *shared_min [[threadgroup(0)]], threadgroup float3 *shared_max [[threadgroup(1)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant BoundsTreePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const uint i = WorkElement(bindless, pc.Work, group);
    if (i == INVALID_OFFSET) return;
    const uint child = i * 256u + tid;
    float3 lo = AabbEmptyMin, hi = AabbEmptyMax;
    if (child < pc.InputCount) {
        const AABB box = BindlessBuffer(AABB, bindless.Buffer, pc.Input.Slot)[pc.Input.Offset + child];
        lo = float3(box.Min); hi = float3(box.Max);
    }
    FoldSharedAabb(shared_min, shared_max, BoundsFoldLanes, tid, lo, hi);
    const AABB box{packed_float3(shared_min[0]), packed_float3(shared_max[0])};
    if (tid == 0u) {
        BindlessBufferMutable(AABB, bindless.Buffer, pc.Output.Slot)[pc.Output.Offset + i] = box;
        MarkWork(bindless, pc.NextWork, i / 256u);
    }
    for (uint k = tid; k < pc.InstanceCount; k += 256u)
        BindlessBufferMutable(AABB, bindless.Buffer, pc.InstanceBounds.Slot)[pc.InstanceBounds.Offset + k] = box;
}
#endif
