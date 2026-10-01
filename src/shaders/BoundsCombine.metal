#ifndef BOUNDSCOMBINE_MSL
#define BOUNDSCOMBINE_MSL

// Combines partial AABBs into each entry's instance bounds.
#include "Bindless.metal"
#include "gpu/AABB.h"
#include "BoundsShared.metal"
#include "gpu/BoundsReducePushConstants.h"
#include "gpu/SelectionAggregate.h"
#include "ElementWorkShared.metal"
#include "VertexBounds.metal"

kernel void BoundsCombineKernel(
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    threadgroup float3 *shared_min [[threadgroup(0)]],
    threadgroup float3 *shared_max [[threadgroup(1)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant BoundsReducePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const uint2 tile=VertexBoundsTile(bindless,pc,group_id);
    if (tile.y == InvalidOffset) return;
    const BoundsEntry entry=scene.BoundsEntries(pc.BoundsEntrySlot)[tile.x];
    if (entry.BoundsNamespace==InvalidOffset) {
        // Static geometry already owns exact bounds in its vertex selection root.
        // Publishing an instance requires no vertex enumeration or copy.
        AABB box{packed_float3(AabbEmptyMin),packed_float3(AabbEmptyMax)};
        if (entry.VertexRoot.Slot!=InvalidSlot) {
            const SelectionAggregate aggregate=BindlessBuffer(SelectionAggregate,bindless.Buffer,entry.VertexRoot.Slot)[entry.VertexRoot.Offset];
            if (aggregate.LiveCount) box=aggregate.Bounds;
        }
        for (uint k=tid;k<entry.InstanceCount;k+=256u)
            BindlessBufferMutable(AABB,bindless.Buffer,pc.BoundsSlot)[entry.FirstInstance+k]=box;
        return;
    }
    const uint child=VertexBoundsIndex(bindless,pc,entry,pc.Level-1u,tile.y*256u+tid);
    float3 lo=AabbEmptyMin, hi=AabbEmptyMax;
    if (child != InvalidOffset) {
        const AABB box=BindlessBuffer(AABB,bindless.Buffer,pc.ValuesSlot)[child];
        lo=float3(box.Min); hi=float3(box.Max);
    }
    FoldSharedAabb(shared_min,shared_max,BoundsFoldLanes,tid,lo,hi);
    const AABB box{packed_float3(shared_min[0]),packed_float3(shared_max[0])};
    if (tid == 0u) {
        const uint destination=VertexBoundsIndex(bindless,pc,entry,pc.Level,tile.y);
        if (destination != InvalidOffset) BindlessBufferMutable(AABB,bindless.Buffer,pc.ValuesSlot)[destination]=box;
        MarkWork(bindless,pc.NextWork,tile.y/256u);
    }
    if (pc.Level == VertexBoundsLevels-1u) {
        for (uint k=tid; k<entry.InstanceCount; k+=256u)
            BindlessBufferMutable(AABB,bindless.Buffer,pc.BoundsSlot)[entry.FirstInstance+k]=box;
    }
}
#endif
