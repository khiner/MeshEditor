#ifndef BOUNDSREDUCE_MSL
#define BOUNDSREDUCE_MSL

// Writes one partial AABB per 256-vertex_id tile for the bounds-combine pass.
#include "Bindless.metal"
#include "gpu/AABB.h"
#include "BoundsShared.metal"
#include "ElementWorkShared.metal"
#include "gpu/BoundsReducePushConstants.h"
#include "VertexBounds.metal"

kernel void BoundsReduceKernel(
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
    const DrawData draw=scene.BoundsDraw(entry);
    const uint vertex_id=tile.y*256u+tid;
    float3 lo = AabbEmptyMin;
    float3 hi = AabbEmptyMax;
    if (BoundsVertexLive(bindless,entry,tile.y,tid)) {
        const float3 pos = scene.GetLocalPosition(draw,vertex_id-draw.VertexOffset);
        lo = pos;
        hi = pos;
    }
    // Min > Max represents an empty tile and is neutral under the combine pass's min/max operations.
    FoldSharedAabb(shared_min, shared_max, BoundsFoldLanes, tid, lo, hi);
    if (tid == 0u) {
        const uint destination=VertexBoundsIndex(bindless,pc,entry,0u,tile.y);
        if (destination != InvalidOffset) {
            BindlessBufferMutable(AABB,bindless.Buffer,pc.ValuesSlot)[destination]={packed_float3(shared_min[0]),packed_float3(shared_max[0])};
        }
        MarkWork(bindless,pc.NextWork,tile.y/256u);
    }
}
#endif
