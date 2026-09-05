#ifndef BOUNDSREDUCE_MSL
#define BOUNDSREDUCE_MSL

// Writes one partial AABB per 256-vertex tile for the bounds-combine pass.
#include "Bindless.metal"
#include "AABB.metal"
#include "BoundsShared.metal"
#include "ElementWorkShared.metal"
#include "BoundsReducePushConstants.metal"

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
    const bool sparse = pc.Work.Storage.Slot != INVALID_SLOT;
    const uint work_id = sparse ? WorkElement(bindless, pc.Work, group_id) : group_id;
    if (work_id == INVALID_OFFSET) return;
    const uint destination = sparse ? BindlessBuffer(uint, bindless.Buffer, pc.EntryFirstTileSlot)[pc.EntryIndex] + work_id : group_id;
    const uint2 tile = uint2(scene.TileMap(pc.TileMapSlot)[destination]);
    const DrawData draw = scene.Draws(pc.DrawDataSlot)[tile.x];
    const uint i = tile.y * 256u + tid;
    float3 lo = AabbEmptyMin;
    float3 hi = AabbEmptyMax;
    if (i < draw.VertexCountOrHeadImageSlot) {
        const float3 pos = scene.GetLocalPosition(draw, i);
        lo = pos;
        hi = pos;
    }
    // Min > Max represents an empty tile and is neutral under the combine pass's min/max operations.
    FoldSharedAabb(shared_min, shared_max, BoundsFoldLanes, tid, lo, hi);
    if (tid == 0u) {
        device AABB *partials = BindlessBufferMutable(AABB, bindless.Buffer, pc.PartialBoundsSlot);
        if (sparse) MarkWork(bindless, pc.NextWork, work_id / 256u);
        partials[destination] = AABB{packed_float3(shared_min[0]), packed_float3(shared_max[0])};
    }
}

#endif
