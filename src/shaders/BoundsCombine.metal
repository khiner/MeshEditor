#ifndef BOUNDSCOMBINE_MSL
#define BOUNDSCOMBINE_MSL

// Combines partial AABBs into each entry's instance bounds.
#include "Bindless.metal"
#include "AABB.metal"
#include "BoundsShared.metal"
#include "BoundsReducePushConstants.metal"

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
    const DrawData draw = scene.Draws(pc.DrawDataSlot)[group_id];
    const uint first_tile = BindlessBuffer(uint, bindless.Buffer, pc.EntryFirstTileSlot)[group_id];
    const uint tile_count = max((draw.VertexCountOrHeadImageSlot + 255u) / 256u, 1u);
    device const AABB *partials = BindlessBuffer(AABB, bindless.Buffer, pc.PartialBoundsSlot);
    float3 lo = AabbEmptyMin;
    float3 hi = AabbEmptyMax;
    for (uint t = tid; t < tile_count; t += 256u) {
        const AABB partial = partials[first_tile + t];
        lo = min(lo, float3(partial.Min));
        hi = max(hi, float3(partial.Max));
    }
    // Min > Max represents an empty entry and matches a newly allocated bounds slot.
    FoldSharedAabb(shared_min, shared_max, BoundsFoldLanes, tid, lo, hi);
    const AABB bounds{packed_float3(shared_min[0]), packed_float3(shared_max[0])};
    device AABB *out_bounds = BindlessBufferMutable(AABB, bindless.Buffer, pc.BoundsSlot);
    for (uint k = tid; k < draw.ElementIdOffset; k += 256u) {
        out_bounds[draw.FirstInstance + k] = bounds;
    }
}

#endif
