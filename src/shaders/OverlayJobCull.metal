#ifndef OVERLAYJOBCULL_MSL
#define OVERLAYJOBCULL_MSL

#include "Bindless.metal"
#include "CompactPresent.metal"
#include "MeshDispatchArgs.metal"
#include "OverlayJob.metal"
#include "OverlayJobCullPushConstants.metal"
#include "OverlayJobKind.metal"
#include "SceneUBO.metal"

constant uint OverlayCullBlockSize = 256u;
constant uint OverlayCullSimdGroups = OverlayCullBlockSize / 32u;

inline bool OverlayJobEnabled(constant SceneViewUBO &view, OverlayJob job, uint instance_state, bool extras_only) {
    if (view.ShowOverlays == 0u) return false;
    if (extras_only) return job.Kind == OverlayJobKind_Extras && view.ShowExtras != 0u;
    if (job.Kind == OverlayJobKind_Extras) return view.ShowExtras != 0u;
    if ((instance_state & STATE_SELECTED) == 0u) return false;
    if (job.Kind == OverlayJobKind_Bounds) {
        return view.ShowBoundingBoxes != 0u;
    }
    return job.Kind == OverlayJobKind_TetWire && view.ShowTetWireframe != 0u;
}

inline uint OverlayJobPresent(
    constant SceneViewUBO &view, device const BindlessSet &bindless,
    constant OverlayJobCullPushConstants &pc, uint i
) {
    const OverlayJob job = BindlessBuffer(OverlayJob, bindless.Buffer, pc.JobsSlot)[i];
    const uint state = uint(BindlessBuffer(uchar, bindless.InstanceStateBuffer, pc.InstanceStateSlot)[job.InstanceIndex]);
    return OverlayJobEnabled(view, job, state, pc.ExtrasOnly != 0u) ? 1u : 0u;
}

kernel void OverlayJobBlockCount(
    uint thread_index [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint block [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant OverlayJobCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint simd_counts[OverlayCullSimdGroups];
    const uint i = block * OverlayCullBlockSize + thread_index;
    uint present = 0u;
    if (i < pc.JobCount) present = OverlayJobPresent(view, bindless, pc, i);
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, OverlayCullSimdGroups);
    if (thread_index == 0u) {
        BindlessBufferMutable(uint, bindless.Buffer, pc.BlockStateSlot)[block] = compact.y;
    }
}

kernel void OverlayJobPrefix(
    uint thread_index [[thread_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant OverlayJobCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (thread_index != 0u) return;
    device uint *blocks = BindlessBufferMutable(uint, bindless.Buffer, pc.BlockStateSlot);
    const uint block_count = (pc.JobCount + OverlayCullBlockSize - 1u) / OverlayCullBlockSize;
    uint total = 0u;
    for (uint block = 0u; block < block_count; ++block) {
        const uint count = blocks[block];
        blocks[block] = total;
        total += count;
    }
    BindlessBufferMutable(MeshDispatchArgs, bindless.Buffer, pc.DispatchArgsSlot)[0] = {total, 1u, 1u};
}

kernel void OverlayJobEmit(
    uint thread_index [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint block [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant OverlayJobCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint simd_counts[OverlayCullSimdGroups];
    const uint i = block * OverlayCullBlockSize + thread_index;
    uint present = 0u;
    if (i < pc.JobCount) present = OverlayJobPresent(view, bindless, pc, i);
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, OverlayCullSimdGroups);
    if (present == 0u) return;
    const uint block_offset = BindlessBuffer(uint, bindless.Buffer, pc.BlockStateSlot)[block];
    BindlessBufferMutable(uint, bindless.Buffer, pc.VisibleSlot)[block_offset + compact.x] = i;
}

#endif
