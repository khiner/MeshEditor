#include "BlockScan.metal"
#include "Bindless.metal"
#include "gpu/ScratchPassPushConstants.h"

kernel void ScratchFill(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant ScratchPassPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i >= pc.Count) return;
    BindlessBufferMutable(uint, b.Buffer, pc.Slot)[pc.Values + i] = pc.Value;
}

// An exclusive scan of a run of values in place, whose sum lands in the entry past them when the run ends with a zero.
kernel void ScratchScanSum(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant ScratchPassPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint sums[ScanSimdGroups + 1u];
    device uint *storage = BindlessBufferMutable(uint, b.Buffer, pc.Slot);
    ScanBlockSum(storage + pc.Values, pc.Count, group_id, storage + pc.Blocks, lane, simd_lane, simd_group, sums);
}
kernel void ScratchScanPrefix(
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant ScratchPassPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint sums[ScanSimdGroups + 1u];
    device uint *storage = BindlessBufferMutable(uint, b.Buffer, pc.Slot);
    ScanBlockPrefix(storage + pc.Blocks, (pc.Count + ScanBlockElements - 1u) / ScanBlockElements, lane, simd_lane, simd_group, sums);
}
kernel void ScratchScanOffsets(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant ScratchPassPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint sums[ScanSimdGroups + 1u];
    device uint *storage = BindlessBufferMutable(uint, b.Buffer, pc.Slot);
    ScanBlockOffsets(storage + pc.Values, pc.Count, group_id, storage + pc.Blocks, storage + pc.Values, lane, simd_lane, simd_group, sums);
}
