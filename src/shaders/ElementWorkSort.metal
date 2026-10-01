#include "ElementWorkShared.metal"
#include "RadixSort.metal"
#include "gpu/ElementWorkSortPushConstants.h"

inline ElementWorkSortJob WorkSortJob(device const BindlessSet &b, constant ElementWorkSortPushConstants &pc, uint domain) {
    return reinterpret_cast<device const ElementWorkSortJob *>(BindlessBuffer(uint, b.Buffer, pc.Jobs.Slot) + pc.Jobs.Offset)[domain];
}

inline RadixSortView WorkSort(device const BindlessSet &b, constant ElementWorkSortPushConstants &pc, uint domain) {
    const auto job = WorkSortJob(b, pc, domain);
    const ElementWork work = job.Work;
    device uint *data = BindlessBufferMutable(uint, b.Buffer, work.Storage.Slot) + work.Storage.Offset;
    device uint *temporary = BindlessBufferMutable(uint, b.Buffer, pc.Jobs.Slot) + job.TemporaryOffset;
    const uint count = data[0], blocks = (count + 255u) / 256u;
    return {data + WorkHeaderWords, data + WorkHeaderWords + work.Capacity * WorkBlockWords,
            temporary, temporary + work.Capacity, temporary + work.Capacity + 16u * blocks,
            count, blocks, WorkBlockWords, 0u, pc.Shift, (pc.Shift & 4u) != 0u};
}

kernel void ElementWorkHistogram(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant ElementWorkSortPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    const auto sort = WorkSort(b, pc, group.z);
    if (group.x >= sort.Blocks) return;
    threadgroup atomic_uint counts[16];
    RadixHistogram(sort, lane, group.x, counts);
}

kernel void ElementWorkPrefix(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant ElementWorkSortPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint sums[9];
    RadixPrefix(WorkSort(b, pc, group.z), lane, group.x, sl, sg, sums);
}

kernel void ElementWorkScatter(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant ElementWorkSortPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    const auto sort = WorkSort(b, pc, group.z);
    if (group.x >= sort.Blocks) return;
    threadgroup uint groups[128];
    RadixScatter(sort, lane, group.x, sl, sg, groups);
}

kernel void ElementWorkFinish(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant ElementWorkSortPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup uint totals[8];
    FinishWork(b, WorkSortJob(b, pc, group.z).Work, lane, totals);
}
