#pragma once

#include "gpu/Types.h"

struct SlotOffset {
    uint32_t Slot DEFAULT(InvalidSlot);
    uint32_t Offset DEFAULT();
};

// The tiles of a pass over scratch records run this many threads.
GPU_CONSTANT uint32_t JobTileThreads = 256u;
// A pass over the tiles of its jobs, and a pass parameter such as a radix shift.
// Jobs holds the jobs, and Tiles the job count then each job's first tile in ascending order.
struct JobTablePushConstants {
    SlotOffset Jobs DEFAULT(), Tiles DEFAULT();
    uint32_t Parameter DEFAULT();
};
// The argument table index of a job table pass's jobs, after the bindless set's indices and a driver's program table.
GPU_CONSTANT uint32_t BufferIndex_Jobs = 6u;

#ifdef __METAL_VERSION__
// A job table kernel's threadgroup and lane, the bindless set, the table and its jobs of type `Job`.
#define JOB_TABLE_ARGS(Job) uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant JobTablePushConstants &table [[buffer(BufferIndex_PushConstants)]], constant Job *jobs [[buffer(BufferIndex_Jobs)]]
// A job table kernel's arguments with layers of threadgroups, whose x position is the tile and z position the layer.
#define JOB_TABLE_LAYERED_ARGS(Job) uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant JobTablePushConstants &table [[buffer(BufferIndex_PushConstants)]], constant Job *jobs [[buffer(BufferIndex_Jobs)]]
// Declares the job `job` of the kernel's threadgroup and the group's tile `tile` within it.
#define JOB_TILE(job) uint tile; constant auto &job = TileJobOf(jobs, b, table, group, tile)
// Also declares the thread's index `index` within the job, for tiles of `width` threads.
#define JOB_TILE_THREAD(job, index, width) JOB_TILE(job); const uint index = tile * (width) + lane
#define JOB_THREAD(job, index) JOB_TILE_THREAD(job, index, JobTileThreads)
#endif
