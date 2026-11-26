#pragma once

#include "gpu/Types.h"

// Byte copies and shifts run 256 threads per tile, and each copy thread moves sixteen bytes.
GPU_CONSTANT uint32_t CloneRunThreads = 256u;
GPU_CONSTANT uint32_t CloneCopyThreadBytes = 16u;
GPU_CONSTANT uint32_t CloneCopyTileBytes = CloneRunThreads * CloneCopyThreadBytes;
// Clone kernels bind their data buffer, job table, and first-tile table after the bindless table.
GPU_CONSTANT uint32_t CloneBufferIndex_Jobs = 1u;
GPU_CONSTANT uint32_t CloneBufferIndex_Tiles = 2u;
GPU_CONSTANT uint32_t CloneBufferIndex_Data = 3u;

// A run of bytes copied within one buffer to Copies destinations, Step bytes apart.
struct ByteCopy {
    uint64_t Source, Destination, Bytes;
    uint64_t Step DEFAULT();
    uint32_t Copies DEFAULT(1u);
};
// Copies runs of Count uint32 handles, Step bytes apart from ByteOffset, each run's handles Stride words apart.
// Each handle of copy k other than the null sentinel gains Shift plus k times ShiftStep.
struct HandleOffset {
    uint64_t ByteOffset;
    uint32_t Count, Stride DEFAULT(1u), Shift;
    uint32_t Copies DEFAULT(1u);
    uint64_t Step DEFAULT();
    uint32_t ShiftStep DEFAULT();
};
