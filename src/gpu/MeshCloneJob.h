#pragma once

#include "gpu/Types.h"

// Byte copies and shifts run 256 threads per tile, and each copy thread moves sixteen bytes.
GPU_CONSTANT uint32_t CloneRunThreads = 256u;
GPU_CONSTANT uint32_t CloneCopyThreadBytes = 16u;
GPU_CONSTANT uint32_t CloneCopyTileBytes = CloneRunThreads * CloneCopyThreadBytes;
// Clone kernels bind their data buffer, job table, and tile table after the bindless table.
GPU_CONSTANT uint32_t CloneBufferIndex_Jobs = 1u;
GPU_CONSTANT uint32_t CloneBufferIndex_Tiles = 2u;
GPU_CONSTANT uint32_t CloneBufferIndex_Data = 3u;

// A run of bytes copied within one buffer.
struct ByteCopy {
    uint64_t Source, Destination, Bytes;
};
static_assert(sizeof(ByteCopy) == 24);
// A run of uint32 handles Stride words apart from ByteOffset, each live one shifted by Shift.
struct HandleOffset {
    uint64_t ByteOffset;
    uint32_t Count, Stride, Shift;
};
static_assert(sizeof(HandleOffset) == 24);
