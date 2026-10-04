#pragma once

#include "gpu/MeshletIndex.h"
#include "gpu/Types.h"

// Byte copies and rebases run 256 threads per tile, and each copy thread moves sixteen bytes.
GPU_CONSTANT uint32_t CloneRunThreads = 256u;
GPU_CONSTANT uint32_t CloneCopyThreadBytes = 16u;
GPU_CONSTANT uint32_t CloneCopyTileBytes = CloneRunThreads * CloneCopyThreadBytes;
GPU_CONSTANT uint32_t ClonePairThreads = 32u;
// Clone kernels bind their data buffer, job table, and tile table after the bindless table.
GPU_CONSTANT uint32_t CloneBufferIndex_Jobs = 1u;
GPU_CONSTANT uint32_t CloneBufferIndex_Tiles = 2u;
GPU_CONSTANT uint32_t CloneBufferIndex_Data = 3u;

// A run of bytes copied within one buffer.
struct ByteCopy {
    uint64_t Source, Destination, Bytes;
};
static_assert(sizeof(ByteCopy) == 24);
// A run of uint32 references Stride words apart from ByteOffset, each non-null one taking Delta.
struct IndexRebase {
    uint64_t ByteOffset;
    uint32_t Count, Delta, Stride;
};
static_assert(sizeof(IndexRebase) == 24);
// A run of uint32 handles Stride words apart from ByteOffset, each live one replaced by First plus its rank among Index's members.
struct RankRebase {
    uint64_t ByteOffset;
    uint32_t Count, Stride, First;
    MeshletIndexRef Index;
};
static_assert(sizeof(RankRebase) == 32);
// A run of Bytes-sized records gathered from Index's members in rank order to the records from Destination.
struct RankGather {
    uint32_t Destination, Count, Bytes;
    MeshletIndexRef Index;
};
static_assert(sizeof(RankGather) == 24);
// A run of copied reference pairs and the deltas its non-null pair members take.
struct ReferencePairCopy {
    uint32_t Source, Destination, Count, FirstDelta, SecondDelta;
};
static_assert(sizeof(ReferencePairCopy) == 20);
