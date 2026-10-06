#pragma once

#include "gpu/ElementWork.h"
#include "gpu/MeshElementBlock.h"
#include "gpu/MeshletIndex.h"
#include "gpu/Types.h"

// Byte copies and rebases run 256 threads per tile, and each copy thread moves sixteen bytes.
GPU_CONSTANT uint32_t CloneRunThreads = 256u;
GPU_CONSTANT uint32_t CloneCopyThreadBytes = 16u;
GPU_CONSTANT uint32_t CloneCopyTileBytes = CloneRunThreads * CloneCopyThreadBytes;
// Clone kernels bind their data buffer, job table, and tile table after the bindless table.
GPU_CONSTANT uint32_t CloneBufferIndex_Jobs = 1u;
GPU_CONSTANT uint32_t CloneBufferIndex_Tiles = 2u;
GPU_CONSTANT uint32_t CloneBufferIndex_Data = 3u;
GPU_CONSTANT uint32_t CloneBufferIndex_Maps = 4u;

// A run of bytes copied within one buffer.
struct ByteCopy {
    uint64_t Source, Destination, Bytes;
};
static_assert(sizeof(ByteCopy) == 24);
// Canonical or origin-relative references remapped through a hash table of owned blocks.
struct BlockRebase {
    uint64_t ByteOffset;
    uint32_t Count, Stride;
    uint32_t MapOffset, MapCapacity;
    uint32_t SourceOrigin, DestinationOrigin, Span;
};
static_assert(sizeof(BlockRebase) == 40);
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
