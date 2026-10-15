#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"
#include "gpu/SlotOffset.h"

// One mesh's classification over its vertex tiles.
// Sparse vertex work tiles its compact order, and a job without work tiles the arena blocks of its live vertices.
struct CornerClassificationJob {
    ConnectivityRef Connectivity DEFAULT();
    ElementWork Vertices DEFAULT();
    uint32_t VertexCount DEFAULT();
    uint32_t FaceCount DEFAULT();
    SlotOffset CornerSectors DEFAULT(); // The record's sector roots, without a slot while it holds none
    ElementWork DirtyBlocks DEFAULT(), NeededBlocks DEFAULT();
    uint32_t State DEFAULT(); // Batch scratch words: incoming count, then face/edge flags
};
static_assert(sizeof(CornerClassificationJob) == 140);

struct CornerClassificationPushConstants {
    uint32_t StorageSlot DEFAULT(InvalidSlot), JobsOffset DEFAULT(), TileMapOffset DEFAULT(), ScratchOffset DEFAULT();
    uint32_t FirstTile DEFAULT();
    uint32_t VertexBlocksSlot DEFAULT(InvalidSlot), EdgeSharpnessSlot DEFAULT(), FaceSharpnessSlot DEFAULT();
};
static_assert(sizeof(CornerClassificationPushConstants) == 32);
