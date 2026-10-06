#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// Selection domains are vertex, edge and face, in that order.
// Seeds and marks also address the halfedge domain.
GPU_CONSTANT uint32_t SelectionHalfedgeDomain = 3u;

// A changed mask word: its elements' blocks and their incident blocks become dirty.
struct SelectionSeed {
    uint32_t Domain DEFAULT(), Word DEFAULT(), Bits DEFAULT();
};
static_assert(sizeof(SelectionSeed) == 12, "SelectionSeed size");

// Open-addressed dirty blocks use zero for an empty slot and entry+1 as the key.
inline uint32_t SelectionDirtyHash(uint32_t key, uint32_t mask) {
    key = (key ^ (key >> 16u)) * 0x7feb352du;
    key = (key ^ (key >> 15u)) * 0x846ca68bu;
    return (key ^ (key >> 16u)) & mask;
}

// One mesh's part of a batched selection update.
// A valid Source domain rewrites the other two domains' words in each of the mesh's entries from it.
// Root identifies its three derived selection trees.
struct SelectionMeshUpdate {
    ConnectivityRef Connectivity DEFAULT();
    GpuArray<uint32_t, 3> Owners DEFAULT(); // The mesh's element sets, or InvalidOffset.
    uint32_t FaceCount DEFAULT();
    uint32_t Source DEFAULT(InvalidOffset);
    uint32_t Root DEFAULT();
};
static_assert(sizeof(SelectionMeshUpdate) == 84, "SelectionMeshUpdate size");

// The canonical streams and the batch's records, which are word offsets into the work slot.
// A sparse dirty-block hash table occupies the first DirtyMask+1 words.
// Seeds holds SeedCount seeds, and SeedUpdates the index of each seed's mesh update.
// The dirty list starts with its entry count and the other two indirect dispatch dimensions, followed by (domain << 30 | block, update) pairs.
struct SelectionUpdatePushConstants {
    uint32_t CornersSlot DEFAULT(InvalidSlot), VerticesSlot DEFAULT(InvalidSlot);
    uint32_t EdgeSharpnessSlot DEFAULT(InvalidSlot), FaceSharpnessSlot DEFAULT(InvalidSlot);
    GpuArray<uint32_t, 4> Blocks DEFAULT(); // Vertex, edge, face and halfedge membership.
    GpuArray<uint32_t, 3> Masks DEFAULT();
    GpuArray<uint32_t, 3> Leaves DEFAULT();
    GpuArray<uint32_t, 3> Hidden DEFAULT();
    uint32_t DirtyMask DEFAULT(), WorkSlot DEFAULT(InvalidSlot);
    uint32_t Updates DEFAULT(), Seeds DEFAULT(), SeedUpdates DEFAULT(), List DEFAULT();
    uint32_t SeedCount DEFAULT();
};
static_assert(sizeof(SelectionUpdatePushConstants) == 96, "SelectionUpdatePushConstants size");

// Scatters the selected handles of listed (block, first output index) word pairs in ascending handle order.
struct SelectionGatherPushConstants {
    SlotOffset Blocks DEFAULT();
    SlotOffset Output DEFAULT();
    uint32_t MaskSlot DEFAULT(InvalidSlot), Count DEFAULT();
};
static_assert(sizeof(SelectionGatherPushConstants) == 24, "SelectionGatherPushConstants size");
