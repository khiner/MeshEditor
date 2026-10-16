#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// Selection domains are vertex, edge and face, in that order.
// Seeds and marks also address the halfedge domain.
GPU_CONSTANT uint32_t SelectionHalfedgeDomain = 3u;

// A changed mask word at an arena word: its elements' blocks and their incident blocks become dirty.
struct SelectionSeed {
    uint32_t Domain DEFAULT(), Word DEFAULT(), Bits DEFAULT();
};

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

// The arena streams and the batch's records, which are word offsets into the work slot.
// Mask word Masks[d].Offset + a / 32 holds the bit of the element at arena address a, and InvalidSlot Leaves and Hidden leave aggregates unwritten and nothing hidden.
// A sparse dirty-block hash table occupies the first DirtyMask+1 words.
// Seeds holds SeedCount seeds, and SeedUpdates the index of each seed's mesh update.
// The dirty list starts with its entry count and the other two indirect dispatch dimensions, followed by (domain << 30 | block, update) pairs.
struct SelectionUpdatePushConstants {
    uint32_t CornersSlot DEFAULT(InvalidSlot), VerticesSlot DEFAULT(InvalidSlot);
    uint32_t EdgeSharpnessSlot DEFAULT(InvalidSlot), FaceSharpnessSlot DEFAULT(InvalidSlot);
    GpuArray<uint32_t, 4> Blocks DEFAULT(); // Vertex, edge, face and halfedge membership.
    GpuArray<SlotOffset, 3> Masks DEFAULT();
    GpuArray<uint32_t, 3> Leaves DEFAULT();
    GpuArray<uint32_t, 3> Hidden DEFAULT();
    uint32_t DirtyMask DEFAULT(), WorkSlot DEFAULT(InvalidSlot);
    uint32_t Updates DEFAULT(), Seeds DEFAULT(), SeedUpdates DEFAULT(), List DEFAULT();
    uint32_t SeedCount DEFAULT();
};

// Scatters the selected elements' handles of listed (arena block, first output index) word pairs, which list blocks in ascending local order.
// BlocksSlot holds the domain's block metadata.
struct SelectionGatherPushConstants {
    SlotOffset Blocks DEFAULT();
    SlotOffset Output DEFAULT();
    uint32_t MaskSlot DEFAULT(InvalidSlot), BlocksSlot DEFAULT(InvalidSlot), Count DEFAULT();
};
