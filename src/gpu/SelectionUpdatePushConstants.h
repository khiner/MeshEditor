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

// The dirty bitset holds one bit per canonical block of each domain, and its word for a domain's block is SelectionDirtyWord.
// Its bits are clear between updates.
inline uint32_t SelectionDirtyWord(uint32_t domain, uint32_t block) { return (block / 32u) * 3u + domain; }

// One mesh's canonical streams, the dirty bitset and the mesh's dirty list.
// A dirty list starts with its entry count and the other two indirect dispatch dimensions, followed by (domain << 30 | block) entries.
// Items are seeds when marking and the dirty list's entries when updating.
// A valid Source domain rewrites the other two domains' words in each entry from it.
struct SelectionUpdatePushConstants {
    ConnectivityRef Connectivity DEFAULT();
    uint32_t CornersSlot DEFAULT(InvalidSlot), VerticesSlot DEFAULT(InvalidSlot);
    uint32_t EdgeSharpnessSlot DEFAULT(InvalidSlot), FaceSharpnessSlot DEFAULT(InvalidSlot);
    GpuArray<uint32_t, 4> Blocks DEFAULT(); // Vertex, edge, face and halfedge membership.
    GpuArray<uint32_t, 3> Owners DEFAULT(); // The mesh's element sets, or InvalidOffset.
    GpuArray<uint32_t, 3> Masks DEFAULT();
    GpuArray<uint32_t, 3> Leaves DEFAULT();
    uint32_t DirtySlot DEFAULT(InvalidSlot);
    SlotOffset List DEFAULT();
    uint32_t FaceCount DEFAULT();
    SlotOffset Items DEFAULT();
    uint32_t Count DEFAULT();
    uint32_t Source DEFAULT(InvalidOffset);
};
static_assert(sizeof(SelectionUpdatePushConstants) == 160, "SelectionUpdatePushConstants size");

// Reduces each domain's leaves over its ascending block list into three consecutive roots.
struct SelectionReducePushConstants {
    GpuArray<SlotOffset, 3> Lists DEFAULT();
    GpuArray<uint32_t, 3> Counts DEFAULT();
    GpuArray<uint32_t, 3> Leaves DEFAULT();
    uint32_t RootsSlot DEFAULT(InvalidSlot), Root DEFAULT();
};
static_assert(sizeof(SelectionReducePushConstants) == 56, "SelectionReducePushConstants size");

// Scatters the selected handles of listed (block, first output index) word pairs in ascending handle order.
struct SelectionGatherPushConstants {
    SlotOffset Blocks DEFAULT();
    SlotOffset Output DEFAULT();
    uint32_t MaskSlot DEFAULT(InvalidSlot), Count DEFAULT();
};
static_assert(sizeof(SelectionGatherPushConstants) == 24, "SelectionGatherPushConstants size");
