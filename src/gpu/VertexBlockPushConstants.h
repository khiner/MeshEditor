#pragma once

#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// Each canonical vertex block spans this many threadgroups, one lane per element slot.
GPU_CONSTANT uint32_t VertexBlockGroups = 4u;

// One instance's draw over its mesh's canonical vertex blocks.
struct VertexBlockPushConstants {
    SlotOffset Blocks DEFAULT(); // The mesh's ascending vertex blocks, from this dispatch's first block.
    uint32_t Instance DEFAULT(); // The drawn instance record.
    uint32_t MembershipSlot DEFAULT(InvalidSlot); // The vertex arena's block metadata with its live masks.
    // Static block bounds, InvalidSlot for a posed instance or a mesh without selection aggregates.
    uint32_t LeafSlot DEFAULT(InvalidSlot);
    // Posed block bounds, InvalidOffset for a static instance.
    uint32_t BoundsNamespace DEFAULT(InvalidOffset);
    uint32_t BoundsNodesSlot DEFAULT(InvalidSlot);
    uint32_t BoundsValuesSlot DEFAULT(InvalidSlot);
    uint32_t BoundsMembersSlot DEFAULT(InvalidSlot);
    // Excite mode draws only the selected vertices and picks them by their unoffset ids.
    uint32_t SoundPoints DEFAULT();
    // A visual draw culls the blocks this depth pyramid hides, and InvalidSlot keeps every block.
    uint32_t PyramidSamplerSlot DEFAULT(InvalidSlot);
    // A visual draw culls the blocks whose projected diameter falls below this many pixels.
    float MinDiameterPixels DEFAULT();
};
static_assert(sizeof(VertexBlockPushConstants) == 48, "VertexBlockPushConstants size");
