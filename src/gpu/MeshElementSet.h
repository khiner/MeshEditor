#pragma once
#include "gpu/Types.h"

// A set's table run: First names its entries in the domain's table arena, BlockCount of the Capacity entries are in use, and an InvalidOffset entry holds no block.
// Flags bit 0 marks a dense set, whose table names consecutive blocks and whose live elements are the first Count handles.
struct MeshElementSet {
    uint32_t First DEFAULT(InvalidOffset), Capacity DEFAULT();
    uint32_t Count DEFAULT(), BlockCount DEFAULT(), Revision DEFAULT(), Flags DEFAULT();
};
static_assert(sizeof(MeshElementSet) == 24, "MeshElementSet size");

// An arena's set headers, table runs and blocks, through which a set's index resolves its table run as the run moves.
struct SetArenaSlots {
    uint32_t Sets DEFAULT(InvalidSlot), Table DEFAULT(InvalidSlot), Blocks DEFAULT(InvalidSlot);
};
static_assert(sizeof(SetArenaSlots) == 12, "SetArenaSlots size");
