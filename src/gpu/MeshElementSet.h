#pragma once
#include "gpu/Types.h"

// First indexes the table arena; BlockCount entries are used within Capacity.
// InvalidOffset entries contain no block.
// Flags bit 0 denotes consecutive blocks with the first Count handles live.
struct MeshElementSet {
    uint32_t First DEFAULT(InvalidOffset), Capacity DEFAULT();
    uint32_t Count DEFAULT(), BlockCount DEFAULT(), Revision DEFAULT(), Flags DEFAULT();
};

// An arena's set headers, table runs and blocks, through which a set's index resolves its table run as the run moves.
struct SetArenaSlots {
    uint32_t Sets DEFAULT(InvalidSlot), Table DEFAULT(InvalidSlot), Blocks DEFAULT(InvalidSlot);
};
