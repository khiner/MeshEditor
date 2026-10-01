#pragma once
#include "gpu/Types.h"

// A stable owner for a linked set of blocks, allocated independently of block IDs.
// Flags bit 0 marks a dense set, whose live elements are a prefix of consecutive blocks from First.
struct MeshElementSet {
    uint32_t First DEFAULT(InvalidOffset), Last DEFAULT(InvalidOffset);
    uint32_t Count DEFAULT(), BlockCount DEFAULT(), Revision DEFAULT(), Flags DEFAULT();
};
static_assert(sizeof(MeshElementSet) == 24, "MeshElementSet size");
