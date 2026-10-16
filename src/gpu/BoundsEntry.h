#pragma once

#include "gpu/EditSelectionStorage.h"

// A run of instance slots sharing the first instance's deform state in the posed prelude.
struct BoundsEntry {
    uint32_t FirstInstance DEFAULT();
    uint32_t InstanceCount DEFAULT();
    EditSelectionStorage Selection DEFAULT();
    uint32_t VertexBlocksSlot DEFAULT(InvalidSlot);
    uint32_t VertexOwner DEFAULT(InvalidOffset);
    uint32_t BoundsNamespace DEFAULT(InvalidOffset);
    SlotOffset VertexRoot DEFAULT(); // The mesh's live vertex bounds, when it has selection state.
};
