#pragma once

#include "gpu/EditSelectionStorage.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// One run of instance slots sharing a deform state in the posed prelude, whose first instance record holds that state.
struct BoundsEntry {
    uint32_t FirstInstance DEFAULT();
    uint32_t InstanceCount DEFAULT();
    EditSelectionStorage Selection DEFAULT();
    uint32_t VertexBlocksSlot DEFAULT(InvalidSlot);
    uint32_t VertexOwner DEFAULT(InvalidOffset);
    uint32_t BoundsNamespace DEFAULT(InvalidOffset);
    SlotOffset VertexRoot DEFAULT(); // The mesh's live vertex bounds, when it has selection state.
};
static_assert(sizeof(BoundsEntry) == 72, "BoundsEntry size");
