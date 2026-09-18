#pragma once

#include "gpu/EditSelectionStorage.h"
#include "gpu/Types.h"

// One run of instance slots sharing a deform state in the posed prelude, whose first instance record holds that state.
struct BoundsEntry {
    uint32_t FirstInstance DEFAULT();
    uint32_t InstanceCount DEFAULT();
    EditSelectionStorage Selection DEFAULT();
};
static_assert(sizeof(BoundsEntry) == 40, "BoundsEntry size");
