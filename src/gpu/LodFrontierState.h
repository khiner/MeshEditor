#pragma once

#include "gpu/Types.h"

// Stores traversal frontier size and dispatch block count for one level.
struct LodFrontierState {
    uint32_t NodeCount DEFAULT();
    uint32_t BlockCount DEFAULT();
};
static_assert(sizeof(LodFrontierState) == 8, "LodFrontierState size");
