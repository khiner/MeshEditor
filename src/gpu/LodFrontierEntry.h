#pragma once

#include "gpu/Types.h"

// Associates a work instance with one span-tree node in the traversal frontier.
struct LodFrontierEntry {
    uint32_t Instance DEFAULT();
    uint32_t Node DEFAULT();
};
static_assert(sizeof(LodFrontierEntry) == 8, "LodFrontierEntry size");
