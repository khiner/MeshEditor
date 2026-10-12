#pragma once

#include "gpu/RenderTableRuns.h"

// Associates a work instance with one span-tree node at its arena address in the traversal frontier, with the table runs of the instance's record.
struct LodFrontierEntry {
    uint32_t Instance DEFAULT();
    uint32_t NodeAddress DEFAULT();
    uint32_t NodeTable DEFAULT();
    RenderTableRuns Tables DEFAULT();
};
static_assert(sizeof(LodFrontierEntry) == 28, "LodFrontierEntry size");
