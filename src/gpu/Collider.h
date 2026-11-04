#pragma once
#include "gpu/SlotOffset.h"

// Word offsets of a collider over Leaves boxes in a job's scratch storage.
// The runs hold Morton keys, sort order and its temporaries, then the radix tree's node boxes, parents, children and refit counters.
struct ColliderRuns {
    uint32_t Leaves DEFAULT();
    uint32_t Keys DEFAULT(), Order DEFAULT(), Temporary DEFAULT(), Histogram DEFAULT(), Totals DEFAULT();
    uint32_t NodeBoxes DEFAULT(), NodeParents DEFAULT(), Children DEFAULT(), Counters DEFAULT();
};
