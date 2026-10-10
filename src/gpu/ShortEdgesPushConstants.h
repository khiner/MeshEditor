#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// A record's halfedges and the word that marks it as holding an edge shorter than Distance.
struct ShortEdgesPushConstants {
    ConnectivityRef Connectivity DEFAULT();
    SlotOffset Flag DEFAULT();
    uint32_t VertexSlot DEFAULT(InvalidSlot), CornerSlot DEFAULT(InvalidSlot);
    uint32_t HalfedgeFirst DEFAULT(), HalfedgeCount DEFAULT(), FaceCount DEFAULT();
    float Distance DEFAULT();
};
static_assert(sizeof(ShortEdgesPushConstants) == 92, "ShortEdgesPushConstants size");
