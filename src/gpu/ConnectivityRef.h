#pragma once

#include "gpu/SlotOffset.h"

// Independently addressed canonical topology streams.
// Growing one domain does not change the addresses of the other domains.
// Stored references are absolute arena handles.
// Offsets describe the current mesh allocation, not reference bases.
struct ConnectivityRef {
    SlotOffset Outgoing DEFAULT();
    SlotOffset Opposites DEFAULT();
    SlotOffset HalfedgeEdges DEFAULT();
    SlotOffset HalfedgeFaces DEFAULT();
    SlotOffset FaceRanges DEFAULT();
    SlotOffset Edges DEFAULT();
    SlotOffset VertexCorners DEFAULT(); // (first packed fan item, count), per vertex
    uint32_t FanItemsSlot DEFAULT(InvalidSlot); // (canonical corner, canonical face)
};
static_assert(sizeof(ConnectivityRef) == 60, "ConnectivityRef size");
