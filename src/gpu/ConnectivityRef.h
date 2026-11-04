#pragma once

#include "gpu/ElementTables.h"

// Each stream contains its domain's values at arena addresses, and every stored reference is a record-relative handle.
struct ConnectivityRef {
    ElementTables Tables DEFAULT();
    uint32_t CornersSlot DEFAULT(InvalidSlot); // Each corner's vertex handle, in the index buffers
    uint32_t OutgoingSlot DEFAULT(InvalidSlot);
    uint32_t OppositesSlot DEFAULT(InvalidSlot);
    uint32_t HalfedgeEdgesSlot DEFAULT(InvalidSlot);
    uint32_t HalfedgeFacesSlot DEFAULT(InvalidSlot);
    uint32_t FaceRangesSlot DEFAULT(InvalidSlot);
    uint32_t EdgesSlot DEFAULT(InvalidSlot);
    uint32_t VertexCornersSlot DEFAULT(InvalidSlot); // (first packed fan item, count), per vertex
    uint32_t FanItemsSlot DEFAULT(InvalidSlot); // (corner, face)
};
