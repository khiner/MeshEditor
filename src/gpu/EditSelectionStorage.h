#pragma once

#include "gpu/Types.h"
#include "gpu/SlotOffset.h"

struct EditSelectionStorage {
    SlotOffset VertexBits DEFAULT();
    SlotOffset EdgeBits DEFAULT();
    SlotOffset FaceBits DEFAULT();
    SlotOffset Summary DEFAULT();
};
static_assert(sizeof(EditSelectionStorage) == 32, "EditSelectionStorage size");
