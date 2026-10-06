#pragma once

#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

struct EditSelectionStorage {
    SlotOffset VertexBits DEFAULT();
    SlotOffset EdgeBits DEFAULT();
    SlotOffset FaceBits DEFAULT();
    SlotOffset Summary DEFAULT();
    uint32_t VertexHiddenSlot DEFAULT(InvalidSlot), EdgeHiddenSlot DEFAULT(InvalidSlot), FaceHiddenSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(EditSelectionStorage) == 44, "EditSelectionStorage size");
