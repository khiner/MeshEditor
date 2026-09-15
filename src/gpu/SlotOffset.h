#pragma once

#include "gpu/Types.h"

struct SlotOffset {
    uint32_t Slot DEFAULT(InvalidSlot);
    uint32_t Offset DEFAULT();
};
static_assert(sizeof(SlotOffset) == 8, "SlotOffset size");
