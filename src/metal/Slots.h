#pragma once

#include "gpu/BindlessBindings.h"

struct TypedSlot {
    SlotType Type;
    uint32_t Slot;
    bool operator==(const TypedSlot &) const = default;
};
