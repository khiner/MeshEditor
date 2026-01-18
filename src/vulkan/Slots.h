#pragma once

#include <cstddef>
#include <cstdint>

#include "BindlessBindings.h"

constexpr uint32_t InvalidSlot{~0u};

struct TypedSlot {
    SlotType Type;
    uint32_t Slot;
    bool operator==(const TypedSlot &) const = default;
};
