#pragma once

#include "Range.h"
#include "gpu/SlotOffset.h"

struct SlottedRange : SlotOffset {
    uint32_t Count{0};

    SlottedRange() = default;
    SlottedRange(Range r, uint32_t slot) : SlotOffset{slot, r.Offset}, Count{r.Count} {}

    operator struct Range() const { return {Offset, Count}; }
};
