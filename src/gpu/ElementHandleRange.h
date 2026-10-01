#pragma once

#include "gpu/SlotOffset.h"

// A compact output domain, either a canonical run or GPU-produced handle list.
struct ElementHandleRange {
    SlotOffset Handles DEFAULT();
    uint32_t First DEFAULT(), Count DEFAULT();
};
static_assert(sizeof(ElementHandleRange) == 16);
