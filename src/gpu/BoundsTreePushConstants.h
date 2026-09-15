#pragma once

#include "gpu/Types.h"
#include "gpu/ElementWork.h"
#include "gpu/SlotOffset.h"

struct BoundsTreePushConstants {
    ElementWork Work DEFAULT();
    ElementWork NextWork DEFAULT();
    SlotOffset Input DEFAULT();
    SlotOffset Output DEFAULT();
    uint32_t InputCount DEFAULT();
    SlotOffset InstanceBounds DEFAULT();
    uint32_t InstanceCount DEFAULT();
};
static_assert(sizeof(BoundsTreePushConstants) == 56, "BoundsTreePushConstants size");
