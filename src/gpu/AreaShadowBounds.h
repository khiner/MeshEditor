#pragma once
#include "gpu/AABB.h"

// Stored just beyond the light array, once per shutter pose.
struct AreaShadowBounds {
    AABB Box;
    uint32_t Valid;
    uint32_t CullBackfaces;
};
struct AreaShadowBoundsParams {
    uint32_t InstanceCount, BoundsSlot, ModelSlot, CullBackfaces;
};
