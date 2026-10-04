#pragma once

#include "gpu/Types.h"

// Origin dots of the selected and active objects, one instance per instance slot.
struct ObjectOriginPushConstants {
    uint32_t TransformSlot DEFAULT();
    float RadiusPx DEFAULT(); // The dot's outer radius in render pixels.
    float OutlinePx DEFAULT(); // The width of its dark rim in render pixels.
};
static_assert(sizeof(ObjectOriginPushConstants) == 12, "ObjectOriginPushConstants size");
