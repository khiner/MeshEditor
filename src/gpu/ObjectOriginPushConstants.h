#pragma once

#include "gpu/Types.h"

// Origin dots of the selected and active objects, one instance per entry of the instance drawing order.
struct ObjectOriginPushConstants {
    uint32_t TransformSlot DEFAULT();
    uint32_t InstanceMapSlot DEFAULT(); // The instance slots in drawing order.
    float RadiusPx DEFAULT(); // The dot's outer radius in render pixels.
    float OutlinePx DEFAULT(); // The width of its dark rim in render pixels.
};
