#pragma once

#include "gpu/Types.h"

// Polar/azimuth angles in a canonical polygon-corner frame. Negative polar is
// absent. Zero polar remains a valid authored direction when the derived normal
// is zero and the frame uses its deterministic +Z fallback.
struct CustomNormal {
    vec2 Offset DEFAULT(vec2{-1.f, 0.f});
};
static_assert(sizeof(CustomNormal) == 8);
