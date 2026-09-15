#pragma once

#include "gpu/Types.h"

#ifndef __METAL_VERSION__
#include <limits>
#endif

// Min > Max denotes empty local-space bounds and disables culling.
struct AABB {
    vec3 Min DEFAULT(std::numeric_limits<float>::max());
    vec3 Max DEFAULT(-std::numeric_limits<float>::max());
#ifndef __METAL_VERSION__
    bool operator==(const AABB &) const = default;
#endif
};
static_assert(sizeof(AABB) == 24, "AABB size");
