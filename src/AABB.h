#pragma once

#include "numeric/vec3.h"

#include <limits>

struct AABB {
    vec3 Min, Max;

    AABB() : Min(std::numeric_limits<float>::max()), Max(-std::numeric_limits<float>::max()) {}
};
