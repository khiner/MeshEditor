#pragma once

#include "numeric/vec3.h"

#include <limits>

struct BBox {
    vec3 Min, Max;

    BBox() : Min(std::numeric_limits<float>::max()), Max(-std::numeric_limits<float>::max()) {}
};
