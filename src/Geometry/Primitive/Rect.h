#pragma once

#include "Geometry/Geometry.h"

struct Rect : Geometry {
    Rect(glm::vec2 half_extents = {1, 1});
};
