#pragma once

#include "Geometry/Geometry.h"

struct Cuboid : Geometry {
    Cuboid(glm::vec3 half_extents = {1, 1, 1});
};
