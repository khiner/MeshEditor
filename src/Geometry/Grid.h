#pragma once

#include "Geometry/Geometry.h"

struct Grid : Geometry {
    Grid(glm::vec2 half_extents = {1, 1});
};
