#pragma once

#include "Mesh/Mesh.h"

struct Cuboid : Mesh {
    Cuboid(glm::vec3 half_extents = {1, 1, 1});
};
