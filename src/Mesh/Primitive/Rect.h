#pragma once

#include "Mesh/Mesh.h"

struct Rect : Mesh {
    Rect(glm::vec2 half_extents = {1, 1});
};
