#pragma once

#include "numeric/vec2.h"

#include "mesh/Mesh.h"

struct Rect : Mesh {
    Rect(vec2 half_extents = {1, 1});
};
