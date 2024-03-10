#pragma once

#include "vec2.h"

#include "Mesh/Mesh.h"

struct Rect : Mesh {
    Rect(vec2 half_extents = {1, 1});
};
