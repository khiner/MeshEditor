#pragma once

#include "numeric/vec2.h"

#include "mesh/Mesh.h"

inline Mesh Rect(vec2 half_extents) {
    const auto x = half_extents.x, y = half_extents.y;
    return { 
        {{-x, -y, 0}, {x, -y, 0}, {x, y, 0}, {-x, y, 0}},
        {{0, 1, 2, 3}}
    };
}
