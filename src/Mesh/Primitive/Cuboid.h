#pragma once

#include "mesh/Mesh.h"

inline Mesh Cuboid(vec3 half_extents) {
    const auto x = half_extents.x, y = half_extents.y, z = half_extents.z;
    return {
        {
            {-x, -y, -z},
            {x, -y, -z},
            {x, y, -z},
            {-x, y, -z},
            {-x, -y, z},
            {x, -y, z},
            {x, y, z},
            {-x, y, z},
        },
        {
            {0, 3, 2, 1}, // front
            {4, 5, 6, 7}, // back
            {0, 1, 5, 4}, // bottom
            {3, 7, 6, 2}, // top
            {0, 4, 7, 3}, // left
            {1, 2, 6, 5}, // right
        }
    };
}
