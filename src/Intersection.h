#pragma once

#include <cstdint>

struct Intersection {
    uint32_t Index; // Face index
    float Distance; // Distance along the test ray (not included!) to the intersection point on the face
};
