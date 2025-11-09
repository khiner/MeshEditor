#pragma once

#include "numeric/quat.h"
#include "numeric/vec3.h"

// Separate components
struct Position {
    vec3 Value;
};
struct Rotation {
    quat Value;
};
struct Scale {
    vec3 Value;
};

struct Transform {
    vec3 P{0}; // Position
    quat R{1, 0, 0, 0}; // Rotation
    vec3 S{1}; // Scale
};
