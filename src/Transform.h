#pragma once

#include "numeric/quat.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

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

// GLM stores quat in memory as {w,x,y,z}, GPU vec4 as {x,y,z,w}.
inline vec4 QuatToVec4(quat q) { return {q.x, q.y, q.z, q.w}; }
inline quat Vec4ToQuat(vec4 v) { return quat{v.w, v.x, v.y, v.z}; }
