#pragma once

#include "numeric/quat.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

struct Transform {
    vec3 P{0}; // Position
    quat R{1, 0, 0, 0}; // Rotation
    vec3 S{1}; // Scale
};

inline Transform ComposeLocalTransforms(const Transform &parent, const Transform &child) {
    return {parent.R * (parent.S * child.P) + parent.P, parent.R * child.R, parent.S * child.S};
}

inline vec4 QuatToVec4(quat q) { return {q.x, q.y, q.z, q.w}; }
inline quat Vec4ToQuat(vec4 v) { return {v.w, v.x, v.y, v.z}; }
