#pragma once

#include "gpu/Transform.h"
#include "numeric/mat4.h"

inline mat4 ToMatrix(const Transform &t) {
    return glm::translate(I4, t.P) * glm::mat4_cast(glm::normalize(t.R)) * glm::scale(I4, t.S);
}

Transform ToTransform(const mat4 &);

inline Transform ComposeLocalTransforms(const Transform &parent, const Transform &child) {
    return {parent.R * (parent.S * child.P) + parent.P, parent.R * child.R, parent.S * child.S};
}

// Reduce a possibly non-uniform or mirrored scale to one positive number.
inline float MeanScale(vec3 s) {
    const auto a = glm::abs(s);
    return (a.x + a.y + a.z) / 3;
}

// A local point in world space.
inline vec3 TransformPoint(const Transform &t, vec3 p) { return t.R * (p * t.S) + t.P; }
// A world-space point in the transform's local frame.
inline vec3 InverseTransformPoint(const Transform &t, vec3 p) { return (glm::conjugate(t.R) * (p - t.P)) / t.S; }
// A world-space direction in the transform's local frame. Rotation only, so the magnitude is preserved.
inline vec3 InverseTransformDir(const Transform &t, vec3 d) { return glm::conjugate(t.R) * d; }
