#pragma once

#include "gpu/Transform.h"
#include "numeric/mat4.h"

inline mat4 ToMatrix(const Transform &t) {
    return numeric::Translate(I4, t.P) * numeric::ToMat4(numeric::Normalize(t.R)) * numeric::Scale(I4, t.S);
}

Transform ToTransform(const mat4 &);

inline Transform ComposeLocalTransforms(const Transform &parent, const Transform &child) {
    return {parent.R * (parent.S * child.P) + parent.P, parent.R * child.R, parent.S * child.S};
}

// Reduce a possibly non-uniform or mirrored scale to one positive number.
inline float MeanScale(vec3 s) {
    const auto a = numeric::Abs(s);
    return (a.x + a.y + a.z) / 3;
}

// Transform a local point to world space.
inline vec3 TransformPoint(const Transform &t, vec3 p) { return t.R * (p * t.S) + t.P; }
// Transform a world-space point to local space.
inline vec3 InverseTransformPoint(const Transform &t, vec3 p) { return (numeric::Conjugate(t.R) * (p - t.P)) / t.S; }
// Rotate a world-space direction to local space while preserving its magnitude.
inline vec3 InverseTransformDir(const Transform &t, vec3 d) { return numeric::Conjugate(t.R) * d; }
