#pragma once

#include "gpu/Transform.h"
#include "numeric/MatrixMath.h"

using numeric::I4;

inline mat4 ToMatrix(const Transform &t) {
    return Translate(I4, t.P) * ToMat4(Normalize(t.R)) * Scale(I4, t.S);
}

Transform ToTransform(const mat4 &);

inline Transform ComposeLocalTransforms(const Transform &parent, const Transform &child) {
    return {parent.R * (parent.S * child.P) + parent.P, parent.R * child.R, parent.S * child.S};
}

// Reduce a possibly non-uniform or mirrored scale to one positive number.
inline float MeanScale(vec3 s) {
    const auto a = Abs(s);
    return (a.x + a.y + a.z) / 3;
}

// Transform a local point to world space.
inline vec3 TransformPoint(const Transform &t, vec3 p) { return t.R * (p * t.S) + t.P; }
// Transform a world-space point to local space.
inline vec3 InverseTransformPoint(const Transform &t, vec3 p) { return (Conjugate(t.R) * (p - t.P)) / t.S; }
// Rotate a world-space direction to local space while preserving its magnitude.
inline vec3 InverseTransformDir(const Transform &t, vec3 d) { return Conjugate(t.R) * d; }
