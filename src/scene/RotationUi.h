#pragma once

#include "numeric/quat.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <cstddef>
#include <variant>

// Mutually exclusive structs to track rotation representation.
// `Transform.R` is the source of truth, these hold slider values only.
struct RotationQuat {
    quat Value; // xyzw
};
struct RotationEuler {
    vec3 Value; // xyz degrees
};
struct RotationAxisAngle {
    vec4 Value; // axis (xyz), angle (degrees)
};
using RotationUiVariant = std::variant<RotationQuat, RotationEuler, RotationAxisAngle>;

// Single home for the conversion between `Transform.R` (the source of truth) and a UI representation.
quat ToRotation(const RotationUiVariant &); // normalized
RotationUiVariant ToUiVariant(quat, std::size_t mode); // `mode` = variant alternative

// Tag: the UI is driving Transform.R this frame, so the reverse sync shouldn't overwrite RotationUiVariant.
struct RotationUiDriving {};
