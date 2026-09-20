#pragma once

#include "numeric/quat.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <variant>

using numeric::quat, numeric::vec3, numeric::vec4;

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
RotationUiVariant ToUiVariant(quat, size_t mode); // `mode` = variant alternative
