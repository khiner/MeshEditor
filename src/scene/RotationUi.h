#pragma once

#include "numeric/quat.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <cstddef>
#include <variant>

// Mutually exclusive structs to track rotation representation.
// Note: `Transform.R` is the source of truth. These are for slider values only.
// `RotationUiVariant` is reactively synced from `Transform.R` in ProcessComponentEvents.
struct RotationQuat {
    quat Value; // wxyz
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

// Tag: the UI is driving Transform.R this frame — reactive sync should not overwrite RotationUiVariant.
// Emplaced by the rotation slider UI before setting Transform.R; cleared by ProcessComponentEvents.
struct RotationUiDriving {};
