#pragma once

#include <glm/gtx/euler_angles.hpp>

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

// Tag: the UI is driving Transform.R this frame — reactive sync should not overwrite RotationUiVariant.
// Emplaced by the rotation slider UI before setting Transform.R; cleared by ProcessComponentEvents.
struct RotationUiDriving {};

// Tracks transform at start of gizmo manipulation. If present, actively manipulating.
struct StartTransform {
    Transform T;
    Transform ParentDelta; // Frozen parent_world * parent_inverse at drag start (identity if unparented)
};

// Bone display length captured at drag start (for head/tail partial transforms).
struct StartBoneLength {
    float Value;
};
