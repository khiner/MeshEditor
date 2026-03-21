#pragma once

#include <glm/gtx/euler_angles.hpp>

// Mutually exclusive structs to track rotation representation.
// Note: `Rotation` is still the source of truth transformation component. These are for slider values only.
// `RotationUiVariant` is reactively synced from `Rotation` in ProcessComponentEvents.
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

// Tag: the UI is driving Rotation this frame — reactive sync should not overwrite RotationUiVariant.
// Emplaced by the rotation slider UI before setting Rotation; cleared by ProcessComponentEvents.
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

inline void SetTransform(entt::registry &r, entt::entity e, const Transform &t, bool propagate_to_children = true) {
    r.emplace_or_replace<Position>(e, t.P);
    r.emplace_or_replace<Rotation>(e, t.R);
    // Frozen entities can't have their scale changed.
    if (!r.all_of<Frozen>(e)) r.emplace_or_replace<Scale>(e, t.S);

    UpdateWorldTransform(r, e, propagate_to_children);
}
