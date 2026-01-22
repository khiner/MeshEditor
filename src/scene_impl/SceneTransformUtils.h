#pragma once

#include <glm/gtx/euler_angles.hpp>

namespace {
// Mutually exclusive structs to track rotation representation.
// Note: `Rotation` is still the source of truth transformation component. These are for slider values only.
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

// Tracks transform at start of gizmo manipulation. If present, actively manipulating.
struct StartTransform {
    Transform T;
};

void SetRotation(entt::registry &r, entt::entity e, const quat &v) {
    r.emplace_or_replace<Rotation>(e, v);
    if (!r.all_of<RotationUiVariant>(e)) {
        r.emplace<RotationUiVariant>(e, RotationQuat{v});
        return;
    }

    r.patch<RotationUiVariant>(e, [&](auto &rotation_ui) {
        std::visit(
            overloaded{
                [&](RotationQuat &v_ui) { v_ui.Value = v; },
                [&](RotationEuler &v_ui) {
                    float x, y, z;
                    glm::extractEulerAngleXYZ(glm::mat4_cast(v), x, y, z);
                    v_ui.Value = glm::degrees(vec3{x, y, z});
                },
                [&](RotationAxisAngle &v_ui) {
                    const auto q = glm::normalize(v);
                    v_ui.Value = {glm::axis(q), glm::degrees(glm::angle(q))};
                },
            },
            rotation_ui
        );
    });
}

void SetTransform(entt::registry &r, entt::entity e, const Transform &t) {
    r.emplace_or_replace<Position>(e, t.P);
    // Avoid replacing rotation UI slider values if the value hasn't changed.
    if (!r.all_of<Rotation>(e) || r.get<Rotation>(e).Value != t.R) SetRotation(r, e, t.R);
    // Frozen entities can't have their scale changed.
    if (!r.all_of<Frozen>(e)) r.emplace_or_replace<Scale>(e, t.S);

    UpdateWorldMatrix(r, e);
}
} // namespace
