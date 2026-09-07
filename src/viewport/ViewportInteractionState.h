#pragma once

#include "gizmo/TransformGizmoTypes.h"

struct PendingTransform {
    vec3 Pivot{};
    quat PivotR{1, 0, 0, 0};
    Transform Delta{};

    // Apply the resolved gizmo delta to an object in world space.
    Transform ApplyTo(const Transform &start, bool scale_locked) const {
        const auto inverse = numeric::Conjugate(PivotR);
        const auto offset = start.P - Pivot;
        return {
            Delta.P + Pivot + numeric::Rotate(Delta.R, scale_locked ? offset : PivotR * (inverse * offset * Delta.S)),
            numeric::Normalize(Delta.R * start.R),
            scale_locked ? start.S : Delta.S * start.S,
        };
    }
};

struct StartScreenTransform {
    TransformGizmo::TransformType Value;
};

struct TransformGizmoState {
    TransformGizmo::Config Config;
    TransformGizmo::Mode Mode;
};
