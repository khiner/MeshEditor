#pragma once

#include "FieldLimits.h"
#include "gizmo/TransformGizmoTypes.h"
#include "numeric/VectorMath.h"

struct PendingTransform {
    vec3 Pivot{};
    quat PivotR{1, 0, 0, 0};
    Transform Delta{};

    // Apply the resolved gizmo delta to an object in world space.
    Transform ApplyTo(const Transform &start, bool scale_locked) const {
        const auto inverse = Conjugate(PivotR);
        const auto offset = start.P - Pivot;
        return {
            Delta.P + Pivot + Rotate(Delta.R, scale_locked ? offset : PivotR * (inverse * offset * Delta.S)),
            Normalize(Delta.R * start.R),
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
template<> struct FieldLimits<&TransformGizmoState::Config, &TransformGizmo::Config::SnapValue> : Within<0.01f, 100.f> {};
