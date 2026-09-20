#pragma once

#include "Field.h"
#include "gizmo/TransformGizmoTypes.h"
#include "numeric/VectorMath.h"

// The pivot a transform drag rotates and scales about, recorded by the drag's first update from the selection it started on.
struct StartPivot {
    vec3 P{};
    quat R{1, 0, 0, 0};
};

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
template<> inline constexpr FieldSpec Spec<TransformGizmo::Config, "SnapValue">{.Min = 0.01, .Max = 100};
