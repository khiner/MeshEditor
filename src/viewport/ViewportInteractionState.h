#pragma once

#include "gizmo/TransformGizmoTypes.h"

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
template<> inline constexpr field::SpecTable<TransformGizmo::Config> Specs<TransformGizmo::Config>{
    {&TransformGizmo::Config::Type, {.Help = "Transform the gizmo applies"}},
    {&TransformGizmo::Config::SnapValue, {.Min = 0.01, .Max = 100, .Unit = FieldUnit::Meters, .Widget = FieldWidget::Drag, .Help = "Snap step per axis when translating, as a factor when scaling, and in degrees from X when rotating"}},
    {&TransformGizmo::Config::Snap, {.Help = "Step gizmo transforms by the snap value"}},
};
template<> inline constexpr field::SpecTable<TransformGizmoState> Specs<TransformGizmoState>{
    {&TransformGizmoState::Mode, {.Help = "Local aligns the gizmo axes with the active object's rotation, and World with the world axes"}},
};
