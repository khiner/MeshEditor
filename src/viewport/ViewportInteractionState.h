#pragma once

#include "gizmo/TransformGizmoTypes.h"

struct PendingTransform {
    vec3 Pivot{};
    quat PivotR{1, 0, 0, 0};
    Transform Delta{};
};

struct StartScreenTransform {
    TransformGizmo::TransformType Value;
};

struct TransformGizmoState {
    TransformGizmo::Config Config;
    TransformGizmo::Mode Mode;
};
