#pragma once

#include "gizmo/TransformGizmoTypes.h"

// Presence indicates active transform
struct PendingTransform {
    vec3 Pivot{};
    quat PivotR{1, 0, 0, 0};
    Transform Delta{};
};

// Requested transform type for the next gizmo drag, latched by keyboard shortcuts.
// Presence == active latch; removed by InteractOverlay after consumption. Singleton on viewport.
struct StartScreenTransform {
    TransformGizmo::TransformType Value;
};

// Current gizmo config/mode (editor state).
struct TransformGizmoState {
    TransformGizmo::Config Config;
    TransformGizmo::Mode Mode;
};
