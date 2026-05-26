#pragma once

#include "gizmo/TransformGizmoTypes.h"
#include "numeric/ray.h"
#include "numeric/vec2.h"

#include <optional>
#include <string>

namespace TransformGizmo {
enum class InteractionOp : uint8_t {
    AxisX,
    AxisY,
    AxisZ,
    YZ,
    ZX,
    XY,
    Screen,
    Trackball, // Rotate only
    Action, // Action-initiated (currently only for translate)
};

struct Interaction {
    TransformType Type;
    InteractionOp Op;

    bool operator==(const Interaction &) const = default;
};

// Local, non-snapped delta from interaction start.
// The Transform delta is derived from this and the start Transform.
struct LocalTransformDelta {
    vec3 P{0}, S{1};
    float RotationAngle{0};
    vec2 RotationYawPitch{0};
};

// Captured when an Interaction starts
struct StartContext {
    GizmoTransform Transform;
    vec2 MousePx;
    ray MouseRayWs;
    float WorldPerNdc; // World units per (signed) NDC at the gizmo origin (sampled along screen-x)
};

struct NumericInput {
    std::string Str; // Typed characters (digits and '.')
    bool Negate{false};

    bool Active() const { return !Str.empty() || Negate; }
    float Value() const { return (Str.empty() || Str == "." ? 0.f : std::stof(Str)) * (Negate ? -1.f : 1.f); }
    void Reset() {
        Str.clear();
        Negate = false;
    }
};
} // namespace TransformGizmo

// Live gizmo interaction state. Component on the viewport entity.
struct GizmoInteraction {
    // Cross-frame: persist for the duration of a drag.
    std::optional<TransformGizmo::Interaction> Current; // Hovered (no Start) or active (with Start) handle.
    std::optional<TransformGizmo::StartContext> Start; // Captured at drag start.
    TransformGizmo::NumericInput NumInput; // Numeric keyboard entry during a drag.

    // Interact -> Render handoff for the current frame.
    std::optional<GizmoTransform> RenderTransform; // Live gizmo pose (with drag P-override) to render; cleared by Render.
    TransformGizmo::LocalTransformDelta Delta;
    vec2 MousePx{}; // Mouse position Interact used; consumed by Render (recompute is wrong after wrap-delta reset).

    bool IsUsing() const { return Start.has_value(); }
};
