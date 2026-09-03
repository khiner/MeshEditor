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
    Trackball,
    Action,
};

struct Interaction {
    TransformType Type;
    InteractionOp Op;

    bool operator==(const Interaction &) const = default;
};

// Stores the local, unsnapped delta from the interaction start.
struct LocalTransformDelta {
    vec3 P{0}, S{1};
    float RotationAngle{0};
    vec2 RotationYawPitch{0};
};

struct StartContext {
    GizmoTransform Transform;
    vec2 MousePx;
    ray MouseRayWs;
    float WorldPerNdc; // World units per (signed) NDC at the gizmo origin (sampled along screen-x)
};

struct NumericInput {
    std::string Str;
    bool Negate{false};

    bool Active() const { return !Str.empty() || Negate; }
    float Value() const { return (Str.empty() || Str == "." ? 0.f : std::stof(Str)) * (Negate ? -1.f : 1.f); }
    void Reset() {
        Str.clear();
        Negate = false;
    }
};
} // namespace TransformGizmo

struct GizmoInteraction {
    std::optional<TransformGizmo::Interaction> Current;
    std::optional<TransformGizmo::StartContext> Start;
    TransformGizmo::NumericInput NumInput;

    // Interact writes these fields for Render and Render clears RenderTransform.
    std::optional<GizmoTransform> RenderTransform;
    TransformGizmo::LocalTransformDelta Delta;
    // Preserve the position used by Interact because pointer wrapping resets the mouse delta before Render.
    vec2 MousePx{};

    bool IsUsing() const { return Start.has_value(); }
};
