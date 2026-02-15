#pragma once

#include "Transform.h"
#include "numeric/mat3.h"
#include "numeric/rect.h"

#include <optional>
#include <string_view>

struct Camera;

namespace TransformGizmo {
enum class Mode : uint8_t {
    Local, // Align to object’s orientation
    World // Align to global axes (no rotation)
};
} // namespace TransformGizmo

struct GizmoTransform : Transform {
    TransformGizmo::Mode Mode; // Local/World

    using enum TransformGizmo::Mode;

    vec3 AxisDirWs(uint32_t i) const { return Mode == World ? I3[i] : R * I3[i]; }
    vec3 LocalDirToWorld(vec3 d_local, bool apply_scale = false) const {
        if (apply_scale) d_local *= S;
        return Mode == World ? d_local : R * d_local;
    }
    vec3 WorldDirToLocal(vec3 d_ws) const { return Mode == World ? d_ws : glm::conjugate(R) * d_ws; }
};

namespace TransformGizmo {
enum class Type : uint8_t {
    None,
    Translate,
    Rotate,
    Scale,
    Universal,
};

// Subset `Type` without `Universal`. (need better names)
enum class TransformType : uint8_t {
    Translate,
    Rotate,
    Scale,
};

struct Config {
    Type Type{};
    vec3 SnapValue{0.5};
    bool Snap{false};
};

bool IsUsing();
std::string_view ToString();

struct Result {
    Transform Start; // Transform at interaction start
    Transform Delta; // Delta transform since interaction start
};

// Processes interaction (hover, click, drag) and returns the transform delta if actively dragging.
// Does NOT render — call Render() afterward with the (potentially updated) transform.
std::optional<Result> Interact(const GizmoTransform &, Config, const Camera &, rect viewport, vec2 mouse_px, std::optional<TransformType> start_screen_transform = {});

// Renders the gizmo using the interaction delta from the last Interact() call.
// Call with the post-delta transform so the gizmo visual matches the applied transform.
void Render(const GizmoTransform &, Type, const Camera &, rect viewport);
} // namespace TransformGizmo
