#pragma once

#include "Transform.h"
#include "numeric/mat3.h"
#include "numeric/vec2.h"

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
std::optional<Result> Draw(const GizmoTransform &, Config, const Camera &, vec2 pos, vec2 size, vec2 mouse_px);
} // namespace TransformGizmo
