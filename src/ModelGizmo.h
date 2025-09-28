// Started with https://github.com/CedricGuillemet/ModelGizmo and modified/simplified heavily.

#pragma once

#include "numeric/mat3.h"
#include "numeric/mat4.h"
#include "numeric/quat.h"
#include "numeric/ray.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"

#include <optional>
#include <string_view>

namespace ModelGizmo {
enum class Type : uint8_t {
    Translate,
    Rotate,
    Scale,
    Universal,
};

enum class Mode : uint8_t {
    Local, // Align to object’s orientation
    World // Align to global axes (no rotation)
};

struct Config {
    Type Type{};
    bool Snap{false}; // Snap translate and scale
    vec3 SnapValue{0.5f};
};

struct Model {
    vec3 P; // Position
    quat R; // Rotation
    vec3 S; // Scale
    Mode Mode; // Local/World

    vec3 AxisDirWs(uint32_t i) const { return Mode == Mode::World ? I3[i] : R * I3[i]; }
    vec3 LocalDirToWorld(vec3 d_local, bool apply_scale = false) const {
        if (apply_scale) d_local *= S;
        return Mode == Mode::World ? d_local : R * d_local;
    }
    vec3 WorldDirToLocal(vec3 d_ws) const { return Mode == Mode::World ? d_ws : glm::conjugate(R) * d_ws; }
};

bool IsUsing();
std::string_view ToString();

bool Draw(Model &, Config, const mat4 &view, const mat4 &proj, vec2 pos, vec2 size, vec2 mouse_px, ray mouse_ray_ws);
} // namespace ModelGizmo
