#pragma once

#include "numeric/mat3.h"
#include "numeric/quat.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"

#include <optional>
#include <string_view>

struct Camera;

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
    vec3 SnapValue{0.5};
    Type Type{};
    bool Snap{false};
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

bool Draw(Model &, Config, const Camera &, vec2 pos, vec2 size, vec2 mouse_px);
} // namespace ModelGizmo
