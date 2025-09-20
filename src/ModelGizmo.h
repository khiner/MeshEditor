// Started with https://github.com/CedricGuillemet/ModelGizmo and modified/simplified heavily.

#pragma once

#include "numeric/mat4.h"
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
    Mode Mode{};
    Type Type{};
    bool Snap{false}; // Snap translate and scale
    vec3 SnapValue{0.5f};
};

bool IsUsing();
std::string_view ToString();

bool Draw(Config, mat4 &m, const mat4 &view, const mat4 &proj, vec2 pos, vec2 size, vec2 mouse_px, ray mouse_ray_ws);
} // namespace ModelGizmo
