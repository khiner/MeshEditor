// Started with https://github.com/CedricGuillemet/ModelGizmo and modified/simplified heavily.

#pragma once

#include "numeric/mat4.h"
#include "numeric/ray.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"

#include <optional>
#include <string_view>

namespace ModelGizmo {
enum class TransformType : uint8_t {
    NoOp = 0,
    Translate,
    Rotate,
    Scale,
    Universal,
};

enum class Mode {
    Local, // Align to object’s orientation
    World // Align to global axes (no rotation)
};

bool IsActive();
std::string_view ToString();

bool Draw(Mode, TransformType, vec2 pos, vec2 size, vec2 mouse_pos, ray mouse_ray, mat4 &m, const mat4 &view, const mat4 &proj, std::optional<vec3> snap = std::nullopt);
} // namespace ModelGizmo
