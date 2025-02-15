// Started with https://github.com/CedricGuillemet/ModelGizmo and modified/simplified heavily.

#pragma once

#include "numeric/mat4.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"

#include <optional>
#include <string_view>

namespace ModelGizmo {
enum class Op {
    NoOp = 0,

    AxisX = 1 << 0,
    AxisY = 1 << 1,
    AxisZ = 1 << 2,
    Screen = 1 << 3,

    Translate = 1 << 4,
    Rotate = 1 << 5,
    Scale = 1 << 6,
    Universal = Translate | Rotate | Scale,

    TranslateYZ = Translate | AxisY | AxisZ,
    TranslateZX = Translate | AxisZ | AxisX,
    TranslateXY = Translate | AxisX | AxisY,
    TranslateScreen = Translate | Screen,

    RotateScreen = Rotate | Screen,

    ScaleXYZ = Scale | AxisX | AxisY | AxisZ,
};

enum Mode {
    Local,
    World
};

bool IsActive();
Op CurrentOp(); // Hovered or active operation.
std::string_view ToString(Op);

bool Draw(vec2 pos, vec2 size, const mat4 &view, const mat4 &proj, Op, Mode, mat4 &m, std::optional<vec3> snap = std::nullopt);
} // namespace ModelGizmo
