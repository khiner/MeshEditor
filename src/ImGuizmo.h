// Started with https://github.com/CedricGuillemet/ImGuizmo and modified/simplified heavily.

#pragma once

#include "numeric/mat4.h"
#include "numeric/vec2.h"

#include <string_view>

namespace ImGuizmo {
enum class Op {
    NoOp = 0,

    AxisX = 1 << 0,
    AxisY = 1 << 1,
    AxisZ = 1 << 2,
    Screen = 1 << 3,

    Translate = 1 << 4,
    Rotate = 1 << 5,
    Scale = 1 << 6,
    ScaleU = 1 << 7, // Universal scale is a different control.
    Universal = Translate | Rotate | ScaleU,

    TranslateYZ = Translate | AxisY | AxisZ,
    TranslateZX = Translate | AxisZ | AxisX,
    TranslateXY = Translate | AxisX | AxisY,
    TranslateScreen = Translate | Screen,

    RotateScreen = Rotate | Screen,

    ScaleXYZ = Scale | AxisX | AxisY | AxisZ,
    ScaleUXYZ = ScaleU | AxisX | AxisY | AxisZ,
};

enum Mode {
    Local,
    World
};

Op HoverOp();
Op UsingOp();
std::string_view ToString(Op);

bool Manipulate(vec2 pos, vec2 size, const mat4 &view, const mat4 &proj, Op, Mode, mat4 &m, const float *snap = nullptr);
} // namespace ImGuizmo
