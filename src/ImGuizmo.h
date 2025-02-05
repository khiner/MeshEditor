// Started with https://github.com/CedricGuillemet/ImGuizmo and modified/simplified heavily.

#pragma once

#include "numeric/vec2.h"
#include "numeric/mat4.h"

namespace ImGuizmo {
enum class Operation {
    NoOperation = 0,
    TranslateX = 1u << 0,
    TranslateY = 1u << 1,
    TranslateZ = 1u << 2,
    RotateX = 1u << 3,
    RotateY = 1u << 4,
    RotateZ = 1u << 5,
    RotateScreen = 1u << 6,
    ScaleX = 1u << 7,
    ScaleY = 1u << 8,
    ScaleZ = 1u << 9,
    ScaleXU = 1u << 10,
    ScaleYU = 1u << 11,
    ScaleZU = 1u << 12,

    Translate = TranslateX | TranslateY | TranslateZ,
    Rotate = RotateX | RotateY | RotateZ | RotateScreen,
    Scale = ScaleX | ScaleY | ScaleZ,
    ScaleU = ScaleXU | ScaleYU | ScaleZU, // universal
    Universal = Translate | Rotate | ScaleU
};

enum Mode {
    Local,
    World
};

// Is mouse cursor over any gizmo control (axis, plan or screen component)
bool IsOver();
// Is cursor over the operation's gizmo
bool IsOver(Operation);
// Is gizmo actively being used
bool IsUsing();

void SetRect(vec2 pos, vec2 size);
bool Manipulate(const mat4 &view, const mat4 &proj, Operation, Mode, mat4 &m, const float *snap = nullptr);
} // namespace ImGuizmo
