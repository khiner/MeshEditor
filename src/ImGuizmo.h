// Started with https://github.com/CedricGuillemet/ImGuizmo and modified/simplified heavily.

#pragma once

#include "numeric/mat4.h"

namespace ImGuizmo {
// call it when you want a gizmo
// Needs view and projection matrices.
// matrix parameter is the source matrix (where will be gizmo be drawn) and might be transformed by the function. Return deltaMatrix is optional
// translation is applied in world space
enum Operation {
    TRANSLATE_X = (1u << 0),
    TRANSLATE_Y = (1u << 1),
    TRANSLATE_Z = (1u << 2),
    ROTATE_X = (1u << 3),
    ROTATE_Y = (1u << 4),
    ROTATE_Z = (1u << 5),
    ROTATE_SCREEN = (1u << 6),
    SCALE_X = (1u << 7),
    SCALE_Y = (1u << 8),
    SCALE_Z = (1u << 9),
    BOUNDS = (1u << 10),
    SCALE_XU = (1u << 11),
    SCALE_YU = (1u << 12),
    SCALE_ZU = (1u << 13),

    TRANSLATE = TRANSLATE_X | TRANSLATE_Y | TRANSLATE_Z,
    ROTATE = ROTATE_X | ROTATE_Y | ROTATE_Z | ROTATE_SCREEN,
    SCALE = SCALE_X | SCALE_Y | SCALE_Z,
    SCALEU = SCALE_XU | SCALE_YU | SCALE_ZU, // universal
    UNIVERSAL = TRANSLATE | ROTATE | SCALEU
};

enum MODE {
    LOCAL,
    WORLD
};

// return true if mouse cursor is over any gizmo control (axis, plan or screen component)
bool IsOver();
// return true if the cursor is over the operation's gizmo
bool IsOver(Operation);
// return true if mouse IsOver or if the gizmo is in moving state
bool IsUsing();

void SetRect(float x, float y, float width, float height);
bool Manipulate(const mat4 &view, const mat4 &projection, Operation, MODE, mat4 &matrix, const float *snap = nullptr);

enum COLOR {
    DIRECTION_X, // directionColor[0]
    DIRECTION_Y, // directionColor[1]
    DIRECTION_Z, // directionColor[2]
    PLANE_X, // planeColor[0]
    PLANE_Y, // planeColor[1]
    PLANE_Z, // planeColor[2]
    SELECTION, // selectionColor
    INACTIVE, // inactiveColor
    TRANSLATION_LINE, // translationLineColor
    SCALE_LINE,
    ROTATION_USING_BORDER,
    ROTATION_USING_FILL,
    HATCHED_AXIS_LINES,
    TEXT,
    TEXT_SHADOW,
    COUNT
};
} // namespace ImGuizmo
