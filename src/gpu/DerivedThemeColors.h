#pragma once

#include "gpu/Types.h"

// Viewport colors computed from the authored theme colors.
struct DerivedThemeColors {
    vec4 GridLine DEFAULT();
    vec4 GridEmphasis DEFAULT();
    vec3 GridAxisX DEFAULT();
    vec3 GridAxisZ DEFAULT();
    vec3 BoneActive DEFAULT();
    vec3 BoneActiveUnsel DEFAULT();
    vec3 BoneSelect DEFAULT();
    vec3 BonePoseActiveUnsel DEFAULT();
};
