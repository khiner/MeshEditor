#pragma once

#include "gpu/Types.h"

struct AxisThemeColors {
    vec3 X DEFAULT();
    vec3 Y DEFAULT();
    vec3 Z DEFAULT();
};
static_assert(sizeof(AxisThemeColors) == 36, "AxisThemeColors size");
