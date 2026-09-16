#pragma once

#include "gpu/AxisThemeColors.h"
#include "gpu/Types.h"
#include "gpu/ViewportThemeColors.h"

struct ViewportTheme {
    ViewportThemeColors Colors DEFAULT();
    AxisThemeColors AxisColors DEFAULT();
    float EdgeWidth DEFAULT();
    uint32_t SilhouetteEdgeWidth DEFAULT();
};
static_assert(sizeof(ViewportTheme) == 424, "ViewportTheme size");
