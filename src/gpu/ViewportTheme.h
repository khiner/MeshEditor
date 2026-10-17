#pragma once

#include "gpu/AxisThemeColors.h"
#include "gpu/ViewportThemeColors.h"

struct ViewportTheme {
    ViewportThemeColors Colors DEFAULT();
    AxisThemeColors AxisColors DEFAULT();
    float EdgeWidth DEFAULT();
    uint32_t SilhouetteEdgeWidth DEFAULT();
};
#ifndef __METAL_VERSION__
#include "Field.h"
template<> inline constexpr field::SpecTable<ViewportTheme> Specs<ViewportTheme>{
    {&ViewportTheme::EdgeWidth, {.Min = .25, .Max = 2}},
    {&ViewportTheme::SilhouetteEdgeWidth, {.Min = 1, .Max = 4}},
};
#endif
