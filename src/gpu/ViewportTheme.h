#pragma once

#include "gpu/AxisThemeColors.h"
#include "gpu/DerivedThemeColors.h"
#include "gpu/ViewportThemeColors.h"

// The viewport theme's GPU record, with its derived colors.
struct ViewportThemeUbo {
    ViewportThemeColors Colors DEFAULT();
    DerivedThemeColors Derived DEFAULT();
    AxisThemeColors AxisColors DEFAULT();
    float EdgeWidth DEFAULT(); // Half width in framebuffer pixels
    uint32_t SilhouetteEdgeWidth DEFAULT();
};

#ifndef __METAL_VERSION__
#include "Field.h"

struct ViewportThemeSizes {
    float EdgeWidth{1};
    uint32_t SilhouetteEdgeWidth{2};
};
template<> inline constexpr field::SpecTable<ViewportThemeSizes> Specs<ViewportThemeSizes>{
    {&ViewportThemeSizes::EdgeWidth, {.Min = .5, .Max = 4, .Unit = FieldUnit::Pixels, .Help = "Width of wireframe, mesh edge, and line overlays"}},
    {&ViewportThemeSizes::SilhouetteEdgeWidth, {.Min = 1, .Max = 4, .Unit = FieldUnit::Pixels, .Help = "Width of the outline around selected objects"}},
};

struct ViewportTheme {
    ViewportThemeColors Colors{};
    AxisThemeColors AxisColors{};
    ViewportThemeSizes Sizes{};
};
template<> inline constexpr field::SpecTable<ViewportTheme> Specs<ViewportTheme>{
    {&ViewportTheme::Colors, {.Widget = FieldWidget::Color}},
    {&ViewportTheme::AxisColors, {.Widget = FieldWidget::Color}},
};
#endif
