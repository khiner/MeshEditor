#pragma once

#include "ui/Rgba8.h"

struct TimelineThemeColors {
    numeric::vec4 Background{Rgba8(30, 30, 30)};
    numeric::vec4 Range{Rgba8(50, 50, 55)};
    numeric::vec4 RulerLine{Rgba8(60, 60, 60)};
    numeric::vec4 MajorTick{Rgba8(80, 80, 80)};
    numeric::vec4 MinorTick{Rgba8(55, 55, 55)};
    numeric::vec4 TickLabel{Rgba8(180, 180, 180)};
    numeric::vec4 Keyframe{Rgba8(230, 230, 230)};
    numeric::vec4 KeyframeOutline{Rgba8(20, 20, 20)};
    numeric::vec4 Playhead{Rgba8(100, 160, 255, 200)};
    numeric::vec4 PlayheadLabel{Rgba8(255, 255, 255)};
    numeric::vec4 PlayheadLabelBackground{Rgba8(80, 130, 200, 200)};
};

struct TimelineTheme {
    TimelineThemeColors Colors{};
    float RulerHeight{20}, TickLabelSpacing{80}, MinorTickSpacing{32};
    float KeyframeRadius{5}, KeyframePadding{4};
    float PlayheadWidth{2}, PlayheadLabelRounding{3}, PlayheadLabelPaddingX{3}, PlayheadLabelPaddingY{1};
};
