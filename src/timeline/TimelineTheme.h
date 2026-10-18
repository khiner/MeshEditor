#pragma once

#include "Field.h"
#include "ui/Rgba8.h"

struct TimelineThemeColors {
    numeric::vec4 Background{Rgba8(30, 30, 30)};
    numeric::vec4 FrameRange{Rgba8(50, 50, 55)};
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

struct TimelineThemeSizes {
    float RulerHeight{20}, TickLabelSpacing{80}, MinorTickSpacing{32};
    float KeyframeRadius{5}, KeyframePadding{4};
    float PlayheadWidth{2}, PlayheadLabelRounding{3};
    numeric::vec2 PlayheadLabelPadding{3, 1};
};
template<> inline constexpr field::SpecTable<TimelineThemeSizes> Specs<TimelineThemeSizes>{
    {&TimelineThemeSizes::RulerHeight, {.Min = 12, .Max = 48, .Unit = FieldUnit::Points, .Help = "Height of the frame number ruler"}},
    {&TimelineThemeSizes::TickLabelSpacing, {.Min = 20, .Max = 200, .Unit = FieldUnit::Points, .Help = "Minimum space between labeled ticks"}},
    {&TimelineThemeSizes::MinorTickSpacing, {.Min = 8, .Max = 128, .Unit = FieldUnit::Points, .Help = "Minimum labeled tick spacing that shows unlabeled ticks between them"}},
    {&TimelineThemeSizes::KeyframeRadius, {.Min = 2, .Max = 12, .Unit = FieldUnit::Points, .Help = "Keyframe marker radius"}},
    {&TimelineThemeSizes::KeyframePadding, {.Min = 0, .Max = 24, .Unit = FieldUnit::Points, .Help = "Space between the ruler and the keyframe markers"}},
    {&TimelineThemeSizes::PlayheadWidth, {.Min = .5, .Max = 8, .Unit = FieldUnit::Points, .Help = "Playhead line width"}},
    {&TimelineThemeSizes::PlayheadLabelRounding, {.Min = 0, .Max = 12, .Unit = FieldUnit::Points, .Help = "Corner radius of the playhead's frame label"}},
    {&TimelineThemeSizes::PlayheadLabelPadding, {.Min = 0, .Max = 12, .Unit = FieldUnit::Points, .Help = "Horizontal and vertical padding around the playhead's frame label"}},
};

struct TimelineTheme {
    TimelineThemeColors Colors{};
    TimelineThemeSizes Sizes{};
};
template<> inline constexpr field::SpecTable<TimelineTheme> Specs<TimelineTheme>{
    {&TimelineTheme::Colors, {.Widget = FieldWidget::Color}},
};
