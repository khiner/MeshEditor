#pragma once

#include "ui/Rgba8.h"

namespace procedure {
using numeric::vec4;

struct NodeThemeColors {
    vec4 Background{Rgba8(30, 30, 33)};
    vec4 Grid{Rgba8(255, 255, 255, 12)};
    vec4 Node{Rgba8(48, 48, 52, 240)};
    vec4 Tray{Rgba8(34, 34, 38)};
    vec4 Title{Rgba8(235, 235, 235)};
    vec4 Label{Rgba8(200, 200, 200)};
    vec4 ParameterLabel{Rgba8(170, 170, 170)};
    vec4 Border{Rgba8(15, 15, 15)};
    vec4 Selected{Rgba8(240, 150, 40)};
    vec4 Active{Rgba8(255, 255, 255)};
    vec4 Error{Rgba8(230, 60, 50)};
    vec4 Outline{Rgba8(0, 0, 0)};
    vec4 Hovered{Rgba8(255, 255, 255)};
    vec4 WireHighlight{Rgba8(255, 255, 255)};
    vec4 InsertMarker{Rgba8(255, 255, 255, 80)};
    vec4 BoxSelect{Rgba8(100, 140, 220, 40)};
    vec4 BoxSelectBorder{Rgba8(100, 140, 220, 160)};
    vec4 Dimmed{Rgba8(74, 74, 78)}; // A socket a held wire cannot link with
    vec4 DimmedLabel{Rgba8(110, 110, 110)};
    vec4 Chevron{Rgba8(130, 130, 130)};
    vec4 ChevronLinked{Rgba8(220, 220, 220)};

    vec4 SourceOutputTitle{Rgba8(120, 60, 60)};
    vec4 OperatorsTitle{Rgba8(40, 100, 90)};
    vec4 CombineTitle{Rgba8(40, 90, 110)};
    vec4 AttributesTitle{Rgba8(80, 60, 130)};
    vec4 ValuesTitle{Rgba8(90, 90, 90)};
    vec4 GeneratorsTitle{Rgba8(50, 110, 60)};
    vec4 InstancesTitle{Rgba8(40, 80, 140)};
    vec4 OutputNodeTitle{Rgba8(170, 80, 70)};

    vec4 GeometrySocket{Rgba8(0, 214, 163)};
    vec4 InstancesSocket{Rgba8(59, 140, 220)};
    vec4 FloatSocket{Rgba8(222, 170, 80)};
    vec4 IntegerSocket{Rgba8(89, 140, 92)};
    vec4 BooleanSocket{Rgba8(204, 166, 214)};
    vec4 VectorSocket{Rgba8(99, 99, 199)};
    vec4 RotationSocket{Rgba8(160, 150, 230)};
    vec4 MixedSocket{Rgba8(150, 150, 190)};
};

struct NodeTheme {
    NodeThemeColors Colors{};
    float NodeWidth{180}, TitleHeight{24}, RowHeight{22}, NodePadding{6}, SocketRadius{5}, Rounding{4};
    float TitlePadding{6}, LabelSpacing{4}, ParameterIndent{6};
    float WireWidth{2}, GridSpacing{32};
    float BorderWidth{1}, HighlightBorderWidth{2}, OutlineWidth{1};
    float WireCurving{.4f}; // Handles reach this fraction of the horizontal span.
};
} // namespace procedure
