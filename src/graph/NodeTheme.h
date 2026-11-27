#pragma once

#include "Field.h"
#include "ui/Rgba8.h"

namespace graph {
using numeric::vec4;

struct NodeThemeColors {
    vec4 Background{Rgba8(30, 30, 33)};
    vec4 Grid{Rgba8(255, 255, 255, 12)};
    vec4 CoarseGrid{Rgba8(255, 255, 255, 28)};
    vec4 Node{Rgba8(48, 48, 52, 240)};
    vec4 Tray{Rgba8(34, 34, 38)};
    vec4 Title{Rgba8(235, 235, 235)};
    vec4 Label{Rgba8(200, 200, 200)};
    vec4 ParameterLabel{Rgba8(170, 170, 170)};
    vec4 Border{Rgba8(15, 15, 15)};
    vec4 Selected{Rgba8(240, 150, 40)};
    vec4 Active{Rgba8(255, 255, 255)};
    vec4 Error{Rgba8(230, 60, 50)};
    vec4 Warning{Rgba8(230, 170, 40)}; // A node whose fault leaves its cook unchanged
    vec4 Outline{Rgba8(0, 0, 0)};
    vec4 Hovered{Rgba8(255, 255, 255)};
    vec4 WireHighlight{Rgba8(255, 255, 255)};
    vec4 BoxSelect{Rgba8(100, 140, 220, 40)};
    vec4 BoxSelectBorder{Rgba8(100, 140, 220, 160)};
    vec4 Dimmed{Rgba8(74, 74, 78)}; // A socket a held wire cannot link with
    vec4 DimmedLabel{Rgba8(110, 110, 110)};
    vec4 Chevron{Rgba8(130, 130, 130)};
    vec4 ChevronLinked{Rgba8(220, 220, 220)};

    vec4 SourcesTitle{Rgba8(120, 60, 60)};
    vec4 OperatorsTitle{Rgba8(40, 100, 90)};
    vec4 CombineTitle{Rgba8(40, 90, 110)};
    vec4 AttributesTitle{Rgba8(80, 60, 130)};
    vec4 MathTitle{Rgba8(53, 96, 131)};
    vec4 GeneratorsTitle{Rgba8(50, 110, 60)};
    vec4 InstancesTitle{Rgba8(40, 80, 140)};
    vec4 DisplayedTitle{Rgba8(170, 80, 70)}; // The title of the displayed node

    vec4 GeometrySocket{Rgba8(0, 214, 163)};
    vec4 InstancesSocket{Rgba8(59, 140, 220)};
    vec4 ImageSocket{Rgba8(99, 56, 99)};
    vec4 FloatSocket{Rgba8(222, 170, 80)};
    vec4 IntegerSocket{Rgba8(89, 140, 92)};
    vec4 BooleanSocket{Rgba8(204, 166, 214)};
    vec4 VectorSocket{Rgba8(99, 99, 199)};
    vec4 Vector4Socket{Rgba8(160, 150, 230)};
    vec4 ColorSocket{Rgba8(199, 199, 41)};
    vec4 MatrixSocket{Rgba8(200, 120, 160)};
    vec4 MixedSocket{Rgba8(150, 150, 190)};
};

// A node grid cell's side.
struct CellSize {
    bool FollowFrameHeight{true};
    float Points{21};
    bool operator==(const CellSize &) const = default;
};

struct NodeThemeSizes {
    CellSize Cell{};
    uint8_t CoarseGridCells{8};
    uint8_t MaxLabelCells{10};
    float TitlePadding{6}, LabelSpacing{4}, ParameterIndent{6};
    float SocketRadius{5}, Rounding{4}, Margin{2};
    float WireWidth{2};
    float WireCurving{.4f};
    float BorderWidth{1}, HighlightBorderWidth{2}, OutlineWidth{1};
    bool operator==(const NodeThemeSizes &) const = default;
};

struct NodeTheme {
    NodeThemeColors Colors{};
    NodeThemeSizes Sizes{};
};
} // namespace graph

template<> inline constexpr field::SpecTable<graph::CellSize> Specs<graph::CellSize>{
    {&graph::CellSize::FollowFrameHeight, {.Label = "Frame height", .Help = "Make each cell one frame height, ignoring the grid size"}},
    {&graph::CellSize::Points, {.Min = 8, .Max = 128, .Unit = FieldUnit::Points, .Label = "Grid size", .Help = "Cell side length"}},
};
template<> inline constexpr field::SpecTable<graph::NodeThemeSizes> Specs<graph::NodeThemeSizes>{
    {&graph::NodeThemeSizes::CoarseGridCells, {.Min = 2, .Max = 64, .Unit = FieldUnit::Cells, .Help = "Number of cells between coarse grid lines"}},
    {&graph::NodeThemeSizes::MaxLabelCells, {.Min = 2, .Max = 64, .Unit = FieldUnit::Cells, .Help = "Maximum socket label width. Longer labels end in an ellipsis."}},
    {&graph::NodeThemeSizes::TitlePadding, {.Min = 0, .Max = 24, .Unit = FieldUnit::Points, .Help = "Space on each side of a node title"}},
    {&graph::NodeThemeSizes::LabelSpacing, {.Min = 0, .Max = 24, .Unit = FieldUnit::Points, .Help = "Space between a socket and its label"}},
    {&graph::NodeThemeSizes::ParameterIndent, {.Min = 0, .Max = 24, .Unit = FieldUnit::Points, .Help = "Extra indent of parameter socket labels"}},
    {&graph::NodeThemeSizes::SocketRadius, {.Min = 2, .Max = 12, .Unit = FieldUnit::Points, .Help = "Socket radius"}},
    {&graph::NodeThemeSizes::Rounding, {.Min = 0, .Max = 12, .Unit = FieldUnit::Points, .Help = "Node corner radius"}},
    {&graph::NodeThemeSizes::Margin, {.Min = 0, .Max = 8, .Unit = FieldUnit::Points, .Help = "Space between a node and the edges of its cells, which separates nodes that touch"}},
    {&graph::NodeThemeSizes::WireWidth, {.Min = .5, .Max = 8, .Unit = FieldUnit::Points, .Help = "Link wire width"}},
    {&graph::NodeThemeSizes::WireCurving, {.Min = 0, .Max = 1, .Help = "Wire curve handle length as a fraction of the link's horizontal span"}},
    {&graph::NodeThemeSizes::BorderWidth, {.Min = .5, .Max = 4, .Unit = FieldUnit::Points, .Help = "Node border width"}},
    {&graph::NodeThemeSizes::HighlightBorderWidth, {.Min = .5, .Max = 6, .Unit = FieldUnit::Points, .Help = "Border width of a selected, active, or failing node"}},
    {&graph::NodeThemeSizes::OutlineWidth, {.Min = 0, .Max = 4, .Unit = FieldUnit::Points, .Help = "Outline width of sockets, chevrons, and the parameter tray"}},
};
template<> inline constexpr field::SpecTable<graph::NodeTheme> Specs<graph::NodeTheme>{
    {&graph::NodeTheme::Colors, {.Widget = FieldWidget::Color}},
};
