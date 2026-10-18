#pragma once

#include "Field.h"

// Editor chrome sizes shared across windows.
struct InterfaceTheme {
    float IconButtonRounding{4};
    numeric::vec2 ToolButtonSize{36, 30}, HeaderButtonSize{27, 22.5};
};
template<> inline constexpr field::SpecTable<InterfaceTheme> Specs<InterfaceTheme>{
    {&InterfaceTheme::IconButtonRounding, {.Min = 0, .Max = 12, .Unit = FieldUnit::Points, .Help = "Corner radius of icon button groups"}},
    {&InterfaceTheme::ToolButtonSize, {.Min = 16, .Max = 64, .Unit = FieldUnit::Points, .Help = "Width and height of viewport tool buttons"}},
    {&InterfaceTheme::HeaderButtonSize, {.Min = 16, .Max = 64, .Unit = FieldUnit::Points, .Help = "Width and height of viewport header buttons"}},
};
