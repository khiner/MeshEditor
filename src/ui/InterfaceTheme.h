#pragma once

#include "Field.h"

// Editor chrome sizes shared across windows.
struct InterfaceTheme {
    float IconButtonRounding{4};
    numeric::vec2 ToolButtonSize{36, 30}, HeaderButtonSize{27, 22.5};
};
template<> inline constexpr field::SpecTable<InterfaceTheme> Specs<InterfaceTheme>{
    {&InterfaceTheme::IconButtonRounding, {.Min = 0, .Max = 12, .Digits = 1}},
    {&InterfaceTheme::ToolButtonSize, {.Min = 16, .Max = 64, .Digits = 1}},
    {&InterfaceTheme::HeaderButtonSize, {.Min = 16, .Max = 64, .Digits = 1}},
};
