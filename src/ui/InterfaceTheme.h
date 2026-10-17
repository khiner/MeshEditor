#pragma once

#include "Field.h"

// Editor chrome sizes shared across windows.
struct InterfaceTheme {
    float IconButtonRounding{4};
};
template<> inline constexpr field::SpecTable<InterfaceTheme> Specs<InterfaceTheme>{{&InterfaceTheme::IconButtonRounding, {.Min = 0, .Max = 12, .Digits = 1}}};
