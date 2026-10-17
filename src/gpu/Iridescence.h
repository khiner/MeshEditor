#pragma once

#include "gpu/TextureInfo.h"

struct Iridescence {
    float Factor DEFAULT(0);
    float Ior DEFAULT(1.3);
    float ThicknessMinimum DEFAULT(100);
    float ThicknessMaximum DEFAULT(400);
    TextureInfo Texture DEFAULT();
    TextureInfo ThicknessTexture DEFAULT();
};
#ifndef __METAL_VERSION__
template<> inline constexpr field::SpecTable<Iridescence> Specs<Iridescence>{
    {&Iridescence::Factor, {.Min = 0, .Max = 1}},
    {&Iridescence::Ior, {.Min = 1, .Max = 5}},
    {&Iridescence::ThicknessMinimum, {.Min = 0, .Max = 1000, .Digits = 0}},
    {&Iridescence::ThicknessMaximum, {.Min = 0, .Max = 1000, .Digits = 0}},
};
#endif
