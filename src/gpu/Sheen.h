#pragma once

#include "gpu/TextureInfo.h"

struct Sheen {
    vec3 ColorFactor DEFAULT(0);
    float RoughnessFactor DEFAULT(0);
    TextureInfo ColorTexture DEFAULT();
    TextureInfo RoughnessTexture DEFAULT();
};
#ifndef __METAL_VERSION__
template<> inline constexpr field::SpecTable<Sheen> Specs<Sheen>{
    {&Sheen::ColorFactor, {.Widget = FieldWidget::Color, .Label = "Color", .Help = "Sheen color, multiplied by the color texture"}},
    {&Sheen::RoughnessFactor, {.Min = 0, .Max = 1, .Label = "Roughness", .Help = "Sheen roughness, multiplied by the roughness texture's alpha channel"}},
    {&Sheen::ColorTexture, {.Help = "sRGB sheen color"}},
    {&Sheen::RoughnessTexture, {.Help = "Sheen roughness in the alpha channel"}},
};
#endif
