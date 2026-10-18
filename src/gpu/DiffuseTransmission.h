#pragma once

#include "gpu/TextureInfo.h"

struct DiffuseTransmission {
    float Factor DEFAULT(0);
    vec3 ColorFactor DEFAULT(1);
    TextureInfo Texture DEFAULT();
    TextureInfo ColorTexture DEFAULT();
};
#ifndef __METAL_VERSION__
template<> inline constexpr field::SpecTable<DiffuseTransmission> Specs<DiffuseTransmission>{
    {&DiffuseTransmission::Factor, {.Min = 0, .Max = 1, .Help = "Fraction of light diffusely transmitted through the surface, multiplied by the texture's alpha channel"}},
    {&DiffuseTransmission::ColorFactor, {.Widget = FieldWidget::Color, .Label = "Color", .Help = "Color of the transmitted light, multiplied by the color texture"}},
    {&DiffuseTransmission::Texture, {.Help = "Diffuse transmission fraction in the alpha channel"}},
    {&DiffuseTransmission::ColorTexture, {.Help = "sRGB color of the transmitted light"}},
};
#endif
