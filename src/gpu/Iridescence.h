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
    {&Iridescence::Factor, {.Min = 0, .Max = 1, .Help = "Iridescence intensity, multiplied by the texture's red channel"}},
    {&Iridescence::Ior, {.Min = 1, .Max = 5, .Label = "IOR", .Help = "Index of refraction of the thin film"}},
    {&Iridescence::ThicknessMinimum, {.Min = 0, .Max = 1000, .Unit = FieldUnit::Nanometers, .Help = "Film thickness where the thickness texture's green channel is 0"}},
    {&Iridescence::ThicknessMaximum, {.Min = 0, .Max = 1000, .Unit = FieldUnit::Nanometers, .Help = "Film thickness where the thickness texture's green channel is 1, or everywhere without a texture"}},
    {&Iridescence::Texture, {.Help = "Iridescence intensity in the red channel"}},
    {&Iridescence::ThicknessTexture, {.Help = "Film thickness in the green channel, mapped from the minimum to the maximum"}},
};
#endif
