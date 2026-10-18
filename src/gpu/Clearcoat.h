#pragma once

#include "gpu/TextureInfo.h"

struct Clearcoat {
    float Factor DEFAULT(0);
    float RoughnessFactor DEFAULT(0);
    float NormalScale DEFAULT(1);
    TextureInfo Texture DEFAULT();
    TextureInfo RoughnessTexture DEFAULT();
    TextureInfo NormalTexture DEFAULT();
};
#ifndef __METAL_VERSION__
template<> inline constexpr field::SpecTable<Clearcoat> Specs<Clearcoat>{
    {&Clearcoat::Factor, {.Min = 0, .Max = 1, .Help = "Clearcoat layer intensity, multiplied by the texture's red channel"}},
    {&Clearcoat::RoughnessFactor, {.Min = 0, .Max = 1, .Label = "Roughness", .Help = "Clearcoat layer roughness, multiplied by the roughness texture's green channel"}},
    {&Clearcoat::NormalScale, {.Min = -2, .Max = 2, .Help = "Multiplier on the sampled clearcoat normal's X and Y"}},
    {&Clearcoat::Texture, {.Help = "Clearcoat layer intensity in the red channel"}},
    {&Clearcoat::RoughnessTexture, {.Help = "Clearcoat layer roughness in the green channel"}},
    {&Clearcoat::NormalTexture, {.Help = "Tangent-space normal map of the clearcoat layer"}},
};
#endif
