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
    {&Clearcoat::Factor, {.Min = 0, .Max = 1}},
    {&Clearcoat::RoughnessFactor, {.Min = 0, .Max = 1}},
    {&Clearcoat::NormalScale, {.Min = -2, .Max = 2}},
};
#endif
