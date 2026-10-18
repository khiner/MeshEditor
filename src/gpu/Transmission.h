#pragma once

#include "gpu/TextureInfo.h"

struct Transmission {
    float Factor DEFAULT(0);
    TextureInfo Texture DEFAULT();
};
#ifndef __METAL_VERSION__
template<> inline constexpr field::SpecTable<Transmission> Specs<Transmission>{
    {&Transmission::Factor, {.Min = 0, .Max = 1, .Help = "Fraction of light transmitted through the surface, multiplied by the texture's red channel"}},
    {&Transmission::Texture, {.Help = "Transmission fraction in the red channel"}},
};
#endif
