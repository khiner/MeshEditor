#pragma once

#include "gpu/TextureInfo.h"

struct Sheen {
    vec3 ColorFactor DEFAULT(0);
    float RoughnessFactor DEFAULT(0);
    TextureInfo ColorTexture DEFAULT();
    TextureInfo RoughnessTexture DEFAULT();
};
#ifndef __METAL_VERSION__
template<> inline constexpr field::SpecTable<Sheen> Specs<Sheen>{{&Sheen::RoughnessFactor, {.Min = 0, .Max = 1}}};
#endif
