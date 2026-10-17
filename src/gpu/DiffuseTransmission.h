#pragma once

#include "gpu/TextureInfo.h"

struct DiffuseTransmission {
    float Factor DEFAULT(0);
    vec3 ColorFactor DEFAULT(1);
    TextureInfo Texture DEFAULT();
    TextureInfo ColorTexture DEFAULT();
};
#ifndef __METAL_VERSION__
template<> inline constexpr field::SpecTable<DiffuseTransmission> Specs<DiffuseTransmission>{{&DiffuseTransmission::Factor, {.Min = 0, .Max = 1}}};
#endif
