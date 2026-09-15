#pragma once

#include "gpu/Types.h"
#include "gpu/TextureInfo.h"

struct Sheen {
    vec3 ColorFactor DEFAULT(0);
    float RoughnessFactor DEFAULT(0);
    TextureInfo ColorTexture DEFAULT();
    TextureInfo RoughnessTexture DEFAULT();
};
static_assert(sizeof(Sheen) == 72, "Sheen size");
