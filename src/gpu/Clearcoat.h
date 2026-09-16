#pragma once

#include "gpu/TextureInfo.h"
#include "gpu/Types.h"

struct Clearcoat {
    float Factor DEFAULT(0);
    float RoughnessFactor DEFAULT(0);
    float NormalScale DEFAULT(1);
    TextureInfo Texture DEFAULT();
    TextureInfo RoughnessTexture DEFAULT();
    TextureInfo NormalTexture DEFAULT();
};
static_assert(sizeof(Clearcoat) == 96, "Clearcoat size");
