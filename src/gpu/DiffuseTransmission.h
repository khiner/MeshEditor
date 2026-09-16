#pragma once

#include "gpu/TextureInfo.h"
#include "gpu/Types.h"

struct DiffuseTransmission {
    float Factor DEFAULT(0);
    vec3 ColorFactor DEFAULT(1);
    TextureInfo Texture DEFAULT();
    TextureInfo ColorTexture DEFAULT();
};
static_assert(sizeof(DiffuseTransmission) == 72, "DiffuseTransmission size");
