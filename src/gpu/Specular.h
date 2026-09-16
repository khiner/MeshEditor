#pragma once

#include "gpu/TextureInfo.h"
#include "gpu/Types.h"

struct Specular {
    float Factor DEFAULT(1);
    vec3 ColorFactor DEFAULT(1);
    TextureInfo Texture DEFAULT();
    TextureInfo ColorTexture DEFAULT();
};
static_assert(sizeof(Specular) == 72, "Specular size");
