#pragma once

#include "gpu/TextureInfo.h"

struct Specular {
    float Factor DEFAULT(1);
    vec3 ColorFactor DEFAULT(1);
    TextureInfo Texture DEFAULT();
    TextureInfo ColorTexture DEFAULT();
};
