#pragma once

#include "gpu/Types.h"
#include "gpu/TextureInfo.h"

struct Anisotropy {
    float Strength DEFAULT(0);
    float Rotation DEFAULT(0);
    TextureInfo Texture DEFAULT();
};
static_assert(sizeof(Anisotropy) == 36, "Anisotropy size");
