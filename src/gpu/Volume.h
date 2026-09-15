#pragma once

#include "gpu/Types.h"
#include "gpu/TextureInfo.h"

struct Volume {
    float ThicknessFactor DEFAULT(0);
    vec3 AttenuationColor DEFAULT(1);
    float AttenuationDistance DEFAULT(0);
    TextureInfo ThicknessTexture DEFAULT();
};
static_assert(sizeof(Volume) == 48, "Volume size");
