#pragma once

#include "gpu/TextureInfo.h"
#include "gpu/Types.h"

struct Volume {
    float ThicknessFactor DEFAULT(0);
    vec3 AttenuationColor DEFAULT(1);
    float AttenuationDistance DEFAULT(0);
    TextureInfo ThicknessTexture DEFAULT();
};
static_assert(sizeof(Volume) == 48, "Volume size");
