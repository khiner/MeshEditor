#pragma once

#include "gpu/TextureInfo.h"

struct Volume {
    float ThicknessFactor DEFAULT(0);
    vec3 AttenuationColor DEFAULT(1);
    float AttenuationDistance DEFAULT(0);
    TextureInfo ThicknessTexture DEFAULT();
};
#ifndef __METAL_VERSION__
template<> inline constexpr field::SpecTable<Volume> Specs<Volume>{{&Volume::ThicknessFactor, {.Min = 0, .Max = 10}}};
#endif
