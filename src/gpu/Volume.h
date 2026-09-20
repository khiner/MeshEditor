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
#ifndef __METAL_VERSION__
#include "Field.h"
template<> inline constexpr FieldSpec Spec<Volume, "ThicknessFactor">{.Min = 0, .Max = 10};
#endif
