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
#ifndef __METAL_VERSION__
#include "Field.h"
template<> inline constexpr FieldSpec Spec<DiffuseTransmission, "Factor">{.Min = 0, .Max = 1};
#endif
