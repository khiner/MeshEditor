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
#ifndef __METAL_VERSION__
#include "Field.h"
template<> inline constexpr FieldSpec Spec<Clearcoat, "Factor">{.Min = 0, .Max = 1};
template<> inline constexpr FieldSpec Spec<Clearcoat, "RoughnessFactor">{.Min = 0, .Max = 1};
template<> inline constexpr FieldSpec Spec<Clearcoat, "NormalScale">{.Min = -2, .Max = 2};
#endif
