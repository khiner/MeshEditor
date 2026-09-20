#pragma once

#include "gpu/TextureInfo.h"
#include "gpu/Types.h"

struct Anisotropy {
    float Strength DEFAULT(0);
    float Rotation DEFAULT(0);
    TextureInfo Texture DEFAULT();
};
static_assert(sizeof(Anisotropy) == 36, "Anisotropy size");
#ifndef __METAL_VERSION__
#include "Field.h"
template<> inline constexpr FieldSpec Spec<Anisotropy, "Strength">{.Min = 0, .Max = 1};
template<> inline constexpr FieldSpec Spec<Anisotropy, "Rotation">{.Min = 0, .Max = 6.2831853f, .Digits = 1, .Unit = FieldUnit::Radians};
#endif
