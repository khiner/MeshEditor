#pragma once

#include "gpu/TextureInfo.h"
#include "gpu/Types.h"

struct Iridescence {
    float Factor DEFAULT(0);
    float Ior DEFAULT(1.3);
    float ThicknessMinimum DEFAULT(100);
    float ThicknessMaximum DEFAULT(400);
    TextureInfo Texture DEFAULT();
    TextureInfo ThicknessTexture DEFAULT();
};
static_assert(sizeof(Iridescence) == 72, "Iridescence size");
#ifndef __METAL_VERSION__
#include "Field.h"
template<> inline constexpr FieldSpec Spec<Iridescence, "Factor">{.Min = 0, .Max = 1};
template<> inline constexpr FieldSpec Spec<Iridescence, "Ior">{.Min = 1, .Max = 5};
template<> inline constexpr FieldSpec Spec<Iridescence, "ThicknessMinimum">{.Min = 0, .Max = 1000, .Digits = 0};
template<> inline constexpr FieldSpec Spec<Iridescence, "ThicknessMaximum">{.Min = 0, .Max = 1000, .Digits = 0};
#endif
