#pragma once

#include "gpu/TextureInfo.h"
#include "gpu/Types.h"

struct Transmission {
    float Factor DEFAULT(0);
    TextureInfo Texture DEFAULT();
};
static_assert(sizeof(Transmission) == 32, "Transmission size");
#ifndef __METAL_VERSION__
#include "Field.h"
template<> inline constexpr FieldSpec Spec<Transmission, "Factor">{.Min = 0, .Max = 1};
#endif
