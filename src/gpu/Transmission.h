#pragma once

#include "gpu/TextureInfo.h"

struct Transmission {
    float Factor DEFAULT(0);
    TextureInfo Texture DEFAULT();
};
#ifndef __METAL_VERSION__
template<> inline constexpr FieldSpec Spec<Transmission, "Factor">{.Min = 0, .Max = 1};
#endif
