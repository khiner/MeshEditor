#pragma once

#include "gpu/Types.h"
#include "gpu/TextureInfo.h"

struct Transmission {
    float Factor DEFAULT(0);
    TextureInfo Texture DEFAULT();
};
static_assert(sizeof(Transmission) == 32, "Transmission size");
