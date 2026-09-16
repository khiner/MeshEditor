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
