#pragma once

#include "gpu/Types.h"

enum class MaterialAlphaMode : uint32_t {
    Opaque = 0,
    Mask = 1,
    Blend = 2,
};
