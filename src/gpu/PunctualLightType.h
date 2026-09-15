#pragma once

#include "gpu/Types.h"

enum class PunctualLightType : uint32_t {
    Directional = 0,
    Point = 1,
    Spot = 2,
};
