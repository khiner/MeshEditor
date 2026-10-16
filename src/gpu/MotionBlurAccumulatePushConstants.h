#pragma once

#include "gpu/Types.h"

struct MotionBlurAccumulatePushConstants {
    uint32_t SceneSamplerSlot DEFAULT();
    float Weight DEFAULT();
};
