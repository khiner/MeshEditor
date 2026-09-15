#pragma once

#include "gpu/Types.h"

struct MotionBlurAccumulatePushConstants {
    uint32_t SceneSamplerSlot DEFAULT();
    float Weight DEFAULT();
};
static_assert(sizeof(MotionBlurAccumulatePushConstants) == 8, "MotionBlurAccumulatePushConstants size");
