#pragma once

#include "gpu/Types.h"

struct MotionBlurResolvePushConstants {
    uint32_t AccumSamplerSlot DEFAULT();
    float InvSteps DEFAULT();
};
static_assert(sizeof(MotionBlurResolvePushConstants) == 8, "MotionBlurResolvePushConstants size");
