#pragma once

#include "gpu/Types.h"

struct MotionBlurResolvePushConstants {
    uint32_t AccumSamplerSlot DEFAULT();
    float InvSteps DEFAULT();
};
