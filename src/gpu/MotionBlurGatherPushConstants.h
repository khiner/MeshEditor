#pragma once

#include "gpu/Types.h"

struct MotionBlurGatherPushConstants {
    uint32_t DepthSamplerSlot DEFAULT();
    uint32_t VelocitySamplerSlot DEFAULT();
    uint32_t ColorSamplerSlot DEFAULT();
    float NoiseOffset DEFAULT();
    vec4 DepthUnproject DEFAULT();
};
static_assert(sizeof(MotionBlurGatherPushConstants) == 32, "MotionBlurGatherPushConstants size");
