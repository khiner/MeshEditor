#pragma once

#include "gpu/VisibilityShadingPushConstants.h"

struct MotionBlurTilesFlattenPushConstants {
    VisibilityShadingPushConstants Visibility DEFAULT();
    mat4 InvViewProj DEFAULT();
    uint32_t CameraMotion DEFAULT();
};
