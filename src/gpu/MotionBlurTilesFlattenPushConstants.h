#pragma once

#include "gpu/Types.h"
#include "gpu/VisibilityShadingPushConstants.h"

struct MotionBlurTilesFlattenPushConstants {
    VisibilityShadingPushConstants Visibility DEFAULT();
    mat4 InvViewProj DEFAULT();
    uint32_t CameraMotion DEFAULT();
};
