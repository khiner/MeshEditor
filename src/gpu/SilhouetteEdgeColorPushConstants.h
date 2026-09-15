#pragma once

#include "gpu/Types.h"

struct SilhouetteEdgeColorPushConstants {
    uint32_t Manipulating DEFAULT();
    uint32_t SilhouetteSamplerIndex DEFAULT();
    uint32_t ActiveObjectId DEFAULT();
    uint32_t SceneDepthSamplerIndex DEFAULT();
};
static_assert(sizeof(SilhouetteEdgeColorPushConstants) == 16, "SilhouetteEdgeColorPushConstants size");
