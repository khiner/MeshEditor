#pragma once

#include "gpu/Types.h"

struct CubeFacePushConstants {
    uint32_t FaceSize DEFAULT();
};
static_assert(sizeof(CubeFacePushConstants) == 4, "CubeFacePushConstants size");
