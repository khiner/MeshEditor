#pragma once

#include "gpu/Types.h"

struct PrefilterPushConstants {
    uint32_t FaceSize DEFAULT();
    uint32_t SourceSize DEFAULT();
    float Roughness DEFAULT();
};
static_assert(sizeof(PrefilterPushConstants) == 12, "PrefilterPushConstants size");
