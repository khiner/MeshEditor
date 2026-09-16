#pragma once

#include "gpu/MeshletDrawPushConstants.h"
#include "gpu/Types.h"

struct WireRasterPushConstants {
    MeshletDrawPushConstants Meshlet DEFAULT();
    uint32_t CoverageSlot DEFAULT(InvalidSlot);
    uint32_t TestDepth DEFAULT();
    // Scales coverage behind the visibility depth when testing. Zero discards it.
    float BehindOpacity DEFAULT();
};
static_assert(sizeof(WireRasterPushConstants) == 80, "WireRasterPushConstants size");
