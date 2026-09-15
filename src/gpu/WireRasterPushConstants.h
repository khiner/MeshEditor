#pragma once

#include "gpu/Types.h"
#include "gpu/MeshletDrawPushConstants.h"

struct WireRasterPushConstants {
    MeshletDrawPushConstants Meshlet DEFAULT();
    uint32_t CoverageSlot DEFAULT(InvalidSlot);
    uint32_t TestDepth DEFAULT();
};
static_assert(sizeof(WireRasterPushConstants) == 76, "WireRasterPushConstants size");
