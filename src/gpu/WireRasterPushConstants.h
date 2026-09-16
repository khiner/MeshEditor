#pragma once

#include "gpu/MeshletDrawPushConstants.h"
#include "gpu/Types.h"

struct WireRasterPushConstants {
    MeshletDrawPushConstants Meshlet DEFAULT();
    uint32_t CoverageSlot DEFAULT(InvalidSlot);
    uint32_t TestDepth DEFAULT();
};
static_assert(sizeof(WireRasterPushConstants) == 76, "WireRasterPushConstants size");
