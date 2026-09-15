#pragma once

#include "gpu/Types.h"

struct WireResolvePushConstants {
    uint32_t CoverageSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(WireResolvePushConstants) == 4, "WireResolvePushConstants size");
