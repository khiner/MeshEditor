#pragma once

#include "gpu/Types.h"

struct WireResolvePushConstants {
    uint32_t CoverageSlot DEFAULT(InvalidSlot);
};
