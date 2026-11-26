#pragma once

#include "gpu/Types.h"

// A fill of Count words at Values in the scratch with Value.
struct ScratchFillRun {
    uint32_t Values DEFAULT(), Count DEFAULT(), Value DEFAULT();
};

// A scan of Count words at Values in place, through one sum per scan block at Blocks.
struct ScratchScanRun {
    uint32_t Values DEFAULT(), Count DEFAULT(), Blocks DEFAULT();
};
