#pragma once

#include "gpu/Types.h"

// A pass over a run of Count words at Values in the buffer at Slot.
// A fill sets every word to Value, and a scan replaces the words with their exclusive prefix sums in place through one sum per scan block at Blocks.
struct ScratchPassPushConstants {
    uint32_t Slot DEFAULT(InvalidSlot);
    uint32_t Values DEFAULT(), Count DEFAULT();
    uint32_t Blocks DEFAULT(), Value DEFAULT();
};
