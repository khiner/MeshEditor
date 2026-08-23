#pragma once

#include <cstdint>
#include <vector>

// Placed on buffer entities. Accumulates instance slots that need GPU erasure.
// Processed and cleared each frame by SyncModelsBuffers.
struct PendingHide {
    std::vector<uint32_t> BufferIndices;
};

// Placed on viewport. Accumulates light buffer indices during Destroy(), batch-compacted later.
struct PendingLightRemovals {
    std::vector<uint32_t> Indices;
};
