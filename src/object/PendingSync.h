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

// Transient edge indices for a collider/tet wireframe, consumed when the edge index buffer is built then removed.
// Camera/light/empty extras instead derive their edges from the object's params.
struct PendingEdgeIndices {
    std::vector<uint32_t> Indices;
};
