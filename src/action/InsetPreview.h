#pragma once

#include "Range.h"
#include "gpu/MeshTopologyOp.h"
#include "metal/Buffer.h"
#include "state/Entity.h"

#include <vector>

namespace action::mesh {
// Gesture-local metadata and GPU basis for parameter-only inset updates.
// Source positions, directions, and canonical vertex writes remain on the GPU.
struct InsetPreviewCache {
    struct Entry {
        state::Entity Entity;
        uint32_t StoreId;
        MeshTopologyOp Op;
        uint32_t Flags;
        mtl::Buffer Basis;
        std::vector<Range> Ranges;
    };
    std::vector<Entry> Entries;
};
} // namespace action::mesh
