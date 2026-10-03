#pragma once

#include "Range.h"
#include "gpu/MeshTopologyOp.h"
#include "metal/BufferArena.h"
#include "state/Entity.h"

#include <vector>

namespace action::mesh {
// Gesture-local metadata and GPU basis for parameter-only inset updates.
// Source positions, directions, and canonical vertex writes remain on the GPU.
// Every entry's basis is a range of one word arena.
struct InsetPreviewCache {
    explicit InsetPreviewCache(mtl::BufferContext &ctx) : Basis{ctx, SlotType::Buffer, mtl::BufferLifetime::Workspace} {}

    struct Entry {
        state::Entity Entity;
        uint32_t StoreId;
        MeshTopologyOp Op;
        uint32_t Flags;
        Range Basis; // Words of InsetVertexBasis records
        std::vector<Range> Ranges;
    };
    BufferArena<uint32_t> Basis;
    std::vector<Entry> Entries;
};
} // namespace action::mesh
