#pragma once

#include "gpu/MeshTopologyOp.h"
#include "metal/BufferArena.h"
#include "state/Entity.h"

namespace action::mesh {
// Cache gesture metadata and GPU ranges for inset parameter updates.
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
