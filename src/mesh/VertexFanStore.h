#pragma once

#include "Range.h"
#include "gpu/BindlessBindings.h"
#include "metal/Buffer.h"
#include "metal/BufferArena.h"

// Packed (canonical corner, canonical face) incidence in runs.
// A vertex root (first, count) owns the run of items it names.
struct VertexFanStore {
    explicit VertexFanStore(mtl::BufferContext &ctx) : Items{ctx, SlotType::Buffer} {}

    void Track(store::History &history, const std::string &name) { Items.Track(history, name + ".items"); }
    void Release(const auto &roots) {
        std::vector<Range> ranges;
        for (const auto root : roots) AppendRange(ranges, {root.x, root.y});
        Items.Release(std::move(ranges));
    }
    void Reset() { Items.Reset(); }

    BufferArena<uvec2> Items;
};
