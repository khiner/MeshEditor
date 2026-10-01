#pragma once

#include "Range.h"
#include "gpu/BindlessBindings.h"
#include "metal/Buffer.h"
#include "metal/BufferArena.h"

// Packed (canonical corner, canonical face) incidence in runs.
// A vertex root (first, count) owns the run of items it names.
struct VertexFanStore {
    explicit VertexFanStore(mtl::BufferContext &ctx) : Items{ctx,SlotType::Buffer} {}

    void Track(store::History &history, const std::string &name) { Items.Track(history,name+".items"); }
    // Frees the runs the roots own, in root order.
    void Release(std::span<const uvec2> roots) {
        Range run{};
        for (const auto root : roots) {
            if (!root.y) continue;
            if (run.Count && run.Offset+run.Count == root.x) { run.Count += root.y; continue; }
            Items.Release(run);
            run = {root.x,root.y};
        }
        Items.Release(run);
    }
    void Reset() { Items.Reset(); }

    BufferArena<uvec2> Items;
};
