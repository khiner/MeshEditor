#pragma once

#include "gpu/BindlessBindings.h"
#include "gpu/ElementSelectQuery.h"
#include "metal/Buffer.h"

// Grow-only query masks address canonical elements.
// Fresh backing is zero.
// Every query consumes and clears its active words before reuse.
// The append list needs no initialization beyond its scalar count.
struct SelectionQuery {
    explicit SelectionQuery(mtl::BufferContext &ctx)
        : Masks(ctx, 0, SlotType::Buffer), Words(ctx, 0, SlotType::Buffer), Count(ctx, 0, SlotType::Buffer) {}
    void Reserve(uint32_t elements) {
        const auto bytes = ((uint64_t(elements) + 31u) / 32u) * sizeof(uint32_t);
        // Retry each reservation independently after a partial allocation failure.
        if (bytes > Masks.UsedSize) Masks.SetUsedSize(bytes);
        if (bytes > Words.UsedSize) Words.SetUsedSize(bytes);
        if (bytes && !Count.UsedSize) Count.Update(as_bytes(uint32_t{0}));
    }
    uint32_t WordCount() const { return Count.UsedSize ? Count.GetSpan<uint32_t>()[0] : 0u; }
    SelectionQueryRef Ref() const { return {Masks.Slot, Words.Slot, Count.Slot}; }
    mtl::Buffer Masks, Words, Count;
};
