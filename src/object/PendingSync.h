#pragma once

#include "Range.h"
#include "object/ObjectSchema.h"
#include "state/DirtySet.h"

// Data owners affected by object destruction, resolved against survivors after the frame's actions.
// Destroyed components queue their store entries and buffer ranges, released together in the settle pass.
struct PendingObjectRemovals {
    state::DirtySet Buffers, Armatures;
    std::vector<uint32_t> StoreIds;
    std::vector<Range> InstanceRanges, DeformRanges, MorphRanges, SoundVertexRanges;
    std::vector<Range> BoundsEntryRanges, DeriveEntryRanges, PreludeTileRanges;
    std::vector<state::Entity> Objects;
    std::vector<state::Entity> Dependents; // Entities a destroyed entity owned, destroyed after it
};
// A scene reset drops the queued removals, whose entities and ranges belong to the reset scene.
void Reset(PendingObjectRemovals &, state::Registry &);

// Slots of destroyed render instances and retired owners, resolved together by SyncModelsBuffers.
struct PendingSlotRemovals {
    struct Removal {
        state::Entity Owner;
        uint32_t Index;
        auto operator<=>(const Removal &) const = default;
    };
    std::vector<Removal> Instances;
    std::vector<state::Entity> Retired;
};
// A scene reset drops the queued removals, whose owners and slots belong to the reset scene.
void Reset(PendingSlotRemovals &, state::Registry &);
