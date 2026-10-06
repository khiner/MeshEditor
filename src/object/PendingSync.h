#pragma once

#include "Range.h"
#include "state/Scene.h"
#include <cstdint>
#include <vector>

// Data owners affected by object destruction, resolved against survivors after the frame's actions.
// Destroyed components queue their store entries and buffer ranges, released together in the settle pass.
struct PendingObjectRemovals {
    state::DirtySet Buffers, Armatures;
    std::vector<uint32_t> StoreIds;
    std::vector<Range> InstanceRanges, DeformRanges, MorphRanges, SoundVertexRanges;
};

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
