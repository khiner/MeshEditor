#pragma once
#include "gpu/MeshletIndex.h"
#include "gpu/SlotOffset.h"

struct LodNodeRefitPushConstants {
    SlotOffset Jobs DEFAULT(); // Node IDs
    uint32_t Count DEFAULT();
    MeshletIndexRef Nodes DEFAULT(), Meshlets DEFAULT(), Groups DEFAULT();
    uint32_t NodeSlot DEFAULT(), ParentSlot DEFAULT(), MeshletSlot DEFAULT(), GroupSlot DEFAULT(), ErrorSlot DEFAULT();
    uint32_t NodeCapacity DEFAULT(), MeshletCapacity DEFAULT(), GroupCapacity DEFAULT(), IndexNodeCapacity DEFAULT();
};
static_assert(sizeof(LodNodeRefitPushConstants) == 84);
