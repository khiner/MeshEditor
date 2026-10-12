#pragma once
#include "gpu/MeshElementSet.h"
#include "gpu/SlotOffset.h"

// Jobs name node handles of the record whose node, meshlet, group and member sets resolve its handles.
struct LodNodeRefitPushConstants {
    SlotOffset Jobs DEFAULT();
    uint32_t Count DEFAULT();
    SetArenaSlots Nodes DEFAULT(), Meshlets DEFAULT(), Groups DEFAULT(), Members DEFAULT();
    uint32_t NodeSet DEFAULT(InvalidOffset), MeshletSet DEFAULT(InvalidOffset), GroupSet DEFAULT(InvalidOffset), MemberSet DEFAULT(InvalidOffset);
    uint32_t NodeSlot DEFAULT(), ParentSlot DEFAULT(), MeshletSlot DEFAULT(), GroupSlot DEFAULT(), MemberSlot DEFAULT(), ErrorSlot DEFAULT();
};
static_assert(sizeof(LodNodeRefitPushConstants) == 100);
