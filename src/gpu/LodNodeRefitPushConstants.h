#pragma once
#include "gpu/MeshElementSet.h"
#include "gpu/SlotOffset.h"

// One node to refit by handle, with its record's node, meshlet, group and member sets, which resolve its handles.
struct LodNodeRefitJob {
    uint32_t Node DEFAULT();
    uint32_t NodeSet DEFAULT(InvalidOffset), MeshletSet DEFAULT(InvalidOffset), GroupSet DEFAULT(InvalidOffset), MemberSet DEFAULT(InvalidOffset);
};

// Jobs name `Count` nodes of every record at one depth.
struct LodNodeRefitPushConstants {
    SlotOffset Jobs DEFAULT();
    uint32_t Count DEFAULT();
    SetArenaSlots Nodes DEFAULT(), Meshlets DEFAULT(), Groups DEFAULT(), Members DEFAULT();
    uint32_t NodeSlot DEFAULT(), ParentSlot DEFAULT(), MeshletSlot DEFAULT(), GroupSlot DEFAULT(), MemberSlot DEFAULT(), ErrorSlot DEFAULT();
};
