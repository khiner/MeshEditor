#pragma once
#include "gpu/MeshElementSet.h"
#include "gpu/SlotOffset.h"

// The clusters from meshlet handle First, whose table run the record's meshlet set names, with their triangle ids from the record's range base.
struct MeshletOwnersPushConstants {
    uint32_t First DEFAULT(), Count DEFAULT();
    SetArenaSlots Meshlets DEFAULT();
    uint32_t MeshletSet DEFAULT(), MeshletSlot DEFAULT(), TriangleIdsSlot DEFAULT(), TriangleIdBase DEFAULT(), Topology DEFAULT();
    SlotOffset Owners DEFAULT(); // The record's owner range of the topology's domain, which covers its handles below HandleBound
    uint32_t HandleBound DEFAULT(), ErrorSlot DEFAULT();
};
static_assert(sizeof(MeshletOwnersPushConstants) == 56);
