#pragma once
#include "gpu/MeshElementSet.h"
#include "gpu/SlotOffset.h"

// One record's clusters from meshlet handle First, whose table run the record's meshlet set names, with their triangle ids from the record's range base.
struct MeshletOwnersJob {
    uint32_t First DEFAULT(), Count DEFAULT();
    uint32_t MeshletSet DEFAULT(), TriangleIdBase DEFAULT(), Topology DEFAULT();
    SlotOffset Owners DEFAULT(); // The record's owner range of the topology's domain, which covers its handles below HandleBound
    uint32_t HandleBound DEFAULT();
};
static_assert(sizeof(MeshletOwnersJob) == 32);

// Each of the Count tiles names a job and one of its clusters.
struct MeshletOwnersPushConstants {
    SlotOffset Jobs DEFAULT(), Tiles DEFAULT();
    uint32_t Count DEFAULT();
    SetArenaSlots Meshlets DEFAULT();
    uint32_t MeshletSlot DEFAULT(), TriangleIdsSlot DEFAULT(), ErrorSlot DEFAULT();
};
static_assert(sizeof(MeshletOwnersPushConstants) == 44);
