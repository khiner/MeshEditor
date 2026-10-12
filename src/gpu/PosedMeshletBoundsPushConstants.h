#pragma once

#include "gpu/ElementWork.h"
#include "gpu/MeshElementSet.h"
#include "gpu/Types.h"

// A full-frame descriptor expands a pinned finest node's member run on the GPU, through the record's member and meshlet sets.
// There is no per-meshlet CPU tile list, and output addresses do not depend on rank.
struct PosedMeshletBoundsJob {
    uint32_t FirstGroup DEFAULT(), Count DEFAULT(), Instance DEFAULT();
    uint32_t Members DEFAULT(), MemberSet DEFAULT(), MeshletSet DEFAULT();
};
static_assert(sizeof(PosedMeshletBoundsJob) == 24);

struct PosedMeshletBoundsPushConstants {
    ElementWork MeshletAddresses DEFAULT();
    uint32_t Instance DEFAULT();
    uint32_t JobsSlot DEFAULT(), JobCount DEFAULT();
    uint32_t MeshletSlot DEFAULT();
    uint32_t MeshletVertexSlot DEFAULT();
    uint32_t PosedMeshletBoundsSlot DEFAULT();
    uint32_t PosedMeshletBoundsNodesSlot DEFAULT();
    SetArenaSlots Members DEFAULT(), Meshlets DEFAULT();
    uint32_t MemberSlot DEFAULT();
};
static_assert(sizeof(PosedMeshletBoundsPushConstants) == 72, "PosedMeshletBoundsPushConstants size");
