#pragma once

#include "gpu/ElementWork.h"
#include "gpu/MeshletIndex.h"
#include "gpu/Types.h"

// A full-frame descriptor expands canonical membership on the GPU. There is
// no per-meshlet CPU tile list, and output addresses do not depend on rank.
struct PosedMeshletBoundsJob {
    uint32_t FirstGroup DEFAULT(), Count DEFAULT(), Instance DEFAULT();
    MeshletIndexRef Clusters DEFAULT();
};
static_assert(sizeof(PosedMeshletBoundsJob) == 24);

struct PosedMeshletBoundsPushConstants {
    ElementWork Work DEFAULT();
    uint32_t Instance DEFAULT();
    uint32_t JobsSlot DEFAULT(), JobCount DEFAULT();
    uint32_t MeshletSlot DEFAULT();
    uint32_t MeshletVertexSlot DEFAULT();
    uint32_t PosedMeshletBoundsSlot DEFAULT();
    uint32_t PosedMeshletBoundsNodesSlot DEFAULT();
};
static_assert(sizeof(PosedMeshletBoundsPushConstants) == 44, "PosedMeshletBoundsPushConstants size");
