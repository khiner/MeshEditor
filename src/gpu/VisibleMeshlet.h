#pragma once

#include "gpu/Types.h"

// A drawn cluster, with its meshlet and primitive at their arena addresses.
struct VisibleMeshlet {
    uint32_t Instance DEFAULT();
    uint32_t MeshletAddress DEFAULT();
    // The instance's mesh record and the cluster's primitive, written when the cull compacts the entry.
    uint32_t Mesh DEFAULT(InvalidOffset);
    uint32_t PrimitiveAddress DEFAULT(InvalidOffset);
};
static_assert(sizeof(VisibleMeshlet) == 16, "VisibleMeshlet size");
