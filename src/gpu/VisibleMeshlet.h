#pragma once

#include "gpu/Types.h"

struct VisibleMeshlet {
    uint32_t Instance DEFAULT();
    uint32_t Meshlet DEFAULT();
    // The instance's mesh record, written when the cull compacts the entry.
    uint32_t Mesh DEFAULT(InvalidOffset);
};
static_assert(sizeof(VisibleMeshlet) == 12, "VisibleMeshlet size");
