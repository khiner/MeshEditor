#pragma once

#include "gpu/Types.h"

struct VisibleMeshlet {
    uint32_t Instance DEFAULT();
    uint32_t Meshlet DEFAULT();
};
static_assert(sizeof(VisibleMeshlet) == 8, "VisibleMeshlet size");
