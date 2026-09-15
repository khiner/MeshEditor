#pragma once

#include "gpu/Types.h"

struct MeshDispatchArgs {
    uint32_t ThreadgroupsX DEFAULT();
    uint32_t ThreadgroupsY DEFAULT();
    uint32_t ThreadgroupsZ DEFAULT();
};
static_assert(sizeof(MeshDispatchArgs) == 12, "MeshDispatchArgs size");
