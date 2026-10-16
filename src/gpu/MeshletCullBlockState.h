#pragma once

#include "gpu/MeshletRoute.h"
#include "gpu/Types.h"

struct MeshletCullBlockState {
    GpuArray<uint32_t, uint32_t(MeshletRoute::Count)> Routes DEFAULT();
};
