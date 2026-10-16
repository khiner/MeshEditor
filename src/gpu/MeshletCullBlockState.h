#pragma once

#include "gpu/MeshletRoute.h"

struct MeshletCullBlockState {
    GpuArray<uint32_t, uint32_t(MeshletRoute::Count)> Routes DEFAULT();
};
