#pragma once

#include "gpu/MeshletRoute.h"

struct MeshletRouteState {
    GpuArray<uint32_t, uint32_t(MeshletRoute::Count)> Counts DEFAULT();
    GpuArray<uint32_t, uint32_t(MeshletRoute::Count)> Offsets DEFAULT();
};
