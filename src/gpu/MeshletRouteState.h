#pragma once

#include "gpu/MeshletRoute.h"
#include "gpu/Types.h"

struct MeshletRouteState {
    GpuArray<uint32_t, uint32_t(MeshletRoute::Count)> Counts DEFAULT();
    GpuArray<uint32_t, uint32_t(MeshletRoute::Count)> Offsets DEFAULT();
};
static_assert(sizeof(MeshletRouteState) == 72, "MeshletRouteState size");
