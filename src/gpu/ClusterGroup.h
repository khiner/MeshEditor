#pragma once

#include "gpu/Types.h"

// Stores conservative mesh-space bounds and monotonic accumulated simplification error for one cluster group.
// Contained member spheres and monotonic error produce a crack-free projected-error cut.
struct ClusterGroup {
    vec3 Center DEFAULT();
    float Radius DEFAULT();
    float Error DEFAULT();
};
static_assert(sizeof(ClusterGroup) == 20, "ClusterGroup size");
