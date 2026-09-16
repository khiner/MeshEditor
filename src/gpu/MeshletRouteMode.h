#pragma once

#include "gpu/Types.h"

// How the meshlet cull assigns routes.
enum class MeshletRouteMode : uint32_t {
    Single = 0,
    Material = 1,
    Transmission = 2,
    // Solid and wireframe frames, where lines and points draw as overlays instead of visibility surfaces.
    Visibility = 3,
    // Visibility routing that keeps lines and points on the coverage route so selection queries can hit them.
    Selection = 4,
};
