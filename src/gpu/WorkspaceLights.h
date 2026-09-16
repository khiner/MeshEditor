#pragma once

#include "gpu/SolidLight.h"
#include "gpu/Types.h"

struct WorkspaceLights {
    GpuArray<SolidLight, 4> Lights DEFAULT();
    vec3 AmbientColor DEFAULT();
    uint32_t UseSpecular DEFAULT();
};
static_assert(sizeof(WorkspaceLights) == 176, "WorkspaceLights size");
