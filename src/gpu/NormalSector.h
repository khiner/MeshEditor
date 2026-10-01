#pragma once
#include "gpu/Types.h"

// Sparse base normal at a canonical root corner. CornerSectors identifies live roots.
struct NormalSector {
    vec3 Normal DEFAULT();
};
static_assert(sizeof(NormalSector) == 12);
