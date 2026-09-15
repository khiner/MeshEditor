#pragma once

#include "gpu/Types.h"

// The top two bits select vertex, face, or normal-sector shading. The low 30 bits index the sector.
enum class CornerClass : uint32_t {
    Vertex = 0,
    Face = 1,
    Seam = 2,
};
