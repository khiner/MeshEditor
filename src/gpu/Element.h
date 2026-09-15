#pragma once

#include "gpu/Types.h"

enum class Element : uint32_t {
    None = 0,
    Vertex = 1,
    Edge = 2,
    Face = 4,
};
