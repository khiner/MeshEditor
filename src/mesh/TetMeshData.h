#pragma once

#include "numeric/vec3.h"

#include <cstdint>
#include <vector>

struct TetMeshData {
    std::vector<vec3> Positions;
    std::vector<uint32_t> EdgeIndices;
};
