#pragma once

#include "numeric/vec3.h"

#include <cstdint>
#include <vector>

struct MeshData {
    std::vector<vec3> Positions;
    std::vector<std::vector<uint32_t>> Faces;
};
