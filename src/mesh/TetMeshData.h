#pragma once

#include "numeric/vec3.h"

#include <cstdint>
#include <vector>

using numeric::vec3;

struct TetMeshData {
    std::vector<vec3> Positions;
    std::vector<uint32_t> EdgeIndices;

    bool operator==(const TetMeshData &) const = default;
};
