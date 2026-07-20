#pragma once

#include "numeric/vec3.h"

#include <array>
#include <cstdint>
#include <vector>

// Tetrahedral volume mesh. Every tet (a, b, c, d) is positively oriented: det[a-d, b-d, c-d] > 0.
struct TetMesh {
    std::vector<dvec3> Points;
    std::vector<std::array<uint32_t, 4>> Tets;
};
