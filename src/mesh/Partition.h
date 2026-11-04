#pragma once
#include "numeric/vec4.h"
#include <array>
#include <vector>

using numeric::vec4;

// Manifold's triangulation of a triangle or quad whose sides divide into Divisions[i] pieces, the fourth zero for a triangle.
// Vertices are the corners, then each side's interior points from its start, then the interior points, each with its weights over the corners in sorted order.
struct PartitionTable {
    std::array<uint32_t, 4> Divisions;
    std::vector<vec4> Weights;
    std::vector<std::array<uint32_t, 3>> Triangles;
    uint32_t InteriorOffset() const { return Divisions[0] + Divisions[1] + Divisions[2] + Divisions[3]; }
};
// The table of the sorted divisions: descending for a triangle, and for a quad rotated so the least side comes first, breaking ties by the side after it.
PartitionTable Partition(std::array<uint32_t, 4> sorted);
