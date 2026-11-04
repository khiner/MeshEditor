#pragma once
#include "mesh/MeshTopology.h"

#include <array>
#include <optional>
#include <vector>

struct MeshStore;

// Manifold's level set lattice over a box: a cubic grid of Size points per axis from Origin at Spacing, and the body centers half a step below each grid point.
// Points() lists every lattice point the field is read at, grid points first, each axis slowest to fastest in x, y, z.
struct Lattice {
    vec3 Origin, Spacing;
    std::array<int32_t, 3> Size;
    uint32_t Count() const;
    std::vector<vec3> Points() const;
};
// The lattice whose spacing is at most the edge length over the box, or none for a box or edge length that is not finite and positive.
std::optional<Lattice> LatticeOver(vec3 min, vec3 max, float edge_length);
// Marches the surface where the record's Float vertex attribute, positive inside, crosses the level into a new mesh, by manifold's marching tetrahedra.
// The record's vertices are the lattice's points in order, as Points() lists them, with none removed since.
MeshTopologyTask LevelSetTask(const MeshStore &, uint32_t id, const Lattice &, uint32_t attribute, float level);
