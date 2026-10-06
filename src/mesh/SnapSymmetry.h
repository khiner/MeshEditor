#pragma once
#include "mesh/GeometrySelection.h"
#include <cstdint>
#include <vector>
struct Mesh;
struct MeshStore;
struct SymmetrySnapVertex {
    uint32_t Vertex, Partner;
};
// Disjoint nearest mirror pairs, in canonical handle order. Includes unselected
// counterparts; the bounds hierarchy and positions are read directly from UMA.
std::vector<SymmetrySnapVertex> PlanSymmetrySnap(const MeshStore &, const Mesh &, const GeometrySelection &, uint32_t axis, float threshold, bool center);
