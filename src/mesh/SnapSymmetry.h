#pragma once
#include "mesh/GeometrySelection.h"
struct Mesh;
struct MeshStore;
struct SymmetrySnapVertex {
    uint32_t Vertex, Partner;
};
struct MeshPipelines;
// Disjoint nearest mirror pairs within the threshold, in canonical handle order, including unselected counterparts.
std::vector<SymmetrySnapVertex> PlanSymmetrySnap(const MeshStore &, const MeshPipelines &, const Mesh &, const GeometrySelection &, uint32_t axis, float threshold, bool center);
