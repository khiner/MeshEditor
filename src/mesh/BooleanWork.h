#pragma once

#include "gpu/SlotOffset.h"

struct MeshClosure;
struct MeshTopologyTask;
namespace mtl {
struct ComputeChain;
} // namespace mtl
namespace state {
struct Scene;
} // namespace state

// Construct boolean work in chain scratch and retain it through topology counting and emission.
// Submit between stages whose sizes depend on prior GPU counts.
struct BooleanWork {
    BooleanWork(state::Scene &, mtl::ComputeChain &, const MeshTopologyTask &, const MeshClosure &first, const MeshClosure &second);

    SlotOffset Job; // The BooleanJob record
    uint32_t Vertices, Intersections, Triangles; // The count entries of each kind, in that order
    uint32_t OutputVertices, OutputTriangles;
};
