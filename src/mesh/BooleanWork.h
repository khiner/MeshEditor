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

// Manifold's boolean of a task's source P and second source Q as GPU passes over the chain's scratch, through the triangles each kept source triangle's polygon emits.
// Construction submits once per stage whose sizes the stage before it counted: the crossings, the component label rounds until they converge, the output counts, and the polygons.
// Its runs stay in the chain's scratch for the topology job's count and emission passes.
struct BooleanWork {
    BooleanWork(state::Scene &, mtl::ComputeChain &, const MeshTopologyTask &, const MeshClosure &first, const MeshClosure &second);

    SlotOffset Job; // The BooleanJob record
    uint32_t Vertices, Intersections, Triangles; // The count entries of each kind, in that order
    uint32_t OutputVertices, OutputTriangles;
};
