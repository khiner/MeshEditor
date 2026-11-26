#pragma once
#include "mesh/MeshTopology.h"

#include <functional>

struct Mesh;
namespace state {
struct Registry;
}
namespace mtl {
struct ComputeChain;
}
// Rotates the selected triangles' shared edges until each quad they form passes the Delaunay empty-circle test.
std::optional<MeshTopologyTask> BeautifyFaceTask(state::Registry &, const Mesh &, const GeometrySelection &);
// Records the passes of BeautifyFaceTask in the record's lane of the chain, each round once the chain completed the one before it, and calls `then` with the task once they completed.
void RecordBeautifyFaceTask(state::Registry &, mtl::ComputeChain &, const Mesh &, const GeometrySelection &, std::function<void(std::optional<MeshTopologyTask>)> then);
