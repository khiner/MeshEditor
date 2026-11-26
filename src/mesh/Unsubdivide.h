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
// Plans one alternating reduction of the selected vertices as a topology task, or none when nothing reduces.
std::optional<MeshTopologyTask> UnsubdivideTask(state::Registry &, const Mesh &, const GeometrySelection &);
// Records the reads of UnsubdivideTask in the record's lane of the chain, each once the chain completed the reads before it, and calls `then` with the task once they completed.
void RecordUnsubdivideTask(state::Registry &, mtl::ComputeChain &, const Mesh &, const GeometrySelection &, std::function<void(std::optional<MeshTopologyTask>)> then);
