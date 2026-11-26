#pragma once
#include "mesh/MeshTopology.h"

#include <functional>

namespace mtl {
struct ComputeChain;
} // namespace mtl
namespace state {
struct Registry;
} // namespace state

// Collapses the selected region's edges on the GPU in rounds, electing greedily by quadric error, with each elected collapse blocking those touching its faces.
// Every collapse costing below the tolerance squared runs first, onto its cheaper end.
// Then the rounds raise the cost until the region's triangles fall to `ratio` of them.
// Records the rounds on `chain` in the record's lane, so the rounds of several records on one chain share their passes, and returns false when the selection holds nothing to simplify.
// Once a round's passes complete, `decimate(task, next)` executes its Decimate task and calls `next()` when the record holds the result.
// `done(changed)` receives whether any round changed the record.
bool RecordSimplify(state::Registry &, mtl::ComputeChain &, uint32_t id, const GeometrySelection &, float tolerance, float ratio, std::function<void(const MeshTopologyTask &, std::function<void()> next)> decimate, std::function<void(bool changed)> done);
// Simplifies on a chain of its own, where `run` executes each round's Decimate task on the record, and returns whether any round changed the record.
bool SimplifyGeometry(state::Registry &, uint32_t id, const GeometrySelection &, float tolerance, float ratio, const std::function<void(const MeshTopologyTask &)> &run);
