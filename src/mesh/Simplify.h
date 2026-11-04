#pragma once
#include "mesh/MeshTopology.h"

#include <functional>

namespace state {
struct Registry;
} // namespace state

// Collapses the selected region's edges on the GPU in rounds, electing greedily by quadric error, with each elected collapse blocking those touching its faces.
// Every collapse costing below the tolerance squared runs first, onto its cheaper end.
// `run` executes each round's Decimate task on the record.
bool SimplifyGeometry(state::Registry &, uint32_t id, const GeometrySelection &, float tolerance, float ratio, const std::function<void(const MeshTopologyTask &)> &run);
