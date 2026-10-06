#pragma once

#include <span>

#include "state/Entity.h"

// Builds canonical face and wire halfedge connectivity on the GPU and completes the records.
// Requires AllocateConnectivity on each store id, with face starts in place for a mesh whose faces are not all triangles.
void BuildConnectivityNow(state::Scene &, std::span<const uint32_t> store_ids);
