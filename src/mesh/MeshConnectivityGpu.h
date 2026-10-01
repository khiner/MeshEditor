#pragma once

#include <span>

#include "state/Entity.h"

// Builds the listed face meshes' halfedge connectivity on the GPU and completes their records.
// Requires AllocateConnectivity on each store id, with face starts in place for a mesh whose faces are not all triangles.
void BuildConnectivityNow(state::Scene &, std::span<const uint32_t> store_ids);
