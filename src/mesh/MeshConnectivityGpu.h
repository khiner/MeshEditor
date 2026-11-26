#pragma once

#include <span>

#include "state/Entity.h"

// Builds canonical face and wire halfedge connectivity and the render triangles of new records on the GPU and completes the records.
// Requires CreateMeshSource on each store id, after its weld.
void BuildConnectivityNow(state::Registry &, std::span<const uint32_t> store_ids);
