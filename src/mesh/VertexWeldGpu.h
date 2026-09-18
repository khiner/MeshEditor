#pragma once

#include "numeric/vec3.h"

#include <span>
#include <vector>

#include "state/Entity.h"

using numeric::vec3;

struct MeshData;

// MorphTangentDeltas points at the host-owned target-major tangent deltas, compacted in place with the vertices.
struct WeldTarget {
    uint32_t StoreId;
    MeshData *Data;
    std::vector<vec3> *MorphTangentDeltas;
};

// Merges vertices identical across every vertex-domain channel and compacts their GPU arenas.
void WeldMeshesNow(state::Scene &, std::span<const WeldTarget>);
