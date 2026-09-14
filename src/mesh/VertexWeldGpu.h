#pragma once

#include <span>

#include "state/Entity.h"

struct MeshData;
struct PreparedMesh;

struct WeldTarget {
    uint32_t StoreId;
    MeshData *Data;
    PreparedMesh *Prepared;
};

// Merges vertices identical across every vertex-domain channel and compacts their GPU arenas.
void WeldMeshesNow(state::Scene &, std::span<const WeldTarget>);
