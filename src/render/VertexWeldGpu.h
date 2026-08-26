#pragma once

#include <span>

#include <entt/entity/fwd.hpp>

struct MeshData;
struct PreparedMesh;

// One mesh whose vertices merge, holding the morph tangent deltas the weld compacts alongside its arenas.
struct WeldTarget {
    uint32_t StoreId;
    MeshData *Data;
    PreparedMesh *Prepared;
};

// Merge each target's vertices identical in every vertex-domain channel, deduplicating and compacting
// the arenas on the GPU.
void WeldMeshesNow(entt::registry &, std::span<const WeldTarget>);
