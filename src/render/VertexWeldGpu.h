#pragma once

#include <span>

#include <entt/entity/fwd.hpp>

struct MeshData;
struct PreparedMesh;

struct WeldTarget {
    uint32_t StoreId;
    MeshData *Data;
    PreparedMesh *Prepared;
};

// Merges vertices identical across every vertex-domain channel and compacts their GPU arenas.
void WeldMeshesNow(entt::registry &, std::span<const WeldTarget>);
