#pragma once

#include <span>
#include <vector>

#include <entt/entity/fwd.hpp>

struct MeshData;

// One mesh whose connectivity builds, naming the store entry the build reads corners from and fills.
struct ConnectivityTarget {
    uint32_t StoreId;
    const MeshData *Data;
};

// Build each target's half-edge connectivity on the GPU, into the arena run the store allocated.
// Returns the targets it left alone, which the store builds itself: an n-gon or face-less mesh, and
// one the pairing found a third halfedge on an edge of.
std::vector<ConnectivityTarget> BuildConnectivityNow(entt::registry &, std::span<const ConnectivityTarget>);
