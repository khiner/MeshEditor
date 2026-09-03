#pragma once

#include <span>
#include <vector>

#include <entt/entity/fwd.hpp>

struct MeshData;

struct ConnectivityTarget {
    uint32_t StoreId;
    const MeshData *Data;
};

// Builds eligible connectivity on the GPU and returns n-gon, face-less, or non-manifold targets for host construction.
std::vector<ConnectivityTarget> BuildConnectivityNow(entt::registry &, std::span<const ConnectivityTarget>);
