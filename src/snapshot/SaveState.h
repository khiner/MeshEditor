#pragma once

#include <entt/entity/fwd.hpp>

#include <span>
#include <vector>

namespace snapshot {
// Serializes persistent registry components followed by length-prefixed MeshStore arena and connectivity data.
std::vector<std::byte> SaveState(const entt::registry &);

// Inverse of SaveState: restore the MeshStore arenas first (so Range/StoreId offsets stay valid), then the registry components.
// The caller clears the scene beforehand and runs one update pass afterward to rebuild derived/GPU state.
void LoadState(entt::registry &, std::span<const std::byte>);
} // namespace snapshot
