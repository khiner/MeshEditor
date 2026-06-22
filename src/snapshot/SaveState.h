#pragma once

#include <entt/entity/fwd.hpp>

#include <cstddef>
#include <span>
#include <vector>

namespace snapshot {
// Full persistent app image: the registry's Persistent components followed by the MeshStore arena +
// connectivity blob (length-prefixed so LoadState can split them).
std::vector<std::byte> SaveState(const entt::registry &);

// Inverse of SaveState: restore the MeshStore arenas first (so Range/StoreId offsets stay valid), then the registry components.
// The caller clears the scene beforehand and runs one update pass afterward to rebuild derived/GPU state.
void LoadState(entt::registry &, std::span<const std::byte>);
} // namespace snapshot
