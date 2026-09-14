#pragma once

#include "state/Entity.h"

#include <span>
#include <vector>

namespace snapshot {
// Serializes entity allocator state, persistent components, materials, and MeshStore data.
std::vector<std::byte> SaveState(const state::Scene &);

// Restores entity allocation state and MeshStore offsets before the components that reference them.
// Requires an empty scene with only the initialized viewport entity, using the snapshot's viewport handle.
// Run one update pass afterward to rebuild derived/GPU state.
void LoadState(state::Scene &, std::span<const std::byte>);
} // namespace snapshot
