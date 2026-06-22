#pragma once

#include <entt/entity/fwd.hpp>

#include <cstddef>
#include <span>
#include <vector>

namespace snapshot {
// A deterministic byte image of the registry's Persistent components (see scene/SceneSnapshotRoles.cpp). Stable across
// runs given identical entity-id allocation, which ClearScene guarantees by resetting the allocator.
std::vector<std::byte> SnapshotSceneState(const entt::registry &);

struct SnapshotDiff {
    bool Equal;
    std::size_t FirstDifferingByte; // == min size when unequal, == size when equal
};
SnapshotDiff Compare(std::span<const std::byte> expected, std::span<const std::byte> actual);

// Restore Persistent components from a SnapshotSceneState blob, recreating entities at their exact handles and re-emplacing each component.
// `r` should be freshly cleared, though the viewport entity may already exist (its components are emplace_or_replace'd).
void RestoreSceneState(entt::registry &, std::span<const std::byte>);
} // namespace snapshot
