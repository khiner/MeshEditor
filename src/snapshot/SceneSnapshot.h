#pragma once

#include <entt/entity/fwd.hpp>

#include <span>
#include <vector>

namespace snapshot {
// A deterministic byte image of the registry's Persistent components (see scene/SceneSnapshotRoles.cpp).
std::vector<std::byte> SnapshotSceneState(const entt::registry &);

struct SnapshotDiff {
    bool Equal;
    size_t FirstDifferingByte; // == min size when unequal, == size when equal
};
SnapshotDiff Compare(std::span<const std::byte> expected, std::span<const std::byte> actual);

// Restore persistent components from a SnapshotSceneState blob.
void RestoreSceneState(entt::registry &, std::span<const std::byte>);
} // namespace snapshot
