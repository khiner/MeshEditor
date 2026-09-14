#pragma once

#include "state/Entity.h"

#include <span>
#include <vector>

namespace snapshot {
// A deterministic byte image of the scene's Persistent components (see snapshot/SnapshotRoles.cpp).
std::vector<std::byte> SnapshotSceneState(const state::Scene &);

struct SnapshotDiff {
    bool Equal;
    size_t FirstDifferingByte; // == min size when unequal, == size when equal
};
SnapshotDiff Compare(std::span<const std::byte> expected, std::span<const std::byte> actual);

// Restore persistent components from a SnapshotSceneState blob.
void RestoreSceneState(state::Scene &, std::span<const std::byte>);
} // namespace snapshot
