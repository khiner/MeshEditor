#pragma once

#include "snapshot/SnapshotSchema.h"
#include "state/Entity.h"
#include "state/Resources.h"

#include <span>
#include <vector>

namespace snapshot {
// Per Workspace type, the values held for identities without them, ordered by identity.
// Retention outlives a registry reset.
struct Retention {
    std::array<state::Resources::Slot, state::SchemaSize> Slots;
};

// Emplaces the retention resource and retains each dying Workspace value under its identity.
void BindRetention(state::Registry &);
// An identity returning through a restore returns with its workspace values.
// A fresh identity evicts the values retained under its key and starts from defaults.
void ReturnRetained(state::Registry &, state::Entity);

// Encodes live and retained Workspace values.
std::vector<std::byte> SerializeWorkspace(const state::Registry &);
// Replaces each value its identity holds, and retains each value of an identity without one.
// Skips unknown types and records that fail to decode.
void ApplyWorkspace(state::Registry &, std::span<const std::byte>);
} // namespace snapshot
