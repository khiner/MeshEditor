#pragma once

#include "project/store/Blob.h"
#include "state/Entity.h"
#include "state/Slot.h"

#include <vector>

namespace state {
struct Table;
} // namespace state

namespace snapshot {
struct NativePage {
    uint32_t Mask{};
};
enum class Role : uint8_t {
    Persistent, // Recorded in history and the project.
    Workspace, // Saved in the workspace file outside history, and retained with its identity.
    Derived, // Rebuilt from Persistent state.
};
enum class Encoding : uint8_t {
    Tag,
    Bytes, // The value's object representation is its encoding.
    Serialized,
};

// An entity identity and its value's encoding.
using WorkspaceRecord = std::pair<state::Entity, std::vector<std::byte>>;

struct SnapshotEntry {
    Encoding How;
    void (*Serialize)(const void *component, std::vector<std::byte> &out);
    void (*Emplace)(state::Registry &, state::Entity, std::span<const std::byte>);
    uint32_t Size{}; // Bytes only.
    std::string_view Name{};
    Role Kind{Role::Derived}; // Unclassified entries read as Derived.
    state::TypeKey Key{};
    void (*Rebind)(state::Registry &, state::Entity previous, state::Entity current){};
    store::Blob (*CopyPage)(const state::Table &, uint32_t page){};
    const void *(*PageValue)(const store::Blob &, uint32_t slot){};
    void (*MovePageValue)(state::Registry &, state::Entity, store::Blob &, uint32_t slot){};
    bool (*Equal)(const void *, const void *){};
    // Workspace only.
    void (*Connect)(state::Registry &){}; // Retains each dying value under its identity.
    void (*Return)(state::Registry &, state::Entity, bool emplace){}; // Takes the identity's retained value, emplacing or replacing it when asked.
    void (*AppendRecords)(const state::Registry &, std::vector<WorkspaceRecord> &){}; // Appends live values in index order, then retained values in identity order.
    void (*Place)(state::Registry &, state::Entity, std::span<const std::byte>){}; // Replaces the value its identity holds, or retains it for an identity without one.
};

// Returns the serializer table for Persistent and Workspace components.
using SnapshotEntries = std::array<SnapshotEntry, state::SchemaSize>;
const SnapshotEntries &SnapshotTable();
// Returns the role's types sorted by name.
const std::vector<state::TypeId> &TypesOf(Role);

// Throws if a live component pool is unclassified.
void VerifyCoverage(const state::Registry &);

// Compares two component values or returns nullopt for unsupported types.
std::optional<bool> ComponentValuesEqual(state::TypeId, const void *, const void *);
} // namespace snapshot
