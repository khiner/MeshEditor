#pragma once

#include "project/store/Blob.h"
#include "state/Entity.h"
#include "state/Schema.h"

#include <array>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace snapshot {
enum class Encoding : uint8_t {
    Tag,
    Bytes,
    Serialized,
};

struct SnapshotEntry {
    Encoding How;
    uint32_t Size;
    void (*Serialize)(const void *component, std::vector<std::byte> &out);
    void (*Emplace)(state::Scene &, state::Entity, std::span<const std::byte>);
    std::string_view Name{};
    bool History{true}; // Workspace-only values stay out of history.
    store::Blob (*Copy)(const void *){};
    void (*Move)(state::Scene &, state::Entity, store::Blob){};
};

// Returns the serializer table for Persistent components.
using SnapshotEntries = std::array<SnapshotEntry, state::SchemaSize>;
const SnapshotEntries &SnapshotTable();

// Throws if a live component pool is absent from Persistent and Derived.
void VerifyCoverage(const state::Scene &);

// Compares two component values or returns nullopt for unsupported types.
std::optional<bool> ComponentValuesEqual(state::TypeId, const void *, const void *);
} // namespace snapshot
