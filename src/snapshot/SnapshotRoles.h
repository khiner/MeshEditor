#pragma once

#include <entt/entity/fwd.hpp>

#include <optional>
#include <span>
#include <unordered_map>
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
    void (*Emplace)(entt::registry &, entt::entity, std::span<const std::byte>);
    bool (*SkipEntity)(const entt::registry &, entt::entity);
};

// Returns the serializer table for Persistent components.
const std::unordered_map<entt::id_type, SnapshotEntry> &SnapshotTable();

// Throws if a live component pool is absent from Persistent and Derived.
void VerifyCoverage(const entt::registry &);

// Compares two component values or returns nullopt for unsupported types.
std::optional<bool> ComponentValuesEqual(entt::id_type, const void *, const void *);
} // namespace snapshot
