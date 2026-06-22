#pragma once

#include <entt/entity/fwd.hpp>

#include <span>
#include <unordered_map>
#include <vector>

namespace snapshot {
enum class Encoding : uint8_t {
    Tag, // Empty component: presence only, no value bytes (the entity id carries the info).
    Bytes, // Trivially copyable: memcpy `Size` bytes.
    Serialized, // Holds heap data (string/vector/...): call `Serialize`.
};

struct SnapshotEntry {
    Encoding How;
    uint32_t Size; // value bytes for Encoding::Bytes
    void (*Serialize)(const void *component, std::vector<std::byte> &out); // for Encoding::Serialized
    void (*Emplace)(entt::registry &, entt::entity, std::span<const std::byte>); // inverse: decode + emplace_or_replace
    bool (*SkipEntity)(const entt::registry &, entt::entity); // optional: skip entities whose value is derived (null = serialize all)
};

// Persistent-component serializer table, built lazily from the hardcoded Persistent list on first use.
const std::unordered_map<entt::id_type, SnapshotEntry> &SnapshotTable();

// Throws if any live component pool is classified neither Persistent nor Derived.
void VerifyCoverage(const entt::registry &);
} // namespace snapshot
