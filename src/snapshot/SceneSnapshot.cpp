#include "snapshot/SceneSnapshot.h"
#include "snapshot/SnapshotRoles.h"

#include <entt/entity/registry.hpp>

#include <algorithm>
#include <cassert>

namespace snapshot {
namespace {
template<typename T>
void Append(std::vector<std::byte> &out, const T &value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto *p = reinterpret_cast<const std::byte *>(&value);
    out.insert(out.end(), p, p + sizeof(T));
}
} // namespace

std::vector<std::byte> SnapshotSceneState(const entt::registry &r) {
    VerifyCoverage(r); // hard fail if any present component is unclassified, before it silently drops from the image
    const auto &table = SnapshotTable();

    // Collect the Persistent pools, sorted by type hash so section order doesn't depend on storage creation order.
    // Skip empty pools: once entt creates a pool it lingers even when empty, so its presence reflects registry history, not state.
    // Including it would make the byte image history-dependent.
    std::vector<std::pair<entt::id_type, const entt::sparse_set *>> pools;
    for (auto [id, set] : r.storage()) {
        if (!set.empty() && table.contains(id)) pools.emplace_back(id, &set);
    }
    std::ranges::sort(pools, {}, [](const auto &p) { return p.first; });

    std::vector<std::byte> out;
    for (const auto [id, set] : pools) {
        const auto &entry = table.at(id);
        // Skip tombstones (in-place-delete leaves them in iteration) and omit pools with no live entities, since both
        // reflect deletion history, not state. Sort by integral id so byte order doesn't depend on insertion order.
        std::vector<entt::entity> ents;
        for (const auto e : *set) {
            if (e != entt::tombstone && !SnapshotSkipsEntity(r, e) && !(entry.SkipEntity && entry.SkipEntity(r, e))) ents.emplace_back(e);
        }
        if (ents.empty()) continue;

        std::ranges::sort(ents, {}, [](entt::entity e) { return entt::to_integral(e); });

        Append(out, id);
        Append(out, uint32_t(ents.size()));
        for (const auto e : ents) {
            Append(out, entt::to_integral(e));
            switch (entry.How) {
                case Encoding::Tag: break;
                case Encoding::Bytes: {
                    const auto *p = static_cast<const std::byte *>(set->value(e));
                    out.insert(out.end(), p, p + entry.Size);
                    break;
                }
                case Encoding::Serialized: {
                    // Length-prefix so the restore reader can advance past a variable-length value.
                    const auto len_pos = out.size();
                    Append(out, uint32_t(0));
                    entry.Serialize(set->value(e), out);
                    const auto len = uint32_t(out.size() - len_pos - sizeof(uint32_t));
                    std::memcpy(out.data() + len_pos, &len, sizeof(len));
                    break;
                }
            }
        }
    }
    return out;
}

SnapshotDiff Compare(std::span<const std::byte> expected, std::span<const std::byte> actual) {
    const auto n = std::min(expected.size(), actual.size());
    for (size_t i = 0; i < n; ++i) {
        if (expected[i] != actual[i]) return {false, i};
    }
    if (expected.size() != actual.size()) return {false, n};
    return {true, expected.size()};
}

void RestoreSceneState(entt::registry &r, std::span<const std::byte> bytes) {
    const auto &table = SnapshotTable();
    size_t pos = 0;
    const auto read = [&](auto &value) {
        if (pos + sizeof(value) > bytes.size()) return false;
        std::memcpy(&value, bytes.data() + pos, sizeof(value));
        pos += sizeof(value);
        return true;
    };
    while (pos < bytes.size()) {
        entt::id_type hash;
        uint32_t count;
        if (!read(hash) || !read(count)) return;
        const auto it = table.find(hash);
        if (it == table.end()) return; // a section whose type isn't registered: corrupt or stale
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t entity_bits;
            if (!read(entity_bits)) return;
            const auto e = entt::entity{entity_bits};
            if (!r.valid(e)) {
                [[maybe_unused]] const auto created = r.create(e); // recreate the exact handle (slot is free in a cleared registry)
                assert(created == e);
            }

            std::span<const std::byte> value;
            switch (it->second.How) {
                case Encoding::Tag: break;
                case Encoding::Bytes:
                    if (pos + it->second.Size > bytes.size()) return;
                    value = bytes.subspan(pos, it->second.Size);
                    pos += it->second.Size;
                    break;
                case Encoding::Serialized: {
                    uint32_t len;
                    if (!read(len) || pos + len > bytes.size()) return;
                    value = bytes.subspan(pos, len);
                    pos += len;
                    break;
                }
            }
            it->second.Emplace(r, e, value);
        }
    }
}
} // namespace snapshot
