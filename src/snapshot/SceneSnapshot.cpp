#include "snapshot/SceneSnapshot.h"
#include "snapshot/SnapshotRoles.h"

#include "state/Scene.h"

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

std::vector<std::byte> SnapshotSceneState(const state::Scene &r) {
    VerifyCoverage(r);
    const auto &table = SnapshotTable();

    std::vector<std::byte> out;
    // Storage already visits component slots in schema order.
    for (auto [id, set] : r.storage()) {
        const auto &entry = table[id];
        if (set.empty() || !entry.Emplace) continue;
        // Sort by integral entity ID for history-independent output.
        std::vector<state::Entity> ents;
        for (const auto e : set) {
            if (!(entry.SkipEntity && entry.SkipEntity(r, e))) ents.emplace_back(e);
        }
        if (ents.empty()) continue;

        std::ranges::sort(ents, {}, [](state::Entity e) { return state::Integral(e); });

        Append(out, id);
        Append(out, uint32_t(ents.size()));
        for (const auto e : ents) {
            Append(out, state::Integral(e));
            switch (entry.How) {
                case Encoding::Tag: break;
                case Encoding::Bytes: {
                    const auto *p = static_cast<const std::byte *>(set.value(e));
                    out.insert(out.end(), p, p + entry.Size);
                    break;
                }
                case Encoding::Serialized: {
                    // Length-prefix variable-size values for sequential restoration.
                    const auto len_pos = out.size();
                    Append(out, uint32_t(0));
                    entry.Serialize(set.value(e), out);
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
    if (n && std::memcmp(expected.data(), actual.data(), n) != 0) {
        for (size_t i = 0; i < n; ++i) {
            if (expected[i] != actual[i]) return {false, i};
        }
    }
    if (expected.size() != actual.size()) return {false, n};
    return {true, expected.size()};
}

void RestoreSceneState(state::Scene &r, std::span<const std::byte> bytes) {
    const auto &table = SnapshotTable();
    size_t pos = 0;
    const auto read = [&](auto &value) {
        if (pos + sizeof(value) > bytes.size()) return false;
        std::memcpy(&value, bytes.data() + pos, sizeof(value));
        pos += sizeof(value);
        return true;
    };
    while (pos < bytes.size()) {
        state::TypeId hash;
        uint32_t count;
        if (!read(hash) || !read(count)) return;
        if (hash >= table.size() || !table[hash].Emplace) return; // corrupt or stale schema
        const auto &entry = table[hash];
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t entity_bits;
            if (!read(entity_bits)) return;
            const auto e = state::Entity{entity_bits};
            if (!r.valid(e)) {
                [[maybe_unused]] const auto created = r.create(e); // recreate the exact handle (slot is free in a cleared scene)
                assert(created == e);
            }

            std::span<const std::byte> value;
            switch (entry.How) {
                case Encoding::Tag: break;
                case Encoding::Bytes:
                    if (pos + entry.Size > bytes.size()) return;
                    value = bytes.subspan(pos, entry.Size);
                    pos += entry.Size;
                    break;
                case Encoding::Serialized: {
                    uint32_t len;
                    if (!read(len) || pos + len > bytes.size()) return;
                    value = bytes.subspan(pos, len);
                    pos += len;
                    break;
                }
            }
            entry.Emplace(r, e, value);
        }
    }
}
} // namespace snapshot
