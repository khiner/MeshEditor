#include "snapshot/SnapshotRegistration.h"
#include <set>
#include <string>

namespace snapshot {
namespace {
const detail::Tables &GetTables() {
    static const auto tables = [] {
        detail::Tables result;
        detail::RegisterAudio(result);
        detail::RegisterPhysics(result);
        detail::RegisterArmature(result);
        detail::RegisterMesh(result);
        detail::RegisterViewport(result);
        detail::RegisterAssets(result);
        detail::RegisterScene(result);
        return result;
    }();
    return tables;
}
} // namespace
const std::unordered_map<entt::id_type, SnapshotEntry> &SnapshotTable() { return GetTables().Snapshots; }

void VerifyCoverage(const entt::registry &r) {
    std::set<std::string> unclassified; // Stable diagnostic ordering.
    for (auto [id, set] : r.storage()) {
        if (set.empty()) continue;
        const auto &info = set.info();
        if (!std::string_view{info.name()}.starts_with("entt::")) {
            if (!GetTables().Comparators.contains(info.hash())) unclassified.emplace(info.name());
        }
    }
    if (unclassified.empty()) return;

    std::string msg = "snapshot: component(s) in registry storage are classified neither Persistent nor Derived "
                      "(classify in the domain snapshot registration):";
    for (const auto &name : unclassified) (msg += "\n  ") += name;
    throw std::runtime_error(msg);
}

std::optional<bool> ComponentValuesEqual(entt::id_type type_hash, const void *a, const void *b) {
    const auto &comparators = GetTables().Comparators;
    const auto it = comparators.find(type_hash);
    if (it == comparators.end() || it->second == nullptr) return std::nullopt;
    return it->second(a, b);
}
} // namespace snapshot
