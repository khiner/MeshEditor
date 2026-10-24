#include "snapshot/SnapshotRegistration.h"
#include <set>

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
const SnapshotEntries &SnapshotTable() { return GetTables().Snapshots; }

const std::vector<state::TypeId> &TypesOf(Role role) {
    static const auto by_role = [] {
        const auto &entries = SnapshotTable();
        std::array<std::vector<state::TypeId>, 3> types;
        for (state::TypeId type = 0; type < entries.size(); ++type) types[size_t(entries[type].Kind)].push_back(type);
        for (auto &list : types) std::ranges::sort(list, {}, [&](state::TypeId type) { return entries[type].Name; });
        return types;
    }();
    return by_role[size_t(role)];
}

void VerifyCoverage(const state::Registry &r) {
    std::set<std::string> unclassified; // Stable diagnostic ordering.
    for (const auto id : r.Active) {
        if (!r.Tables[id].empty() && !GetTables().Classified[id]) unclassified.emplace(r.Tables[id].Name);
    }
    if (unclassified.empty()) return;

    std::string msg = "snapshot: component(s) in scene storage are classified neither Persistent, Workspace, nor Derived "
                      "(classify in the domain snapshot registration):";
    for (const auto &name : unclassified) (msg += "\n  ") += name;
    throw std::runtime_error(msg);
}

std::optional<bool> ComponentValuesEqual(state::TypeId type_hash, const void *a, const void *b) {
    if (type_hash >= state::SchemaSize || !GetTables().Comparators[type_hash]) return std::nullopt;
    return GetTables().Comparators[type_hash](a, b);
}
} // namespace snapshot
