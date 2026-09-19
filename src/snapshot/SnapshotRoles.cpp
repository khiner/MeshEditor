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
const SnapshotEntries &SnapshotTable() { return GetTables().Snapshots; }

void VerifyCoverage(const state::Scene &r) {
    std::set<std::string> unclassified; // Stable diagnostic ordering.
    for (const auto id : r.Active)
        if (!r.Tables[id].empty() && !GetTables().Classified[id]) unclassified.emplace(state::SchemaNames[id]);
    if (unclassified.empty()) return;

    std::string msg = "snapshot: component(s) in scene storage are classified neither Persistent nor Derived "
                      "(classify in the domain snapshot registration):";
    for (const auto &name : unclassified) (msg += "\n  ") += name;
    throw std::runtime_error(msg);
}

std::optional<bool> ComponentValuesEqual(state::TypeId type_hash, const void *a, const void *b) {
    if (type_hash >= state::SchemaSize || !GetTables().Comparators[type_hash]) return std::nullopt;
    return GetTables().Comparators[type_hash](a, b);
}
} // namespace snapshot
