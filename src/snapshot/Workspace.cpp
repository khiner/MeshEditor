#include "snapshot/Workspace.h"
#include "snapshot/SnapshotRoles.h"
#include "state/Registry.h"

#include <zpp_bits.h>

namespace snapshot {
namespace {
struct Section {
    state::TypeKey Key;
    std::vector<WorkspaceRecord> Records;
};
} // namespace

void CaptureWorkspace(state::Registry &r, state::TypeId type, state::Entity e) {
    if (!r.Capture) return;
    const state::PageMask page{state::Index(e) / state::Table::PageCount, 1u << (state::Index(e) % state::Table::PageCount)};
    r.Capture(r, type, std::span{&page, 1u});
}

void BindRetention(state::Registry &r) {
    r.Resources.emplace<Retention>();
    const auto &entries = SnapshotTable();
    for (const auto type : TypesOf(Role::Workspace)) entries[type].Connect(r);
}

void ReturnRetained(state::Registry &r, state::Entity e) {
    const auto &entries = SnapshotTable();
    for (const auto type : TypesOf(Role::Workspace)) entries[type].Return(r, e, r.Restoring);
}

std::vector<std::byte> SerializeWorkspace(const state::Registry &r) {
    const auto &entries = SnapshotTable();
    std::vector<Section> sections;
    for (const auto type : TypesOf(Role::Workspace)) {
        const auto &entry = entries[type];
        Section section{entry.Key, {}};
        entry.AppendRecords(r, section.Records);
        if (!section.Records.empty()) sections.push_back(std::move(section));
    }
    std::vector<std::byte> bytes;
    zpp::bits::out{bytes}(sections).or_throw();
    return bytes;
}

void ApplyWorkspace(state::Registry &r, std::span<const std::byte> bytes) {
    std::vector<Section> sections;
    zpp::bits::in in{bytes};
    if (zpp::bits::failure(in(sections)) || in.position() != bytes.size()) return;
    const auto &entries = SnapshotTable();
    const auto &types = TypesOf(Role::Workspace);
    for (const auto &[key, records] : sections) {
        const auto type = std::ranges::find(types, key, [&](state::TypeId type) { return entries[type].Key; });
        if (type == types.end()) continue;
        for (const auto &[e, value] : records) entries[*type].Place(r, e, value);
    }
}
} // namespace snapshot
