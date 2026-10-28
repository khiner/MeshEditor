#include "project/WorkspacePool.h"
#include "snapshot/Workspace.h"
#include "state/Registry.h"
#include <algorithm>

namespace project {
WorkspacePool::WorkspacePool(state::Registry &r, const snapshot::SnapshotEntries &entries) : R(r), Entries(entries) {
    for (state::TypeId i = 0; i < entries.size(); ++i)
        if (entries[i].Kind == snapshot::Role::Workspace) Types.push_back(i);
    std::ranges::sort(Types, {}, [&](auto type) { return entries[type].Name; });
    Ordinals.fill(UINT32_MAX);
    for (uint32_t i = 0; i < Types.size(); ++i) Ordinals[Types[i]] = i;
}
std::span<const std::byte> WorkspacePool::Read(uint64_t slot) {
    Entries[Types[slot % Types.size()]].ReadWorkspacePage(R, uint32_t(slot / Types.size()), Scratch);
    return Scratch;
}
std::span<const std::byte> WorkspacePool::Encode(const store::Blob &blob) {
    if (!blob.Destroy) return blob.View();
    Entries[reinterpret_cast<const snapshot::WorkspacePageHeader *>(blob.Data)->Type].EncodeWorkspacePage(blob, Scratch);
    return Scratch;
}
store::Blob WorkspacePool::Copy(uint64_t slot) {
    return Entries[Types[slot % Types.size()]].CopyWorkspacePage(R, uint32_t(slot / Types.size()));
}
void WorkspacePool::Capture(state::TypeId type, std::span<const state::PageMask> pages) {
    if (Ordinals[type] == UINT32_MAX || R.Resources.get<const snapshot::Retention>().Installing) return;
    for (const auto &page : pages) {
        const auto slot = uint64_t(page.Page) * Types.size() + Ordinals[type];
        Trie.Capture(slot, 1, this, [](void *p, uint64_t s) -> std::optional<store::Blob> {
            auto value = static_cast<WorkspacePool *>(p)->Copy(s);
            return value.Data ? std::optional{value} : std::nullopt;
        });
        Trie.MarkDirty(slot, 1);
        Extent = std::max(Extent, slot + 1);
    }
}
void WorkspacePool::Settle() {
    for (const auto slot : Trie.Dirty()) {
        const auto bytes = Read(slot);
        Trie.Rehash(slot, bytes.empty() ? std::nullopt : std::optional{bytes});
    }
    Trie.ClearDirty();
}
void WorkspacePool::Apply(store::RestorePlan &plan) {
    auto &installing = R.Resources.get<snapshot::Retention>().Installing;
    installing = true;
    for (auto &change : plan.Changes) {
        change.Old = Copy(change.Slot);
        change.WasPresent = change.Old.Data != nullptr;
        Entries[Types[change.Slot % Types.size()]].ApplyWorkspacePage(R, uint32_t(change.Slot / Types.size()), change.Incoming);
        store::FreeBlob(change.Incoming);
    }
    installing = false;
    Extent = plan.Length;
}
bool WorkspacePool::Restore(const store::Version &version) {
    Settle();
    auto plan = Trie.PlanRestore(version);
    Apply(plan);
    return Trie.CommitRestore(std::move(plan));
}
void WorkspacePool::Load(uint64_t length, std::span<const std::pair<uint64_t, store::Hash128>> changes, const std::unordered_map<store::Hash128, std::vector<std::byte>, store::Hash128Hasher> &leaves) {
    Settle();
    auto plan = Trie.PlanLoad(length, changes, leaves);
    Apply(plan);
    Trie.CommitLoad(std::move(plan));
}
} // namespace project
