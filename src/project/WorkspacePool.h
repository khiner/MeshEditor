#pragma once

#include "project/store/LiveTrie.h"
#include "snapshot/SnapshotRoles.h"

namespace state {
struct PageMask;
}
namespace project {
// One trie over workspace component pages, including values retained for absent owners.
struct WorkspacePool {
    WorkspacePool(state::Registry &, const snapshot::SnapshotEntries &);
    void Capture(state::TypeId, std::span<const state::PageMask>);
    void Settle();
    uint64_t Length() const { return Extent; }
    std::span<const std::byte> Read(uint64_t);
    std::span<const std::byte> Encode(const store::Blob &);
    bool Present(uint64_t slot) { return !Read(slot).empty(); }
    bool Restore(const store::Version &);
    void Load(uint64_t, std::span<const std::pair<uint64_t, store::Hash128>>, const std::unordered_map<store::Hash128, std::vector<std::byte>, store::Hash128Hasher> &);
    store::Version Pin() {
        Settle();
        return Trie.Pin(Extent);
    }
    state::Registry &R;
    const snapshot::SnapshotEntries &Entries;
    std::vector<state::TypeId> Types;
    std::array<uint32_t, state::SchemaSize> Ordinals{};
    store::LiveTrie Trie{4};
    uint64_t Extent{};
    std::vector<std::byte> Scratch;

private:
    store::Blob Copy(uint64_t);
    void Apply(store::RestorePlan &);
};
} // namespace project
