#pragma once

#include "project/store/LiveTrie.h"
#include "snapshot/SnapshotRoles.h"

namespace state {
struct Table;
struct PageMask;
}

namespace project {
struct EntityStore;

// Versioned component pages of one type, using the scene table's occupancy mask.
// Values stay in the scene table; history copies one native page before its first write.
struct ComponentPool {
    ComponentPool(EntityStore &, state::TypeId, const snapshot::SnapshotEntry &);

    uint64_t Length() const;
    bool Present(uint64_t page) const;
    // The returned span remains valid until the next Read.
    std::span<const std::byte> Read(uint64_t page);
    std::span<const std::byte> Encode(const store::Blob &);

    // Capture affected native pages before external writes.
    void Capture(std::span<const state::PageMask>);
    void Settle();
    bool Restore(const store::Version &);
    void Load(uint64_t length, std::span<const std::pair<uint64_t, store::Hash128>> changes, const std::unordered_map<store::Hash128, std::vector<std::byte>, store::Hash128Hasher> &leaves);

    EntityStore &S;
    state::TypeId Type;
    const snapshot::SnapshotEntry &Encoding;
    std::vector<std::byte> Scratch, PageScratch, SnapshotScratch;
    store::LiveTrie Trie;

private:
    state::Table &Storage() const;
    state::Entity Stored(uint32_t index) const;
    store::Blob Copy(uint32_t page);
    void Apply(store::RestorePlan &, bool compare);
};
} // namespace project
