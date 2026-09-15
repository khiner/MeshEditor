#pragma once

#include "project/store/LiveTrie.h"
#include "snapshot/SnapshotRoles.h"

namespace state {
struct TableBase;
}

namespace project {
struct EntityStore;

// Versioned component values of one type, keyed by entity index.
// Values live in the scene table. Restoration emplaces, moves, and removes them through the snapshot encoding.
struct ComponentPool {
    ComponentPool(EntityStore &, state::TypeId, const snapshot::SnapshotEntry &);

    uint64_t Length() const;
    bool Present(uint64_t index) const;
    // The returned span remains valid until the next Read.
    std::span<const std::byte> Read(uint64_t index);
    // Serialize a native copy, or view raw bytes. The returned span remains valid until the next Encode.
    std::span<const std::byte> Encode(const store::Blob &);

    // Capture the component at index before an external write.
    void Capture(uint32_t index);
    void Settle();
    bool Restore(const store::Version &);
    void Load(uint64_t length, std::span<const std::pair<uint64_t, store::Hash128>> changes, const std::unordered_map<store::Hash128, std::vector<std::byte>, store::Hash128Hasher> &leaves);

    EntityStore &S;
    state::TypeId Type;
    const snapshot::SnapshotEntry &Encoding;
    std::vector<std::byte> Scratch, SnapshotScratch;
    store::LiveTrie Trie;

private:
    state::TableBase *Storage() const;
    state::Entity Stored(uint32_t index) const;
    store::Blob Copy(uint32_t index) const;
    void Apply(store::RestorePlan &, bool compare);
};
} // namespace project
