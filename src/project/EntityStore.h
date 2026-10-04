#pragma once

#include "project/store/Pages.h"
#include "snapshot/SnapshotRoles.h"

#include <algorithm>
#include <memory>

namespace store {
struct History;
}
namespace state {
enum class Event : uint8_t;
struct PageMask;
}

namespace project {
struct ComponentPool;

struct EntityStore {
    EntityStore(state::Scene &, store::History &, const snapshot::SnapshotEntries &);
    ~EntityStore();

    void Capture(state::TypeId, std::span<const state::PageMask>);

    void BeginRestore();
    std::vector<state::Entity> RemovedEntities() const;
    void FinishRestore(std::span<const state::Entity> removed);
    struct Change {
        state::TypeId Type;
        state::Entity Entity;
        state::Event Event;
    };
    std::vector<Change> TakeChanges() { return std::exchange(Changes, {}); }

    // Visit the entity index range [first, last) of each generation page that changed during the current restoration.
    void ForEachIdentityChangeRun(auto &&fn) const {
        const auto per_page = Table.P.PageBytes / sizeof(uint32_t);
        const auto end = std::max(Table.size(), PreviousLength);
        for (const auto page : Table.P.Trie.ChangedSlots) {
            const auto first = page * per_page;
            if (first < end) fn(uint32_t(first), uint32_t(std::min<uint64_t>(end, (page + 1) * per_page)));
        }
    }
    // Visit every entity index whose generation page changed during the current restoration.
    void ForEachIdentityChange(auto &&fn) const {
        ForEachIdentityChangeRun([&](uint32_t first, uint32_t last) {
            for (auto i = first; i < last; ++i) fn(i);
        });
    }

    state::Scene &R;
    store::VersionedVector<uint32_t> &Table;
    size_t PreviousLength{};
    std::array<std::unique_ptr<ComponentPool>, state::SchemaSize> Pools;
    std::vector<Change> Changes;
};
} // namespace project
