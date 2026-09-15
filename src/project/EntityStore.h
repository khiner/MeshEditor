#pragma once

#include "project/store/History.h"
#include "snapshot/SnapshotRoles.h"

#include <memory>

namespace store {
template<typename T> struct VersionedVector;
}
namespace state {
enum class Event : uint8_t;
}

namespace project {
struct EntityStore {
    EntityStore(state::Scene &, store::History &, const snapshot::SnapshotEntries &);
    ~EntityStore();

    void Capture(state::TypeId, state::Entity);

    void BeginRestore();
    std::vector<state::Entity> RemovedEntities() const;
    void FinishRestore(std::span<const state::Entity> removed);
    struct Change {
        state::TypeId Type;
        state::Entity Entity;
        state::Event Event;
    };
    std::vector<Change> TakeChanges() { return std::exchange(Changes, {}); }

    struct Pool;
    void ForEachIdentityChange(auto &&fn) const;

    state::Scene &R;
    store::VersionedVector<uint32_t> &Table;
    size_t PreviousLength{};
    std::array<std::unique_ptr<Pool>, state::SchemaSize> Pools;
    std::vector<Change> Changes;
};
} // namespace project
