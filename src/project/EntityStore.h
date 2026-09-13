#pragma once

#include "project/Registry.h"
#include "project/store/History.h"
#include "project/store/Versioned.h"
#include "snapshot/SnapshotRoles.h"

#include <memory>

namespace project {
struct EntityStore {
    EntityStore(entt::registry &, store::History &, const std::unordered_map<entt::id_type, snapshot::SnapshotEntry> &);
    ~EntityStore();

    entt::entity Create();
    void Destroy(entt::entity);
    void Capture(entt::id_type, entt::entity);

    // Destroys all live entities and resets allocation, preserving pinned versions.
    void Reset();
    void BeginRestore();
    std::vector<entt::entity> RemovedEntities() const;
    void FinishRestore();
    entt::entity Recorded(uint32_t index) const;
    struct Change {
        entt::id_type Type;
        entt::entity Entity;
    };
    std::vector<Change> TakeChanges() { return std::exchange(Changes, {}); }

    struct Pool;
    struct Pending {
        Pool *Target;
        uint32_t Index;
        store::Blob Value;
        bool Erase;
    };
    entt::entity LiveAt(uint32_t index) const;
    void ForEachIdentityChange(auto &&fn) const;

    entt::registry &R;
    store::VersionedVector<uint32_t> Table, Free;
    std::vector<entt::entity> Live;
    std::unordered_map<entt::id_type, std::unique_ptr<Pool>> Pools;
    std::vector<Pending> Staged;
    std::vector<Change> Changes;
    bool InRestore{};
};
} // namespace project
