#include "project/EntityStore.h"
#include "project/ComponentPool.h"
#include "project/store/History.h"
#include "state/Allocation.h"
#include "state/Scene.h"

#include <stdexcept>

namespace project {
EntityStore::EntityStore(state::Scene &r, store::History &history, const snapshot::SnapshotEntries &components) : R(r), Table(r.AllocationState().Generations) {
    R.HistoryOwner = this;
    R.Capture = [](state::Scene &r, state::TypeId type, state::Entity e) {
        if (!r.Restoring) static_cast<EntityStore *>(r.HistoryOwner)->Capture(type, e);
    };
    Table.P.Trie.CollectChanged = true;
    history.Track(Table.P, "entity.table", 0);
    history.Track(r.AllocationState().Free.P, "entity.free", 0);
    // Track pools in name order so the persisted track layout is independent of schema slot numbers.
    std::vector<state::TypeId> tracked;
    for (state::TypeId type = 0; type < components.size(); ++type)
        if (components[type].Emplace && components[type].History) tracked.push_back(type);
    std::ranges::sort(tracked, {}, [&](state::TypeId type) { return components[type].Name; });
    for (const auto type : tracked) {
        auto pool = std::make_unique<ComponentPool>(*this, type, components[type]);
        history.Track(*pool, "component." + std::string(components[type].Name), 1);
        Pools[type] = std::move(pool);
    }
}

EntityStore::~EntityStore() {
    R.Capture = nullptr;
    R.HistoryOwner = nullptr;
}

void EntityStore::Capture(state::TypeId type, state::Entity e) {
    if (const auto &pool = Pools[type]) {
        if (R.DocumentReadOnly) throw std::logic_error("Persistent component mutation during history restoration: " + std::string(pool->Encoding.Name));
        pool->Capture(state::Index(e));
    }
}

void EntityStore::BeginRestore() {
    ++R.Epoch;
    PreviousLength = Table.size();
    R.Restoring = true;
    Changes.clear();
    Table.P.Trie.ChangedSlots.clear();
}

std::vector<state::Entity> EntityStore::RemovedEntities() const {
    std::vector<state::Entity> removed;
    ForEachIdentityChange([&](uint32_t index) {
        const auto e = R.Living.entity_at(index);
        if (e != state::Null && e != R.EntityAt(index)) removed.push_back(e);
    });
    return removed;
}

void EntityStore::FinishRestore(std::span<const state::Entity> removed) {
    for (const auto e : removed) R.RemoveComponents(e);
    R.RestoringEvents = true;
    for (const auto &[type, e, event] : Changes)
        if (event != state::Event::Destroy) state::Notify(R, type, event, e);
    R.RestoringEvents = false;
    R.Restoring = false;
    R.RebuildLiving();
}
} // namespace project
