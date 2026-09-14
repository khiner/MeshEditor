#include "project/EntityStore.h"
#include "state/Allocation.h"
#include "state/Scene.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace project {
store::Live PoolSlots(EntityStore::Pool &);

struct EntityStore::Pool {
    EntityStore &S;
    state::TypeId Type;
    const snapshot::SnapshotEntry &Encoding;
    std::vector<std::byte> Scratch, SnapshotScratch;
    std::vector<bool> RestorePresence;
    store::LiveTrie Trie;

    Pool(EntityStore &s, state::TypeId type, const snapshot::SnapshotEntry &encoding)
        : S(s), Type(type), Encoding(encoding), Trie(PoolSlots(*this), 4, encoding.How == snapshot::Encoding::Bytes ? encoding.Size : 0) {}

    state::TableBase *Storage() const { return S.R.storage(Type); }
    state::Entity Stored(uint32_t index) const {
        const auto *p = Storage();
        return p ? p->entity_at(index) : state::Null;
    }
    bool Present(uint32_t index) const {
        const auto e = Stored(index);
        if (e == state::Null) return false;
        if (S.R.Restoring && Encoding.SkipEntity) return index < RestorePresence.size() && RestorePresence[index];
        return !(Encoding.SkipEntity && Encoding.SkipEntity(S.R, e));
    }
    std::span<const std::byte> Encode(const void *value, std::vector<std::byte> &scratch) const {
        if (Encoding.How != snapshot::Encoding::Serialized) return {static_cast<const std::byte *>(value), Encoding.Size};
        scratch.clear();
        Encoding.Serialize(value, scratch);
        return scratch;
    }
    bool Reusable(uint32_t index) const { return !S.R.Restoring || Stored(index) == S.R.EntityAt(index); }
};

store::Live PoolSlots(EntityStore::Pool &self) {
    store::Live live;
    live.Length = [&self] { return self.S.Table.size(); };
    live.Present = [&self](uint64_t index) { return self.Present(uint32_t(index)); };
    live.Read = [&self](uint64_t index) {
        return self.Encode(self.Storage()->value(self.Stored(uint32_t(index))), self.Scratch);
    };
    live.Replace = [&self](uint64_t index, store::Blob incoming, bool &was_present) {
        was_present = self.Present(uint32_t(index));
        auto old = was_present ? store::Capture(self.Trie.L, index) : store::Blob{};
        const auto previous = self.Stored(uint32_t(index));
        const auto entity = self.S.R.EntityAt(uint32_t(index));
        if (previous != state::Null && previous != entity) self.Storage()->remove(previous);
        if (self.Encoding.SkipEntity) {
            if (self.RestorePresence.size() <= index) self.RestorePresence.resize(index + 1);
            self.RestorePresence[index] = entity != state::Null;
        }
        if (entity != state::Null) {
            if (incoming.Destroy) self.Encoding.Move(self.S.R, entity, incoming);
            else {
                self.Encoding.Emplace(self.S.R, entity, incoming.View());
                store::FreeBlob(incoming);
            }
            self.S.Changes.push_back({self.Type, entity, previous == entity ? state::Event::Update : state::Event::Create});
        } else store::FreeBlob(incoming);
        return old;
    };
    live.Erase = [&self](uint64_t index) {
        const auto old = store::Capture(self.Trie.L, index);
        const auto entity = self.Stored(uint32_t(index));
        if (self.Encoding.SkipEntity && index < self.RestorePresence.size()) self.RestorePresence[index] = false;
        if (entity != state::Null) {
            self.Storage()->remove(entity);
            self.S.Changes.push_back({self.Type, entity, state::Event::Destroy});
        }
        return old;
    };
    live.ForEachPresent = [&self](const std::function<void(uint64_t)> &fn) {
        if (const auto *storage = self.Storage()) {
            for (const auto e : *storage) {
                if (!(self.Encoding.SkipEntity && self.Encoding.SkipEntity(self.S.R, e))) fn(state::Index(e));
            }
        }
    };
    live.Copy = [&self](uint64_t index) {
        return self.Encoding.Copy(self.Storage()->value(self.Stored(uint32_t(index))));
    };
    live.Encode = [&self](const store::Blob &value) { return self.Encode(value.Data, self.SnapshotScratch); };
    live.Reusable = [&self](uint64_t index) { return self.Reusable(uint32_t(index)); };
    live.ForEachNonReusable = [&self](const std::function<void(uint64_t)> &fn) {
        self.S.ForEachIdentityChange([&](uint32_t index) {
            if (!self.Reusable(index) && self.Present(index)) fn(index);
        });
    };
    return live;
}

EntityStore::EntityStore(state::Scene &r, store::History &history, const snapshot::SnapshotEntries &components) : R(r), Components(components), Table(r.AllocationState().Generations) {
    R.HistoryOwner = this;
    R.Capture = [](state::Scene &r, state::TypeId type, state::Entity e) {
        if (!r.Restoring) static_cast<EntityStore *>(r.HistoryOwner)->Capture(type, e);
    };
    Table.Buffer.Trie.CollectChanged = true;
    history.Track(Table.Buffer.Trie, "entity.table", 0);
    history.Track(r.AllocationState().Free.Buffer.Trie, "entity.free", 0);
    for (state::TypeId type = 0; type < components.size(); ++type) {
        const auto &encoding = components[type];
        if (!encoding.Emplace || !encoding.History) continue;
        auto pool = std::make_unique<Pool>(*this, type, encoding);
        history.Track(pool->Trie, "component." + std::string(encoding.Name), 1);
        Pools[type] = std::move(pool);
    }
}

EntityStore::~EntityStore() {
    R.Capture = nullptr;
    R.HistoryOwner = nullptr;
}

void EntityStore::Capture(state::TypeId type, state::Entity e) {
    if (const auto &pool = Pools[type]) {
        const auto &encoding = pool->Encoding;
        if (!encoding.SkipEntity || !encoding.SkipEntity(R, e)) {
            if (R.DocumentReadOnly) throw std::logic_error("Persistent component mutation during history restoration: " + std::string(encoding.Name));
            pool->Trie.Write(state::Index(e), 1);
        }
    }
    if (const auto related = Components[type].CaptureWith; related != state::SchemaSize) Pools[related]->Trie.Write(state::Index(e), 1);
}

void EntityStore::BeginRestore() {
    ++R.Epoch;
    PreviousLength = Table.size();
    for (const auto &pool : Pools) {
        if (!pool || !pool->Encoding.SkipEntity) continue;
        pool->RestorePresence.assign(Table.size(), false);
        for (uint32_t i = 0; i < Table.size(); ++i) pool->RestorePresence[i] = pool->Present(i);
    }
    R.Restoring = true;
    Changes.clear();
    Table.Buffer.Trie.ChangedSlots.clear();
}

void EntityStore::ForEachIdentityChange(auto &&fn) const {
    const auto per_page = Table.Buffer.PageBytes / sizeof(uint32_t);
    const auto end = std::max(Table.size(), PreviousLength);
    for (const auto page : Table.Buffer.Trie.ChangedSlots) {
        for (uint64_t i = page * per_page, last = std::min<uint64_t>(end, (page + 1) * per_page); i < last; ++i) fn(uint32_t(i));
    }
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
