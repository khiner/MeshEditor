#include "project/EntityStore.h"

#include "armature/ArmatureComponents.h"
#include "gpu/Transform.h"
#include "viewport/ViewCamera.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace project {
namespace {
using Traits = entt::entt_traits<entt::entity>;
constexpr uint32_t Alive = 1u << 31;
EntityStore *HistoryOf(const entt::registry &r) {
    const auto *history = r.ctx().find<EntityStore *>();
    return history ? *history : nullptr;
}
} // namespace

store::Live PoolSlots(EntityStore::Pool &);

struct EntityStore::Pool {
    EntityStore &S;
    entt::id_type Type;
    const snapshot::SnapshotEntry &Encoding;
    std::vector<std::byte> Scratch;
    store::LiveTrie Trie;

    Pool(EntityStore &s, entt::id_type type, const snapshot::SnapshotEntry &encoding)
        : S(s), Type(type), Encoding(encoding), Trie(PoolSlots(*this), 4, encoding.How == snapshot::Encoding::Bytes ? encoding.Size : 0) {}

    const entt::sparse_set *Storage() const { return std::as_const(S.R).storage(Type); }
    bool Present(uint32_t index) const {
        const auto e = S.LiveAt(index);
        const auto *storage = Storage();
        return e != entt::null && storage && storage->contains(e) && !(Encoding.SkipEntity && Encoding.SkipEntity(S.R, e));
    }
    bool Reusable(uint32_t index) const { return !S.InRestore || S.LiveAt(index) == S.Recorded(index); }
};

store::Live PoolSlots(EntityStore::Pool &self) {
    store::Live live;
    live.Length = [&self] { return self.S.Table.size(); };
    live.Present = [&self](uint64_t index) { return self.Present(uint32_t(index)); };
    live.Read = [&self](uint64_t index) -> std::span<const std::byte> {
        const auto *value = self.Storage()->value(self.S.LiveAt(uint32_t(index)));
        switch (self.Encoding.How) {
            case snapshot::Encoding::Tag: return {};
            case snapshot::Encoding::Bytes: return {static_cast<const std::byte *>(value), self.Encoding.Size};
            case snapshot::Encoding::Serialized:
                self.Scratch.clear();
                self.Encoding.Serialize(value, self.Scratch);
                return self.Scratch;
        }
        std::unreachable();
    };
    live.Replace = [&self](uint64_t index, store::Blob incoming, bool &was_present) {
        was_present = self.Present(uint32_t(index));
        const auto old = was_present ? store::Capture(self.Trie.L, index) : store::Blob{};
        self.S.Staged.push_back({&self, uint32_t(index), incoming, false});
        return old;
    };
    live.Erase = [&self](uint64_t index) {
        const auto old = store::Capture(self.Trie.L, index);
        self.S.Staged.push_back({&self, uint32_t(index), {}, true});
        return old;
    };
    live.ForEachPresent = [&self](const std::function<void(uint64_t)> &fn) {
        if (const auto *storage = self.Storage()) {
            for (const auto e : *storage) {
                if (e != entt::tombstone && !(self.Encoding.SkipEntity && self.Encoding.SkipEntity(self.S.R, e))) fn(Traits::to_entity(e));
            }
        }
    };
    live.Reusable = [&self](uint64_t index) { return self.Reusable(uint32_t(index)); };
    live.ForEachNonReusable = [&self](const std::function<void(uint64_t)> &fn) {
        self.S.ForEachIdentityChange([&](uint32_t index) {
            if (!self.Reusable(index) && self.Present(index)) fn(index);
        });
    };
    return live;
}

bool Restoring(const entt::registry &r) {
    const auto *history = HistoryOf(r);
    return history && history->InRestore;
}
void Capture(entt::registry &r, entt::id_type type, entt::entity e) {
    if (auto *history = HistoryOf(r)) history->Capture(type, e);
}
entt::entity Create(entt::registry &r) {
    if (auto *history = HistoryOf(r)) return history->Create();
    return r.create();
}
void Destroy(entt::registry &r, entt::entity e) {
    if (auto *history = HistoryOf(r)) history->Destroy(e);
    else r.destroy(e);
}
void Reset(entt::registry &r) {
    if (auto *history = HistoryOf(r)) history->Reset();
    else {
        r.storage<entt::entity>().clear();
        r.storage<entt::entity>().start_from(entt::entity{0});
    }
}

EntityStore::EntityStore(entt::registry &r, store::History &history, const std::unordered_map<entt::id_type, snapshot::SnapshotEntry> &components) : R(r) {
    R.ctx().emplace<EntityStore *>(this);
    Table.Buffer.Trie.CollectChanged = true;
    history.Track(Table.Buffer.Trie, "entity.table", 0);
    history.Track(Free.Buffer.Trie, "entity.free", 0);
    std::vector<entt::id_type> types;
    for (const auto &[type, encoding] : components) {
        if (type != entt::type_hash<ViewCamera>::value()) types.push_back(type);
    }
    std::ranges::sort(types);
    for (const auto type : types) {
        const auto &encoding = components.at(type);
        auto pool = std::make_unique<Pool>(*this, type, encoding);
        history.Track(pool->Trie, "component." + std::string(encoding.Name), 1);
        Pools.emplace(type, std::move(pool));
    }
}

EntityStore::~EntityStore() {
    for (auto &op : Staged) store::FreeBlob(op.Value);
    R.ctx().erase<EntityStore *>();
}

entt::entity EntityStore::LiveAt(uint32_t index) const { return index < Live.size() ? Live[index] : entt::null; }

entt::entity EntityStore::Recorded(uint32_t index) const {
    return index < Table.size() && (Table[index] & Alive) ? Traits::construct(index, Traits::version_type(Table[index] & ~Alive)) : entt::null;
}

entt::entity EntityStore::Create() {
    uint32_t index;
    if (Free.empty()) {
        index = uint32_t(Table.size());
        Table.PushBack(0);
    } else {
        index = Free.Back();
        Free.PopBack();
    }
    Table.Set(index, Table[index] | Alive);
    const auto e = Recorded(index);
    const auto created = R.create(e);
    assert(created == e);
    if (Live.size() <= index) Live.resize(size_t(index) + 1, entt::null);
    Live[index] = created;
    return created;
}

void EntityStore::Capture(entt::id_type type, entt::entity e) {
    if (const auto it = Pools.find(type); it != Pools.end()) {
        const auto &encoding = it->second->Encoding;
        if (!encoding.SkipEntity || !encoding.SkipEntity(R, e)) {
            if (InRestore) throw std::logic_error("Persistent component mutation during history restoration");
            it->second->Trie.Write(Traits::to_entity(e), 1);
        }
    }
    // Transform is Persistent only on entities without BoneIndex.
    if (type == entt::type_hash<BoneIndex>::value()) Pools.at(entt::type_hash<Transform>::value())->Trie.Write(Traits::to_entity(e), 1);
}

void EntityStore::Destroy(entt::entity e) {
    assert(LiveAt(Traits::to_entity(e)) == e);
    for (auto &[type, pool] : Pools) {
        if (pool->Present(Traits::to_entity(e))) Capture(type, e);
    }
    R.destroy(e);
    const auto index = Traits::to_entity(e);
    Live[index] = entt::null;
    Table.Set(index, Traits::to_version(Traits::next(e)));
    Free.PushBack(index);
}

void EntityStore::Reset() {
    for (const auto e : Live) {
        if (e != entt::null && R.valid(e)) Destroy(e);
    }
    Table.Clear();
    Free.Clear();
    Live.clear();
    R.storage<entt::entity>().clear();
}

void EntityStore::BeginRestore() {
    InRestore = true;
    Changes.clear();
    Table.Buffer.Trie.ChangedSlots.clear();
}

void EntityStore::ForEachIdentityChange(auto &&fn) const {
    const auto per_page = Table.Buffer.PageBytes / sizeof(uint32_t);
    const auto end = std::max(Table.size(), Live.size());
    for (const auto page : Table.Buffer.Trie.ChangedSlots) {
        for (uint64_t i = page * per_page, last = std::min<uint64_t>(end, (page + 1) * per_page); i < last; ++i) fn(uint32_t(i));
    }
}

std::vector<entt::entity> EntityStore::RemovedEntities() const {
    std::vector<entt::entity> removed;
    ForEachIdentityChange([&](uint32_t index) {
        const auto e = LiveAt(index);
        if (e != entt::null && e != Recorded(index)) removed.push_back(e);
    });
    return removed;
}

void EntityStore::FinishRestore() {
    if (Live.size() < Table.size()) Live.resize(Table.size(), entt::null);
    ForEachIdentityChange([&](uint32_t index) {
        const auto want = Recorded(index);
        if (Live[index] != entt::null && Live[index] != want) {
            R.destroy(Live[index]);
            Live[index] = entt::null;
        }
    });
    ForEachIdentityChange([&](uint32_t index) {
        const auto want = Recorded(index);
        if (want != entt::null && Live[index] == entt::null) {
            Live[index] = R.create(want);
            assert(Live[index] == want);
        }
    });
    for (const auto &op : Staged) {
        const auto e = LiveAt(op.Index);
        if (op.Erase && e != entt::null) {
            if (auto *storage = R.storage(op.Target->Type)) storage->remove(e);
        }
    }
    for (auto &op : Staged) {
        const auto e = LiveAt(op.Index);
        if (!op.Erase && e != entt::null) op.Target->Encoding.Emplace(R, e, op.Value.View());
        if (e != entt::null) Changes.push_back({op.Target->Type, e});
        store::FreeBlob(op.Value);
    }
    Staged.clear();
    InRestore = false;
}
} // namespace project
