#include "project/ComponentPool.h"
#include "project/EntityStore.h"
#include "state/Scene.h"

#include <cassert>

namespace project {
namespace {
std::span<const std::byte> Serialize(const snapshot::SnapshotEntry &encoding, const void *value, std::vector<std::byte> &scratch) {
    scratch.clear();
    if (encoding.How == snapshot::Encoding::Serialized) encoding.Serialize(value, scratch);
    return scratch;
}
} // namespace

ComponentPool::ComponentPool(EntityStore &s, state::TypeId type, const snapshot::SnapshotEntry &encoding)
    : S(s), Type(type), Encoding(encoding), Trie(4) {}

state::Table &ComponentPool::Storage() const { return S.R.storage(Type); }
state::Entity ComponentPool::Stored(uint32_t index) const { return Storage().entity_at(index); }
store::Blob ComponentPool::Copy(uint32_t index) const { return Encoding.Copy(Storage().value(Stored(index))); }

uint64_t ComponentPool::Length() const { return S.Table.size(); }
bool ComponentPool::Present(uint64_t index) const { return Stored(uint32_t(index)) != state::Null; }
std::span<const std::byte> ComponentPool::Read(uint64_t index) {
    return Serialize(Encoding, Storage().value(Stored(uint32_t(index))), Scratch);
}
std::span<const std::byte> ComponentPool::Encode(const store::Blob &value) {
    return value.Destroy ? Serialize(Encoding, value.Data, SnapshotScratch) : value.View();
}

void ComponentPool::Capture(uint32_t index) {
    assert(!Trie.ExternalWritesForbidden && "component write during track restoration");
    if (Trie.Uncaptured(index)) Trie.Capture(index, Present(index) ? std::optional{Copy(index)} : std::nullopt);
    Trie.MarkDirty(index, 1);
}

void ComponentPool::Settle() {
    for (const auto index : Trie.Dirty()) Trie.Rehash(index, Present(index) ? std::optional{Read(index)} : std::nullopt);
    Trie.ClearDirty();
}

// Values move to the entity now at their index. A changed entity generation never counts as unchanged.
void ComponentPool::Apply(store::RestorePlan &plan, bool compare) {
    for (auto &c : plan.Changes) {
        const auto index = uint32_t(c.Slot);
        const auto previous = Stored(index);
        const auto entity = S.R.EntityAt(index);
        const bool present = previous != state::Null;
        if (c.Erase) {
            if (!present) {
                c.Unchanged = true;
                continue;
            }
            c.Old = Copy(index);
            c.WasPresent = true;
            S.R.remove(Type, previous);
            S.Changes.push_back({Type, previous, state::Event::Destroy});
            continue;
        }
        if (compare && present && previous == entity && c.MaybeEqual && store::Unchanged(Read(index), Encode(c.Incoming))) {
            c.Unchanged = true;
            continue;
        }
        if (present) {
            c.Old = Copy(index);
            c.WasPresent = true;
            if (previous != entity) S.R.remove(Type, previous);
        }
        if (entity != state::Null) {
            if (c.Incoming.Destroy) Encoding.Move(S.R, entity, c.Incoming);
            else {
                Encoding.Emplace(S.R, entity, c.Incoming.View());
                store::FreeBlob(c.Incoming);
            }
            S.Changes.push_back({Type, entity, previous == entity ? state::Event::Update : state::Event::Create});
        } else store::FreeBlob(c.Incoming);
        c.Incoming = {};
    }
}

bool ComponentPool::Restore(const store::Version &v) {
    Settle();
    auto plan = Trie.PlanRestore(v);
    Apply(plan, true);
    return Trie.CommitRestore(std::move(plan));
}

void ComponentPool::Load(uint64_t length, std::span<const std::pair<uint64_t, store::Hash128>> changes, const std::unordered_map<store::Hash128, std::vector<std::byte>, store::Hash128Hasher> &leaves) {
    Settle();
    auto plan = Trie.PlanLoad(length, changes, leaves);
    Apply(plan, false);
    Trie.CommitLoad(std::move(plan));
    // Move values whose entity changed identity to the entity now at their index.
    S.ForEachIdentityChange([&](uint32_t index) {
        if (Trie.IsDirty(index)) return; // Loaded slots already hold their values.
        const auto previous = Stored(index), entity = S.R.EntityAt(index);
        if (previous == state::Null || previous == entity) return;
        if (Trie.Uncaptured(index)) Trie.Capture(index, Copy(index));
        Trie.MarkDirty(index, 1);
        auto value = Copy(index);
        S.R.remove(Type, previous);
        if (entity != state::Null) {
            Encoding.Move(S.R, entity, value);
            S.Changes.push_back({Type, entity, state::Event::Create});
        } else store::FreeBlob(value);
    });
}
} // namespace project
