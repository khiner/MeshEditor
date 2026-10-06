#include "project/ComponentPool.h"
#include "Range.h"
#include "project/EntityStore.h"
#include "project/store/RecordPage.h"
#include "state/Scene.h"

#include <cassert>
#include <stdexcept>

namespace project {
namespace {
std::span<const std::byte> Serialize(const snapshot::SnapshotEntry &encoding, const void *value, std::vector<std::byte> &scratch) {
    if (encoding.How == snapshot::Encoding::Bytes) return {static_cast<const std::byte *>(value), encoding.Size};
    scratch.clear();
    if (encoding.How == snapshot::Encoding::Serialized) encoding.Serialize(value, scratch);
    return scratch;
}
} // namespace

ComponentPool::ComponentPool(EntityStore &s, state::TypeId type, const snapshot::SnapshotEntry &encoding)
    : S(s), Type(type), Encoding(encoding), Trie(4, 0, state::Table::PageCount) {
    static_assert(state::Table::PageCount == store::RecordsPerPage);
}

state::Table &ComponentPool::Storage() const { return S.R.Tables[Type]; }
state::Entity ComponentPool::Stored(uint32_t index) const { return Storage().entity_at(index); }
store::Blob ComponentPool::Copy(uint32_t page) { return Encoding.CopyPage(Storage(), page); }

uint64_t ComponentPool::Length() const { return S.Table.size(); }
bool ComponentPool::Present(uint64_t page) const { return Storage().mask(uint32_t(page)) != 0u; }
std::span<const std::byte> ComponentPool::Read(uint64_t page) {
    const auto &table = Storage();
    return store::EncodeRecordPage(table.mask(uint32_t(page)), PageScratch, [&](uint32_t slot) {
        return Serialize(Encoding, table.value(table.entity(uint32_t(page), slot)), Scratch);
    });
}
std::span<const std::byte> ComponentPool::Encode(const store::Blob &value) {
    if (!value.Destroy) return value.View();
    const auto mask = reinterpret_cast<const snapshot::NativePage *>(value.Data)->Mask;
    return store::EncodeRecordPage(mask, SnapshotScratch, [&](uint32_t slot) {
        return Serialize(Encoding, Encoding.PageValue(value, slot), Scratch);
    });
}

void ComponentPool::Capture(std::span<const state::PageMask> pages) {
    assert(!Trie.ExternalWritesForbidden && "component write during track restoration");
    ForEachIndexRun(pages, [&](size_t begin, size_t count) {
        const auto first = pages[begin].Page;
        Trie.Capture(first, count, this, [](void *owner, uint64_t index) -> std::optional<store::Blob> {
            auto &pool = *static_cast<ComponentPool *>(owner);
            return pool.Present(index) ? std::optional{pool.Copy(uint32_t(index))} : std::nullopt;
        });
        Trie.MarkDirty(first, count); }, &state::PageMask::Page);
}

void ComponentPool::Settle() {
    for (const auto index : Trie.Dirty()) Trie.Rehash(index, Present(index) ? std::optional{Read(index)} : std::nullopt);
    Trie.ClearDirty();
}

// Values move to the entity now at their index. A changed entity generation never counts as unchanged.
void ComponentPool::Apply(store::RestorePlan &plan, bool compare) {
    for (auto &c : plan.Changes) {
        const auto page = uint32_t(c.Slot);
        const auto old_mask = Storage().mask(page);
        const bool native = c.Incoming.Destroy != nullptr;
        auto bytes = native ? std::span<const std::byte>{} : c.Incoming.View();
        const auto mask = c.Erase ? 0u : native ? reinterpret_cast<const snapshot::NativePage *>(c.Incoming.Data)->Mask :
                                                  store::RecordMask(bytes);
        bool identities_match = true;
        for (auto bits = old_mask; bits; bits &= bits - 1u) {
            const auto index = page * state::Table::PageCount + uint32_t(std::countr_zero(bits));
            if (Stored(index) != S.R.EntityAt(index)) identities_match = false;
        }
        if ((c.Erase && !old_mask) || (compare && old_mask && identities_match && c.MaybeEqual && store::Unchanged(Read(page), Encode(c.Incoming)))) {
            c.Unchanged = true;
            continue;
        }
        if (old_mask) {
            c.Old = Copy(page);
            c.WasPresent = true;
        }
        for (auto bits = old_mask | mask; bits; bits &= bits - 1u) {
            const auto bit = 1u << std::countr_zero(bits);
            const auto index = page * state::Table::PageCount + uint32_t(std::countr_zero(bits));
            const auto previous = Stored(index), entity = S.R.EntityAt(index);
            if (!(mask & bit)) {
                S.R.remove(Type, previous);
                S.Changes.push_back({Type, previous, state::Event::Destroy});
                continue;
            }
            const auto slot = uint32_t(std::countr_zero(bits));
            const auto value = native ? std::span<const std::byte>{} : store::TakeRecord(bytes);
            if (compare && previous == entity && previous != state::Null && (native ? Encoding.Equal(Storage().value(previous), Encoding.PageValue(c.Incoming, slot)) : store::Unchanged(Serialize(Encoding, Storage().value(previous), Scratch), value))) continue;
            if (previous != state::Null && previous != entity) S.R.remove(Type, previous);
            if (entity != state::Null) {
                if (native) Encoding.MovePageValue(S.R, entity, c.Incoming, slot);
                else Encoding.Emplace(S.R, entity, value);
                S.Changes.push_back({Type, entity, previous == entity ? state::Event::Update : state::Event::Create});
            }
        }
        if (!bytes.empty()) throw std::invalid_argument("Trailing component page bytes.");
        store::FreeBlob(c.Incoming);
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
    // Only the table's occupied slots hold values, and loaded pages already hold theirs.
    S.ForEachIdentityChangeRun([&](uint32_t first, uint32_t last) {
        for (auto page = first / state::Table::PageCount; page * state::Table::PageCount < last; ++page) {
            if (Trie.IsDirty(page)) continue;
            for (auto bits = Storage().mask(page); bits; bits &= bits - 1u) {
                const auto index = page * state::Table::PageCount + uint32_t(std::countr_zero(bits));
                if (index < first || index >= last) continue;
                const auto previous = Stored(index), entity = S.R.EntityAt(index);
                if (previous == entity) continue;
                Encoding.Rebind(S.R, previous, entity);
                if (entity != state::Null) S.Changes.push_back({Type, entity, state::Event::Create});
            }
        }
    });
}
} // namespace project
