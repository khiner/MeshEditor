#include "state/Scene.h"
#include "state/Allocation.h"
#include "Range.h"
#include <new>
#include <stdexcept>
namespace state {
namespace {
constexpr uint32_t Alive = 1u << 31;
std::byte *Bytes(Table::Page &page) { return reinterpret_cast<std::byte *>(&page); }

std::vector<PageMask> RemovalPages(const Scene &r, std::span<const Entity> requested) {
    std::vector<Entity> entities{requested.begin(), requested.end()};
    if (!std::ranges::is_sorted(entities, {}, Index)) std::ranges::sort(entities, {}, Index);
    std::vector<PageMask> pages;
    for (const auto e : entities) {
        if (!r.Living.contains(e)) continue;
        const auto page = Index(e) / Table::PageCount;
        if (pages.empty() || pages.back().Page != page) pages.push_back({page, 0u});
        pages.back().Mask |= 1u << (Index(e) % Table::PageCount);
    }
    return pages;
}

void RemoveComponentPages(Scene &r, std::span<const PageMask> pages) {
    // Capture each type's affected pages once before callbacks mutate its values.
    // Callbacks retain access to the component being destroyed until its page is unlinked.
    for (size_t i = 0; i < r.Active.size(); ++i) {
        const auto type = r.Active[i];
        auto &table = r.Tables[type];
        if (table.empty()) continue;
        std::vector<PageMask> affected;
        for (const auto [page, mask] : pages) {
            auto removed = table.mask(page) & mask;
            for (auto bits = removed; bits; bits &= bits - 1u) {
                const auto slot = uint32_t(std::countr_zero(bits));
                if (table.entity(page, slot) != r.Living.entity(page, slot)) removed &= ~(1u << slot);
            }
            if (removed) affected.push_back({page, removed});
        }
        if (r.Capture && !affected.empty()) r.Capture(r, type, affected);
        for (const auto [page, mask] : affected) {
            const auto removed = table.mask(page) & mask;
            if (!removed) continue;
            for (auto bits = removed; bits; bits &= bits - 1u)
                Notify(r, type, Event::Destroy, table.entity(page, std::countr_zero(bits)));
            table.erase(page, removed);
        }
    }
    // Change sets are small or empty after actions; inspect their members once.
    for (auto &set : r.Changes) {
        for (size_t i = 0; i < set.Entities.size();) {
            const auto e = set.Entities[i];
            const auto page = Index(e) / Table::PageCount;
            const auto found = std::ranges::lower_bound(pages, page, {}, &PageMask::Page);
            if (found != pages.end() && found->Page == page &&
                (found->Mask & (1u << (Index(e) % Table::PageCount))) && r.Living.contains(e)) set.remove(e);
            else ++i;
        }
    }
}
} // namespace
void *Table::insert(Entity e) {
    const auto index = Index(e), p = index / PageCount, i = index % PageCount;
    if (p >= Pages.size()) {
        Pages.resize(p + 1);
        Occupied.resize(p / 64 + 1);
    }
    if (!Pages[p]) {
        Pages[p] = new (::operator new(ValueOffset + PageCount * Size, std::align_val_t(Align))) Page{};
        Pages[p]->Owners.fill(Null);
        Occupied[p / 64] |= uint64_t{1} << p % 64;
    }
    auto &page = *Pages[p];
    assert(!(page.Mask & (1u << i)));
    page.Mask |= 1u << i;
    page.Owners[i] = e;
    ++Count;
    return Bytes(page) + ValueOffset + i * Size;
}
void Table::erase(Entity e) {
    assert(contains(e));
    erase(Index(e) / PageCount, 1u << (Index(e) % PageCount));
}
void Table::erase(uint32_t p, uint32_t mask) {
    auto &page = *Pages[p];
    mask &= page.Mask;
    Count -= uint32_t(std::popcount(mask));
    if (mask == page.Mask) {
        Release(p);
        return;
    }
    for (auto bits = mask; bits; bits &= bits - 1u) {
        const auto i = uint32_t(std::countr_zero(bits));
        if (Destroy) Destroy(Bytes(page) + ValueOffset + i * Size);
        page.Owners[i] = Null;
    }
    page.Mask &= ~mask;
}
void Table::Release(uint32_t p) {
    auto &page = *Pages[p];
    if (Destroy)
        for (uint32_t i = 0; i < PageCount; ++i)
            if (page.Mask & (1u << i)) Destroy(Bytes(page) + ValueOffset + i * Size);
    ::operator delete(&page, std::align_val_t(Align));
    Pages[p] = nullptr;
    Occupied[p / 64] &= ~(uint64_t{1} << p % 64);
}
void Table::clear() {
    for (uint32_t p = 0; p < Pages.size(); ++p)
        if (Pages[p]) Release(p);
    Pages.clear();
    Occupied.clear();
    Count = 0;
}
Scene::Scene() : AllocationStorage(std::make_unique<Allocation>()) {
    for (auto &c : Changes) c.bind(*this);
}
Scene::~Scene() { Context.Clear(); }
DirtySet::~DirtySet() {
    if (Owner)
        for (auto [type, event] : Bindings) std::erase(Owner->Dirty[type][size_t(event)], this);
}
void DirtySet::Track(TypeId type, Event event) {
    assert(Owner);
    Owner->Dirty[type][size_t(event)].push_back(this);
    Bindings.emplace_back(type, event);
}
void BeforeWrite(Scene &r, TypeId type, Entity e) {
    if (r.Capture) {
        const PageMask page{Index(e) / Table::PageCount, 1u << (Index(e) % Table::PageCount)};
        r.Capture(r, type, std::span{&page, 1u});
    }
}
void Notify(Scene &r, TypeId type, Event event, Entity e) {
    if (r.Restoring && !r.RestoringEvents && event != Event::Destroy) return;
    for (auto *dirty : r.Dirty[type][size_t(event)]) dirty->emplace(e);
    const auto &handlers = r.Handlers[type][size_t(event)];
    for (auto it = handlers.rbegin(); it != handlers.rend(); ++it) (*it)(r, e);
}
bool Scene::valid(Entity e) const { return e != Null && Index(e) < AllocationStorage->Generations.size() && AllocationStorage->Generations[Index(e)] == (Alive | Generation(e)); }
uint32_t Scene::EntityCapacity() const { return uint32_t(AllocationStorage->Generations.size()); }
Entity Scene::EntityAt(uint32_t index) const {
    if (index >= EntityCapacity()) return Null;
    const auto value = AllocationStorage->Generations[index];
    return value & Alive ? MakeEntity(index, value & ~Alive) : Null;
}
Entity Scene::create() {
    if (DocumentReadOnly) throw std::logic_error("Entity creation during derived restoration");
    uint32_t i;
    if (AllocationStorage->Free.empty()) {
        i = uint32_t(AllocationStorage->Generations.size());
        if (i >= Index(Null)) throw std::length_error("Entity indices exhausted");
        AllocationStorage->Generations.PushBack(0);
    } else {
        i = AllocationStorage->Free.Back();
        AllocationStorage->Free.PopBack();
    }
    AllocationStorage->Generations.Set(i, AllocationStorage->Generations[i] | Alive);
    const auto e = MakeEntity(i, AllocationStorage->Generations[i] & ~Alive);
    Living.insert(e);
    return e;
}
void Scene::destroy(Entity e) {
    destroy(std::span{&e, 1u});
}
void Scene::destroy(std::span<const Entity> entities) {
    if (DocumentReadOnly) throw std::logic_error("Entity destruction during derived restoration");
    for (const auto e : entities) assert(valid(e));
    const auto pages = RemovalPages(*this, entities);
    RemoveComponentPages(*this, pages);
    std::vector<uint32_t> freed;
    auto &generations = AllocationStorage->Generations;
    // Capture adjacent generation pages once, including partial component pages.
    ForEachIndexRun(pages, [&](size_t first, size_t count) {
        const auto start = pages[first].Page * Table::PageCount;
        const auto end = std::min<uint64_t>(generations.size(), (pages[first].Page + count) * Table::PageCount);
        auto values = std::span{reinterpret_cast<uint32_t *>(generations.P.Mutable(uint64_t(start) * sizeof(uint32_t), (end - start) * sizeof(uint32_t)).data()), size_t(end - start)};
        for (size_t j = first; j < first + count; ++j) {
            for (auto bits = pages[j].Mask; bits; bits &= bits - 1u) {
                const auto index = pages[j].Page * Table::PageCount + uint32_t(std::countr_zero(bits));
                auto &value = values[index - start];
                value = (value & ~Alive) + 1u;
                if (value < 0xfffu) freed.push_back(index);
            }
            Living.erase(pages[j].Page, pages[j].Mask);
        }
    }, &PageMask::Page);
    auto &free = AllocationStorage->Free;
    const auto before = free.P.Length();
    free.P.Resize(before + freed.size() * sizeof(uint32_t));
    if (!freed.empty()) std::memcpy(free.P.Storage.data() + before, freed.data(), freed.size() * sizeof(uint32_t));
}
void Scene::RemoveComponents(Entity e) {
    RemoveComponents(std::span{&e, 1u});
}
void Scene::RemoveComponents(std::span<const Entity> entities) {
    RemoveComponentPages(*this, RemovalPages(*this, entities));
}
void Scene::clear(TypeId type) {
    auto &table = Tables[type];
    if (table.empty()) return;
    std::vector<PageMask> pages;
    for (uint32_t word = 0; word < table.Occupied.size(); ++word) {
        for (auto bits = table.Occupied[word]; bits; bits &= bits - 1u) {
            const auto page = word * 64u + uint32_t(std::countr_zero(bits));
            pages.push_back({page, table.mask(page)});
        }
    }
    if (Capture) Capture(*this, type, pages);
    for (const auto [page, _] : pages) {
        const auto mask = table.mask(page);
        if (!mask) continue;
        for (auto bits = mask; bits; bits &= bits - 1u) Notify(*this, type, Event::Destroy, table.entity(page, uint32_t(std::countr_zero(bits))));
        table.erase(page, mask);
    }
}
void Scene::ResetEntities() {
    ++Epoch;
    for (auto &type : Dirty)
        for (auto &event : type)
            for (auto *set : event) set->clear();
    AllocationStorage->Generations.Clear();
    AllocationStorage->Free.Clear();
    Living.clear();
}
} // namespace state
