#include "state/Scene.h"
#include "state/Allocation.h"
#include <new>
#include <stdexcept>
namespace state {
namespace {
constexpr uint32_t Alive = 1u << 31;
std::byte *Bytes(Table::Page &page) { return reinterpret_cast<std::byte *>(&page); }
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
    const auto index = Index(e), p = index / PageCount, i = index % PageCount;
    auto &page = *Pages[p];
    if (Destroy) Destroy(Bytes(page) + ValueOffset + i * Size);
    page.Mask &= ~(1u << i);
    page.Owners[i] = Null;
    --Count;
    if (!page.Mask) Release(p);
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
    if (r.Capture) r.Capture(r, type, e);
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
    if (DocumentReadOnly) throw std::logic_error("Entity destruction during derived restoration");
    assert(valid(e));
    RemoveComponents(e);
    Living.erase(e);
    const auto next = Generation(e) + 1;
    AllocationStorage->Generations.Set(Index(e), next);
    if (next < 0xfffu) AllocationStorage->Free.PushBack(Index(e));
}
void Scene::RemoveComponents(Entity e) {
    for (const auto type : Active) remove(type, e);
    for (auto &set : Changes) set.remove(e);
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
void Scene::RebuildLiving() {
    Living.clear();
    for (uint32_t i = 0; i < AllocationStorage->Generations.size(); ++i)
        if (const auto e = EntityAt(i); e != Null) Living.insert(e);
}
} // namespace state
