#include "state/Scene.h"
#include "state/Allocation.h"
#include <stdexcept>
namespace state {
namespace {
constexpr uint32_t Alive = 1u << 31;
}
Scene::Scene() : AllocationStorage(std::make_unique<Allocation>()) {}
Scene::~Scene() {
    Context.Clear();
    for (auto &p : Changes) p.reset();
}
DirtySet::~DirtySet() {
    if (Owner)
        for (auto [type, event] : Bindings) std::erase(Owner->Dirty[type][size_t(event)], this);
}
DirtySet &Scene::changes(TypeId id) {
    auto &set = Changes[id];
    if (!set) {
        set = std::make_unique<DirtySet>();
        set->bind(*this);
    }
    return *set;
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
Entity Scene::create(Entity requested) {
    if (DocumentReadOnly) throw std::logic_error("Entity creation during derived restoration");
    uint32_t i;
    if (requested != Null) {
        i = Index(requested);
        if (i == Index(Null) || Generation(requested) >= 0xfffu) throw std::logic_error("Reserved entity identity");
        if (EntityAt(i) != Null) throw std::logic_error("Creating an occupied entity slot");
        while (AllocationStorage->Generations.size() <= i) {
            const auto slot = uint32_t(AllocationStorage->Generations.size());
            AllocationStorage->Generations.PushBack(0);
            if (slot != i) AllocationStorage->Free.PushBack(slot);
        }
        for (size_t n = 0; n < AllocationStorage->Free.size(); ++n) {
            if (AllocationStorage->Free[n] != i) continue;
            for (size_t j = n + 1; j < AllocationStorage->Free.size(); ++j) AllocationStorage->Free.Set(j - 1, AllocationStorage->Free[j]);
            AllocationStorage->Free.PopBack();
            break;
        }
        AllocationStorage->Generations.Set(i, Alive | Generation(requested));
    } else {
        if (AllocationStorage->Free.empty()) {
            i = uint32_t(AllocationStorage->Generations.size());
            if (i >= Index(Null)) throw std::length_error("Entity indices exhausted");
            AllocationStorage->Generations.PushBack(0);
        } else {
            i = AllocationStorage->Free.Back();
            AllocationStorage->Free.PopBack();
        }
        AllocationStorage->Generations.Set(i, AllocationStorage->Generations[i] | Alive);
    }
    const auto e = MakeEntity(i, AllocationStorage->Generations[i] & ~Alive);
    Living.emplace(e);
    return e;
}
void Scene::destroy(Entity e) {
    if (DocumentReadOnly) throw std::logic_error("Entity destruction during derived restoration");
    assert(valid(e));
    RemoveComponents(e);
    Living.remove(e);
    const auto next = Generation(e) + 1;
    AllocationStorage->Generations.Set(Index(e), next);
    if (next < 0xfffu) AllocationStorage->Free.PushBack(Index(e));
}
void Scene::RemoveComponents(Entity e) {
    for (auto &p : Tables)
        if (p) p->remove(e);
    for (auto &set : Changes)
        if (set) set->remove(e);
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
        if (const auto e = EntityAt(i); e != Null) Living.emplace(e);
}
} // namespace state
