#pragma once

#include "state/DirtySet.h"
#include "state/Resources.h"
#include "state/View.h"

namespace state {
struct Allocation;

void Notify(Registry &, TypeId, Event, Entity);
void BeforeWrite(Registry &, TypeId, Entity);

struct Registry {
    Registry();
    ~Registry();
    Registry(const Registry &) = delete;
    Registry &operator=(const Registry &) = delete;
    Resources Resources;
    std::array<Table, SchemaSize> Tables;
    std::vector<TypeId> Active; // Tables bound to a component type, in first-touch order.
    std::array<std::array<std::vector<DirtySet *>, 3>, SchemaSize> Dirty;
    std::vector<DirtySet *> Bound; // Every set bound to the registry, in bind order.
    using Handler = void (*)(Registry &, Entity);
    std::array<std::array<std::vector<Handler>, 3>, SchemaSize> Handlers;
    // The generation table and ordered free list are the only allocation authority.
    std::unique_ptr<Allocation> AllocationStorage;
    Allocation &AllocationState() { return *AllocationStorage; } // NOLINT(readability-make-member-function-const)
    const Allocation &AllocationState() const { return *AllocationStorage; }
    Table Living;
    void (*Capture)(Registry &, TypeId, std::span<const PageMask>){};
    void *HistoryOwner{};
    bool Restoring{}, DocumentReadOnly{}, RestoringEvents{};
    uint64_t Epoch{1};
    Entity create();
    void destroy(Entity);
    void destroy(std::span<const Entity>);
    void RemoveComponents(Entity);
    void RemoveComponents(std::span<const Entity>);
    bool valid(Entity e) const;
    uint32_t EntityCapacity() const;
    Entity EntityAt(uint32_t index) const;
    // Returns the registry to its empty state: every component, entity, change set, and scene-state resource.
    // Components drop without destroy handlers, since each resource resets whole.
    // Entity allocation restarts from zero so a replay reproduces the same identities.
    void Reset();
    // The non-const form binds the table to its component type, recording the table as active on first use.
    template<typename C> Table &storage() {
        if constexpr (std::same_as<std::remove_const_t<C>, Entity>) return Living;
        else {
            auto &t = Tables[Type<C>()];
            if (!t.Size) {
                t.template Bind<std::remove_const_t<C>>();
                Active.push_back(Type<C>());
            }
            return t;
        }
    }
    template<typename C> const Table &storage() const {
        if constexpr (std::same_as<std::remove_const_t<C>, Entity>) return Living;
        else return Tables[Type<C>()];
    }
    template<typename C> const C *try_get(Entity e) const { return Tables[Type<C>()].template at<const C>(e); }
    template<typename C> C &edit(Entity e) {
        BeforeWrite(*this, Type<C>(), e);
        auto *p = Tables[Type<C>()].template at<C>(e);
        assert(p);
        return *p;
    }
    template<typename C> C *try_edit(Entity e) { return all_of<C>(e) ? &edit<C>(e) : nullptr; }
    template<typename C> const C &get(Entity e) const {
        auto *p = try_get<C>(e);
        assert(p);
        return *p;
    }
    template<typename... C> bool all_of(Entity e) const { return (... && (try_get<C>(e) != nullptr)); }
    template<typename... C> bool any_of(Entity e) const { return (... || (try_get<C>(e) != nullptr)); }
    template<typename C, typename... A> C &emplace(Entity e, A &&...a) {
        auto &t = storage<C>();
        assert(!t.contains(e));
        BeforeWrite(*this, Type<C>(), e);
        auto *slot = t.insert(e);
        C *value;
        if constexpr (std::is_aggregate_v<C> && !std::is_constructible_v<C, A...>) value = ::new (slot) C{std::forward<A>(a)...};
        else value = std::construct_at(static_cast<C *>(slot), std::forward<A>(a)...);
        Notify(*this, Type<C>(), Event::Create, e);
        return *value;
    }
    template<typename C, typename... A> C &replace(Entity e, A &&...a) {
        return patch<C>(e, [&](C &value) { value = C{std::forward<A>(a)...}; });
    }
    // Replaces the value when it differs and returns whether it did.
    template<typename C> bool replace_if_changed(Entity e, const C &value) {
        if (get<C>(e) == value) return false;
        replace<C>(e, value);
        return true;
    }
    template<typename C, typename... A> C &emplace_or_replace(Entity e, A &&...a) {
        return all_of<C>(e) ? replace<C>(e, std::forward<A>(a)...) : emplace<C>(e, std::forward<A>(a)...);
    }
    template<typename C, typename... A> C &get_or_emplace(Entity e, A &&...a) {
        return all_of<C>(e) ? edit<C>(e) : emplace<C>(e, std::forward<A>(a)...);
    }
    template<typename C, typename... F> C &patch(Entity e, F &&...fn) {
        auto &value = edit<C>(e);
        (fn(value), ...);
        Notify(*this, Type<C>(), Event::Update, e);
        return value;
    }
    bool remove(TypeId type, Entity e) {
        auto &t = Tables[type];
        if (!t.contains(e)) return false;
        BeforeWrite(*this, type, e);
        Notify(*this, type, Event::Destroy, e);
        t.erase(e);
        return true;
    }
    template<typename... C> size_t remove(Entity e) { return (size_t{0} + ... + remove(Type<C>(), e)); }
    // Captures the type's occupied pages once and notifies each removed value before its page is unlinked.
    void clear(TypeId);
    template<typename... C> void clear() { (clear(Type<C>()), ...); }
    // Clears every managed set.
    void ClearChanges() {
        for (auto *set : Bound) {
            if (set->Clears == Clearing::Managed) set->clear();
        }
    }
    template<typename C, auto Fn> void connect(Event event) {
        Handlers[Type<C>()][size_t(event)].push_back([](Registry &r, Entity e) { std::invoke(Fn, r, e); });
    }
    template<typename C, auto Fn> void on_construct() { connect<C, Fn>(Event::Create); }
    template<typename C, auto Fn> void on_update() { connect<C, Fn>(Event::Update); }
    template<typename C, auto Fn> void on_destroy() { connect<C, Fn>(Event::Destroy); }
    template<typename... C, typename... X> auto view(ExcludeList<X...> = {}) {
        return View<Registry, ExcludeList<X...>, C...>{*this};
    }
    template<typename... C, typename... X> auto view(ExcludeList<X...> = {}) const {
        return View<const Registry, ExcludeList<X...>, const C...>{*this};
    }
};
} // namespace state
