#pragma once

#include "state/Entity.h"
#include "state/Schema.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace state {
struct Allocation;
// Fixed service slots are selected at compile time. Definitions stay in the owning TU.
struct Services {
    struct Slot {
        void *Value{};
        void (*Destroy)(void *){};
        ~Slot() { Reset(); }
        void Reset() {
            if (Value) Destroy(std::exchange(Value, nullptr));
        }
    };
    std::array<Slot, SchemaSize> Slots;
    template<typename T> T *find() const { return static_cast<T *>(Slots[Type<T>()].Value); }
    template<typename T> T &get() const {
        auto *p = find<T>();
        assert(p);
        return *p;
    }
    template<typename T, typename... A> T &emplace(A &&...args) {
        auto &s = Slots[Type<T>()];
        if (!s.Value) {
            s.Value = new T(std::forward<A>(args)...);
            s.Destroy = [](void *p) { delete static_cast<T *>(p); };
        }
        return *static_cast<T *>(s.Value);
    }
    template<typename T> void erase() { Slots[Type<T>()].Reset(); }
    void Clear() {
        for (auto &slot : Slots) slot.Reset();
    }
};

struct EntityRange {
    std::vector<Entity> Entities;
    auto begin() const { return Entities.rbegin(); }
    auto end() const { return Entities.rend(); }
    size_t size() const { return Entities.size(); }
    bool empty() const { return Entities.empty(); }
};

struct TableBase : EntityRange {
    virtual ~TableBase() = default;
    virtual bool contains(Entity) const = 0;
    virtual void *value(Entity) const = 0;
    virtual Entity entity_at(uint32_t) const = 0;
    virtual bool remove(Entity) = 0;
    virtual void clear() {
        while (!empty()) remove(Entities.back());
    }
};

enum class On : uint8_t { Create = 1,
                          Update = 2,
                          Destroy = 4 };
constexpr On operator|(On a, On b) { return On(uint8_t(a) | uint8_t(b)); }

enum class Event : uint8_t { Create,
                             Update,
                             Destroy };
struct DirtySet : EntityRange {
    ~DirtySet();
    std::vector<std::pair<TypeId, Event>> Bindings;
    void Track(TypeId, Event);
    std::vector<uint32_t> Positions;
    Scene *Owner{};
    bool contains(Entity e) const {
        const auto i = Index(e);
        return i < Positions.size() && Positions[i] < Entities.size() && Entities[Positions[i]] == e;
    }
    Entity entity_at(uint32_t i) const { return i < Positions.size() && Positions[i] < Entities.size() ? Entities[Positions[i]] : Null; }
    void emplace(Entity e) {
        const auto i = Index(e);
        if (i >= Positions.size()) Positions.resize(i + 1, UINT32_MAX);
        const auto pos = Positions[i];
        if (pos < Entities.size()) {
            Entities[pos] = e;
            return;
        }
        Positions[i] = uint32_t(Entities.size());
        Entities.push_back(e);
    }
    bool remove(Entity e) {
        if (!contains(e)) return false;
        const auto pos = Positions[Index(e)];
        Entities[pos] = Entities.back();
        Positions[Index(Entities[pos])] = pos;
        Positions[Index(e)] = UINT32_MAX;
        Entities.pop_back();
        return true;
    }
    void clear() {
        for (auto e : Entities) Positions[Index(e)] = UINT32_MAX;
        Entities.clear();
    }
    void bind(Scene &r) { Owner = &r; }
    template<typename C> DirtySet &on(On events) {
        for (auto event : {Event::Create, Event::Update, Event::Destroy})
            if (uint8_t(events) & (1u << uint8_t(event))) Track(Type<C>(), event);
        return *this;
    }
};

void Notify(Scene &, TypeId, Event, Entity);
void BeforeWrite(Scene &, TypeId, Entity);

template<typename C> struct Table final : TableBase {
    // Small pages keep unrelated sparse components inexpensive and addresses stable.
    static constexpr uint32_t PageCount = 32;
    struct Page {
        uint32_t Mask{};
        std::array<uint32_t, PageCount> Positions{};
        alignas(C) std::byte Bytes[PageCount * sizeof(C)];
        C *address(uint32_t i) { return reinterpret_cast<C *>(Bytes + i * sizeof(C)); }
        C *at(uint32_t i) { return std::launder(address(i)); }
        ~Page() {
            for (uint32_t i = 0; i < PageCount; ++i)
                if (Mask & (1u << i)) std::destroy_at(at(i));
        }
    };
    Scene &Owner;
    std::vector<std::unique_ptr<Page>> Pages;
    explicit Table(Scene &owner) : Owner(owner) {}
    Page *page(Entity e) const {
        const auto p = Index(e) / PageCount;
        return p < Pages.size() ? Pages[p].get() : nullptr;
    }
    bool contains(Entity e) const override {
        return e != Null && entity_at(Index(e)) == e;
    }
    Entity entity_at(uint32_t index) const override {
        const auto e = Entity{index};
        auto *p = page(e);
        const auto i = index % PageCount;
        return p && (p->Mask & (1u << i)) ? Entities[p->Positions[i]] : Null;
    }
    void *value(Entity e) const override { return contains(e) ? page(e)->at(Index(e) % PageCount) : nullptr; }
    C &get(Entity e) const {
        auto *p = static_cast<C *>(value(e));
        assert(p);
        return *p;
    }
    template<typename... A> C &emplace(Entity e, A &&...args) {
        assert(!contains(e));
        BeforeWrite(Owner, Type<C>(), e);
        const auto p = Index(e) / PageCount, i = Index(e) % PageCount;
        if (p >= Pages.size()) Pages.resize(p + 1);
        if (!Pages[p]) Pages[p] = std::make_unique<Page>();
        auto &page = *Pages[p];
        assert(!(page.Mask & (1u << i)));
        if constexpr (std::is_aggregate_v<C> && !std::is_constructible_v<C, A...>) ::new (static_cast<void *>(page.address(i))) C{std::forward<A>(args)...};
        else std::construct_at(page.address(i), std::forward<A>(args)...);
        page.Mask |= 1u << i;
        page.Positions[i] = uint32_t(Entities.size());
        Entities.push_back(e);
        Notify(Owner, Type<C>(), Event::Create, e);
        return get(e);
    }
    bool remove(Entity e) override {
        if (!contains(e)) return false;
        BeforeWrite(Owner, Type<C>(), e);
        Notify(Owner, Type<C>(), Event::Destroy, e);
        auto &p = *page(e);
        const auto i = Index(e) % PageCount, pos = p.Positions[i];
        std::destroy_at(p.at(i));
        p.Mask &= ~(1u << i);
        const auto moved = Entities.back();
        Entities[pos] = moved;
        page(moved)->Positions[Index(moved) % PageCount] = pos;
        Entities.pop_back();
        if (!p.Mask) Pages[Index(e) / PageCount].reset();
        return true;
    }
};

template<typename... C> struct ExcludeList {};
template<typename... C> inline constexpr ExcludeList<C...> Exclude{};
template<typename R, typename... C> struct View;

struct Scene {
    Scene();
    ~Scene();
    Scene(const Scene &) = delete;
    Scene &operator=(const Scene &) = delete;
    Services Context;
    std::array<std::unique_ptr<TableBase>, SchemaSize> Tables;
    std::array<std::array<std::vector<DirtySet *>, 3>, SchemaSize> Dirty;
    std::array<std::unique_ptr<DirtySet>, SchemaSize> Changes;
    struct Handler {
        void (*Apply)(void *, Scene &, Entity);
        void *Owner{};
        void operator()(Scene &r, Entity e) const { Apply(Owner, r, e); }
    };
    std::array<std::array<std::vector<Handler>, 3>, SchemaSize> Handlers;
    // The generation table and ordered free list are the only allocation authority.
    std::unique_ptr<Allocation> AllocationStorage;
    Allocation &AllocationState() { return *AllocationStorage; }
    const Allocation &AllocationState() const { return *AllocationStorage; }
    DirtySet Living;
    void (*Capture)(Scene &, TypeId, Entity){};
    void *HistoryOwner{};
    bool Restoring{}, DocumentReadOnly{}, RestoringEvents{};
    uint64_t Epoch{1};
    Services &ctx() { return Context; }
    const Services &ctx() const { return Context; }
    Entity create();
    void destroy(Entity);
    void RemoveComponents(Entity);
    bool valid(Entity e) const;
    uint32_t EntityCapacity() const;
    Entity EntityAt(uint32_t index) const;
    void ResetEntities();
    void RebuildLiving();
    TableBase *storage(TypeId id) { return Tables[id].get(); }
    const TableBase *storage(TypeId id) const { return Tables[id].get(); }
    DirtySet &changes(TypeId);
    template<typename C> auto &storage() {
        if constexpr (std::is_same_v<C, Entity>) return Living;
        else {
            using T = std::remove_const_t<C>;
            auto &p = Tables[Type<T>()];
            if (!p) p = std::make_unique<Table<T>>(*this);
            return *static_cast<Table<T> *>(p.get());
        }
    }
    auto storage() const {
        return std::views::iota(size_t{0}, SchemaSize) | std::views::filter([this](size_t i) { return bool(Tables[i]); }) |
            std::views::transform([this](size_t i) { return std::pair<TypeId, const TableBase &>{TypeId(i), *Tables[i]}; });
    }
    template<typename C> const C *try_get(Entity e) const {
        const auto *p = static_cast<const Table<std::remove_const_t<C>> *>(Tables[Type<C>()].get());
        return p ? static_cast<const C *>(p->value(e)) : nullptr;
    }
    template<typename C> C &edit(Entity e) {
        BeforeWrite(*this, Type<C>(), e);
        return storage<C>().get(e);
    }
    template<typename C> C *try_edit(Entity e) { return all_of<C>(e) ? &edit<C>(e) : nullptr; }
    template<typename C> const C &get(Entity e) const {
        auto *p = try_get<C>(e);
        assert(p);
        return *p;
    }
    template<typename... C> bool all_of(Entity e) const { return (... && (try_get<C>(e) != nullptr)); }
    template<typename... C> bool any_of(Entity e) const { return (... || (try_get<C>(e) != nullptr)); }
    template<typename C, typename... A> decltype(auto) emplace(Entity e, A &&...a) { return storage<C>().emplace(e, std::forward<A>(a)...); }
    template<typename C, typename... A> C &replace(Entity e, A &&...a) {
        return patch<C>(e, [&](C &value) { value = C{std::forward<A>(a)...}; });
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
    template<typename... C> size_t remove(Entity e) {
        return (size_t{0} + ... + (Tables[Type<C>()] ? Tables[Type<C>()]->remove(e) : false));
    }
    template<typename... C> void clear() { ((Tables[Type<C>()] ? Tables[Type<C>()]->clear() : void()), ...); }
    void ClearChanges() {
        for (auto &p : Changes)
            if (p) p->clear();
    }
    struct Sink {
        Scene &R;
        TypeId Type;
        Event Kind;
        template<auto Fn> void connect() {
            R.Handlers[Type][size_t(Kind)].push_back({[](void *, Scene &r, Entity e) { std::invoke(Fn, r, e); }});
        }
        template<auto Fn, typename T> void connect(T &owner) {
            R.Handlers[Type][size_t(Kind)].push_back({[](void *p, Scene &r, Entity e) { std::invoke(Fn, *static_cast<T *>(p), r, e); }, &owner});
        }
    };
    template<typename C> Sink on_construct() { return {*this, Type<C>(), Event::Create}; }
    template<typename C> Sink on_update() { return {*this, Type<C>(), Event::Update}; }
    template<typename C> Sink on_destroy() { return {*this, Type<C>(), Event::Destroy}; }
    template<typename... C, typename... X> auto view(ExcludeList<X...> = {}) {
        static constexpr std::array<TypeId, sizeof...(X)> excluded{Type<X>()...};
        (storage<std::remove_const_t<C>>(), ...);
        return View<Scene, C...>{*this, excluded};
    }
    template<typename... C, typename... X> auto view(ExcludeList<X...> = {}) const {
        static constexpr std::array<TypeId, sizeof...(X)> excluded{Type<X>()...};
        return View<const Scene, const C...>{*this, excluded};
    }
};

template<typename R, typename... C> struct View : std::ranges::view_interface<View<R, C...>> {
    R *Owner{};
    std::span<const TypeId> Excluded;
    const EntityRange *Driver{};
    View(R &r, std::span<const TypeId> excluded) : Owner(&r), Excluded(excluded) {
        if constexpr ((std::is_same_v<std::remove_const_t<C>, Entity> && ...)) Driver = &r.Living;
        else {
            for (auto id : {Type<C>()...}) {
                auto *p = r.storage(id);
                if (!p) {
                    Driver = nullptr;
                    break;
                }
                if (!Driver || p->size() < Driver->size()) Driver = p;
            }
        }
    }
    bool contains(Entity e) const {
        const bool present = ([&] {
            if constexpr (std::is_same_v<std::remove_const_t<C>, Entity>) return Owner->valid(e);
            else return Owner->template all_of<C>(e);
        }() && ...);
        if (!present) return false;
        for (auto id : Excluded) {
            auto *p = Owner->storage(id);
            if (p && p->contains(e)) return false;
        }
        return true;
    }
    struct Iterator {
        using value_type = Entity;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        const View *V{};
        size_t Position{};
        void Skip() {
            while (Position && (Position > V->Driver->size() || !V->contains(V->Driver->Entities[Position - 1]))) --Position;
        }
        Entity operator*() const { return V->Driver->Entities[Position - 1]; }
        Iterator &operator++() {
            --Position;
            Skip();
            return *this;
        }
        Iterator operator++(int) {
            auto old = *this;
            ++*this;
            return old;
        }
        bool operator==(const Iterator &) const = default;
    };
    Iterator begin() const {
        Iterator i{this, Driver ? Driver->size() : 0};
        i.Skip();
        return i;
    }
    Iterator end() const { return {this, 0}; }
    // A single component without exclusions answers from its table.
    static constexpr bool Direct = sizeof...(C) == 1;
    bool empty() const {
        if constexpr (Direct)
            if (Excluded.empty()) return !Driver || Driver->empty();
        return begin() == end();
    }
    size_t size() const {
        if constexpr (Direct)
            if (Excluded.empty()) return Driver ? Driver->size() : 0;
        size_t n = 0;
        for ([[maybe_unused]] auto e : *this) ++n;
        return n;
    }
    Entity front() const { return *begin(); }
    template<typename T> decltype(auto) get(Entity e) const { return Owner->template get<T>(e); }
    template<typename T> static auto Value(R &owner, Entity e) {
        if constexpr (std::is_empty_v<T> || std::is_same_v<std::remove_const_t<T>, Entity>) return std::tuple<>{};
        else if constexpr (std::is_const_v<T> || std::is_const_v<R>) return std::tie(owner.template get<T>(e));
        else return std::tie(owner.template edit<T>(e));
    }
    auto each() const {
        return View{*this} | std::views::transform([owner = Owner](Entity e) { return std::tuple_cat(std::tuple{e}, Value<C>(*owner, e)...); });
    }
};

} // namespace state
