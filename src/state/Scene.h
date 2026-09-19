#pragma once

#include "state/Changes.h"
#include "state/Entity.h"
#include "state/Schema.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <functional>
#include <memory>
#include <ranges>
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

// Paged sparse set of raw values. Each page holds its slot owners and then the values.
// Small pages keep unrelated sparse components inexpensive and addresses stable.
struct Table {
    static constexpr uint32_t PageCount = 32;
    struct Page {
        uint32_t Mask{};
        std::array<Entity, PageCount> Owners;
    };
    uint32_t Size{}, Align{alignof(Page)}, ValueOffset{sizeof(Page)}, Count{};
    void (*Destroy)(void *){};
    std::vector<uint64_t> Occupied; // One bit per allocated page.
    std::vector<Page *> Pages;
    Table() = default;
    Table(const Table &) = delete;
    Table &operator=(const Table &) = delete;
    ~Table() { clear(); }
    // Describes the value type. An unbound table holds no value bytes.
    template<typename C> void Bind() {
        Size = sizeof(C);
        Align = uint32_t(std::max(alignof(C), alignof(Page)));
        ValueOffset = (uint32_t(sizeof(Page)) + Align - 1) / Align * Align;
        if constexpr (!std::is_trivially_destructible_v<C>) Destroy = [](void *p) { std::destroy_at(static_cast<C *>(p)); };
    }
    uint32_t pages() const { return uint32_t(Pages.size()); }
    uint32_t mask(uint32_t page) const { return page < Pages.size() && Pages[page] ? Pages[page]->Mask : 0; }
    uint64_t occupied(uint32_t word) const { return word < Occupied.size() ? Occupied[word] : 0; }
    Entity entity(uint32_t page, uint32_t slot) const { return Pages[page]->Owners[slot]; }
    Entity entity_at(uint32_t index) const {
        const auto p = index / PageCount;
        return p < Pages.size() && Pages[p] ? Pages[p]->Owners[index % PageCount] : Null;
    }
    bool contains(Entity e) const { return e != Null && entity_at(Index(e)) == e; }
    std::byte *slot(Entity e, size_t stride) const {
        const auto index = Index(e), p = index / PageCount, i = index % PageCount;
        auto *page = e != Null && p < Pages.size() ? Pages[p] : nullptr;
        return page && page->Owners[i] == e ? reinterpret_cast<std::byte *>(page) + ValueOffset + i * stride : nullptr;
    }
    void *value(Entity e) const { return slot(e, Size); }
    template<typename C> C *at(Entity e) const {
        auto *p = slot(e, sizeof(C));
        return p ? std::launder(reinterpret_cast<C *>(p)) : nullptr;
    }
    size_t size() const { return Count; }
    bool empty() const { return Count == 0; }
    // Links the slot and returns its uninitialized value storage.
    void *insert(Entity);
    // Destroys the value and unlinks the slot.
    void erase(Entity);
    void clear();

private:
    void Release(uint32_t page);
};

enum class On : uint8_t { Create = 1,
                          Update = 2,
                          Destroy = 4 };
constexpr On operator|(On a, On b) { return On(uint8_t(a) | uint8_t(b)); }

enum class Event : uint8_t { Create,
                             Update,
                             Destroy };
struct DirtySet {
    ~DirtySet();
    std::vector<Entity> Entities;
    auto begin() const { return Entities.rbegin(); }
    auto end() const { return Entities.rend(); }
    size_t size() const { return Entities.size(); }
    bool empty() const { return Entities.empty(); }
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

template<typename... C> struct ExcludeList {};
template<typename... C> inline constexpr ExcludeList<C...> Exclude{};
template<typename R, typename Ex, typename... C> struct View;

struct Scene {
    Scene();
    ~Scene();
    Scene(const Scene &) = delete;
    Scene &operator=(const Scene &) = delete;
    Services Context;
    std::array<Table, SchemaSize> Tables;
    std::vector<TypeId> Active; // Tables bound to a component type, in first-touch order.
    std::array<std::array<std::vector<DirtySet *>, 3>, SchemaSize> Dirty;
    std::array<DirtySet, size_t(Change::Count)> Changes;
    using Handler = void (*)(Scene &, Entity);
    std::array<std::array<std::vector<Handler>, 3>, SchemaSize> Handlers;
    // The generation table and ordered free list are the only allocation authority.
    std::unique_ptr<Allocation> AllocationStorage;
    Allocation &AllocationState() { return *AllocationStorage; }
    const Allocation &AllocationState() const { return *AllocationStorage; }
    Table Living;
    void (*Capture)(Scene &, TypeId, Entity){};
    void *HistoryOwner{};
    bool Restoring{}, DocumentReadOnly{}, RestoringEvents{};
    uint64_t Epoch{1};
    Entity create();
    void destroy(Entity);
    void RemoveComponents(Entity);
    bool valid(Entity e) const;
    uint32_t EntityCapacity() const;
    Entity EntityAt(uint32_t index) const;
    void ResetEntities();
    void RebuildLiving();
    // The non-const form binds the table to its component type on first use.
    template<typename C> Table &storage() {
        if constexpr (std::is_same_v<std::remove_const_t<C>, Entity>) return Living;
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
        if constexpr (std::is_same_v<std::remove_const_t<C>, Entity>) return Living;
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
    void clear(TypeId type) {
        auto &t = Tables[type];
        for (uint32_t p = 0; p < t.pages(); ++p)
            while (const auto m = t.mask(p)) remove(type, t.entity(p, std::countr_zero(m)));
    }
    template<typename... C> void clear() { (clear(Type<C>()), ...); }
    void ClearChanges() {
        for (auto &c : Changes) c.clear();
    }
    template<typename C, auto Fn> void connect(Event event) {
        Handlers[Type<C>()][size_t(event)].push_back([](Scene &r, Entity e) { std::invoke(Fn, r, e); });
    }
    template<typename C, auto Fn> void on_construct() { connect<C, Fn>(Event::Create); }
    template<typename C, auto Fn> void on_update() { connect<C, Fn>(Event::Update); }
    template<typename C, auto Fn> void on_destroy() { connect<C, Fn>(Event::Destroy); }
    template<typename... C, typename... X> auto view(ExcludeList<X...> = {}) {
        return View<Scene, ExcludeList<X...>, C...>{*this};
    }
    template<typename... C, typename... X> auto view(ExcludeList<X...> = {}) const {
        return View<const Scene, ExcludeList<X...>, const C...>{*this};
    }
};

inline DirtySet &reactive(Scene &r, Change c) { return r.Changes[size_t(c)]; }
inline const DirtySet &reactive(const Scene &r, Change c) { return r.Changes[size_t(c)]; }

// Entities holding every listed component and none of the excluded ones, joined page by page on the table masks.
template<typename R, typename... X, typename... C>
struct View<R, ExcludeList<X...>, C...> : std::ranges::view_interface<View<R, ExcludeList<X...>, C...>> {
    using Driver = std::tuple_element_t<0, std::tuple<C...>>;
    // Walks the live slots in entity index order. Slots removed ahead of the cursor are skipped.
    struct Iterator {
        using value_type = Entity;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        static constexpr uint32_t End = UINT32_MAX;
        View Source{};
        uint32_t Page{}, Bits{};
        // Move to the first page at or after Page with live slots.
        void Seek() {
            while (Page < Source.pages()) {
                const auto word = Source.occupied(Page / 64) & (~uint64_t{0} << Page % 64);
                if (!word) {
                    Page = (Page / 64 + 1) * 64;
                    continue;
                }
                Page = Page / 64 * 64 + std::countr_zero(word);
                if ((Bits = Source.mask(Page))) return;
                ++Page;
            }
            Page = End;
            Bits = 0;
        }
        Entity operator*() const { return Source.entity(Page, std::countr_zero(Bits)); }
        Iterator &operator++() {
            Bits = (Bits & (Bits - 1)) & Source.mask(Page);
            if (!Bits) {
                ++Page;
                Seek();
            }
            return *this;
        }
        Iterator operator++(int) {
            auto old = *this;
            ++*this;
            return old;
        }
        bool operator==(const Iterator &) const = default;
    };
    R *Owner{};
    View() = default;
    explicit View(R &r) : Owner(&r) {}
    bool operator==(const View &o) const { return Owner == o.Owner; }
    template<typename T> const Table &table() const { return std::as_const(*Owner).template storage<T>(); }
    uint32_t pages() const { return std::min({table<C>().pages()...}); }
    uint64_t occupied(uint32_t word) const { return (table<C>().occupied(word) & ...); }
    uint32_t mask(uint32_t page) const {
        auto m = (table<C>().mask(page) & ...);
        ((m &= ~table<X>().mask(page)), ...);
        return m;
    }
    Entity entity(uint32_t page, uint32_t slot) const { return table<Driver>().entity(page, slot); }
    bool contains(Entity e) const {
        return table<Driver>().contains(e) && (mask(Index(e) / Table::PageCount) >> (Index(e) % Table::PageCount) & 1);
    }
    Iterator begin() const {
        Iterator it{*this};
        it.Seek();
        return it;
    }
    Iterator end() const { return {*this, Iterator::End, 0}; }
    // A single component without exclusions answers from its table.
    static constexpr bool Direct = sizeof...(C) == 1 && sizeof...(X) == 0;
    bool empty() const {
        if constexpr (Direct) return table<Driver>().empty();
        else return begin() == end();
    }
    size_t size() const {
        if constexpr (Direct) return table<Driver>().size();
        else {
            size_t n = 0;
            for (auto it = begin(); it != end(); it.Page++, it.Seek()) n += std::popcount(it.Bits);
            return n;
        }
    }
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

namespace std::ranges {
template<typename R, typename Ex, typename... C> inline constexpr bool enable_borrowed_range<state::View<R, Ex, C...>> = true;
} // namespace std::ranges
