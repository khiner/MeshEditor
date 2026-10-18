#pragma once

#include "state/Entity.h"
#include "state/Slot.h"

#include <vector>

namespace state {
struct Registry;

enum class On : uint8_t { Create = 1,
                          Update = 2,
                          Destroy = 4 };
constexpr On operator|(On a, On b) { return On(uint8_t(a) | uint8_t(b)); }

enum class Event : uint8_t { Create,
                             Update,
                             Destroy };
// A managed set empties whenever the registry clears its changes, and drops each destroyed entity.
// An explicit set keeps its entities, destroyed ones included, until its owner clears it.
enum class Clearing : uint8_t { Managed,
                                Explicit };
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
    // One byte per entity index, toggled by each Create and Destroy event the set receives.
    std::vector<uint8_t> Parity;
    Registry *Owner{};
    Clearing Clears{};
    bool contains(Entity e) const {
        const auto i = Index(e);
        return i < Positions.size() && Positions[i] < Entities.size() && Entities[Positions[i]] == e;
    }
    Entity entity_at(uint32_t i) const { return i < Positions.size() && Positions[i] < Entities.size() ? Entities[Positions[i]] : Null; }
    // Whether a tracked component existed on the entity before the set's first event, given whether it exists now.
    bool Existed(Entity e, bool exists_now) const { return exists_now != (Index(e) < Parity.size() && Parity[Index(e)]); }
    void emplace(Entity e, Event event) {
        emplace(e);
        if (event != Event::Update) Parity[Index(e)] ^= 1u;
    }
    // Sizes the per-index vectors to store the index.
    void Grow(uint32_t index);
    void emplace(Entity e) {
        const auto i = Index(e);
        if (i >= Positions.size()) Grow(i);
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
        Parity[Index(e)] = 0u;
        Entities.pop_back();
        return true;
    }
    void clear() {
        for (const auto e : Entities) {
            Positions[Index(e)] = UINT32_MAX;
            Parity[Index(e)] = 0u;
        }
        Entities.clear();
    }
    // Registers the set with the registry, which clears it on reset.
    DirtySet &bind(Registry &, Clearing = Clearing::Managed);
    template<typename C> DirtySet &on(On events) {
        for (const auto event : {Event::Create, Event::Update, Event::Destroy}) {
            if (uint8_t(events) & (1u << uint8_t(event))) Track(Type<C>(), event);
        }
        return *this;
    }
};

// Whether any of the sets contains an entity.
inline bool AnyChanged(const std::same_as<DirtySet> auto &...sets) { return (... || !sets.empty()); }
} // namespace state
