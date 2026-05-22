#pragma once

#include <entt/entity/registry.hpp>

#include <functional>
#include <vector>

enum class On : uint8_t {
    Create = 1,
    Update = 2,
    Destroy = 4
};
constexpr On operator|(On a, On b) { return On(uint8_t(a) | uint8_t(b)); }

// Like the default `basic_reactive_mixin::emplace_element`, but erases a stale-version
// slot entry first so a recycled id doesn't collide with the prior version.
inline void EmplaceSafe(entt::storage_for_t<entt::reactive> &s, const entt::registry &, entt::entity e) {
    if (s.contains(e)) return;
    using traits = entt::entt_traits<entt::entity>;
    if (const auto stored_ver = s.current(e); stored_ver != traits::to_version(entt::tombstone)) {
        s.erase(traits::construct(traits::to_entity(e), stored_ver));
    }
    s.emplace(e);
}

struct ReactiveTracker {
    entt::storage_for_t<entt::reactive> &s;
    template<typename T> ReactiveTracker &on(On events) {
        if (uint8_t(events) & uint8_t(On::Create)) s.on_construct<T, &EmplaceSafe>();
        if (uint8_t(events) & uint8_t(On::Update)) s.on_update<T, &EmplaceSafe>();
        if (uint8_t(events) & uint8_t(On::Destroy)) s.on_destroy<T, &EmplaceSafe>();
        return *this;
    }
};

template<typename Change>
ReactiveTracker track(entt::registry &r) { return {r.storage<entt::reactive>(entt::type_hash<Change>::value())}; }

template<typename Change>
auto &reactive(entt::registry &r) { return r.storage<entt::reactive>(entt::type_hash<Change>::value()); }

// Lets domain systems (e.g. audio) register per-frame reactive handlers without coupling to Scene.
using ComponentEventHandler = std::function<void(entt::registry &)>;

inline void RegisterComponentEventHandler(entt::registry &r, ComponentEventHandler handler) {
    // emplace is a no-op if the type already exists (try_emplace semantics).
    r.ctx().emplace<std::vector<ComponentEventHandler>>().emplace_back(std::move(handler));
}

inline void ProcessComponentEventHandlers(entt::registry &r) {
    if (const auto *handlers = r.ctx().find<std::vector<ComponentEventHandler>>()) {
        for (const auto &h : *handlers) h(r);
    }
}
