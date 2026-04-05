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

struct ReactiveTracker {
    entt::storage_for_t<entt::reactive> &s;
    template<typename T> ReactiveTracker &on(On events) {
        if (uint8_t(events) & uint8_t(On::Create)) s.on_construct<T>();
        if (uint8_t(events) & uint8_t(On::Update)) s.on_update<T>();
        if (uint8_t(events) & uint8_t(On::Destroy)) s.on_destroy<T>();
        return *this;
    }
};

template<typename Change>
ReactiveTracker track(entt::registry &r) { return {r.storage<entt::reactive>(entt::type_hash<Change>::value())}; }

template<typename Change>
auto &reactive(entt::registry &r) { return r.storage<entt::reactive>(entt::type_hash<Change>::value()); }

// Only usage of EnTT registry context in the app. Used to let domain systems (e.g. audio)
// register per-frame reactive handlers without coupling to Scene. May want to revisit this.
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
