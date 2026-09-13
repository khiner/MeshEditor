#pragma once

#include <entt/entity/registry.hpp>

#include <vector>

namespace project {
bool Restoring(const entt::registry &);
void Capture(entt::registry &, entt::id_type, entt::entity);
entt::entity Create(entt::registry &);
void Destroy(entt::registry &, entt::entity);
void Reset(entt::registry &);

template<typename C> void Capture(entt::registry &r, entt::entity e) {
    Capture(r, entt::type_hash<C>::value(), e);
}

template<typename C, typename... Args> decltype(auto) Emplace(entt::registry &r, entt::entity e, Args &&...args) {
    Capture<C>(r, e);
    return r.emplace<C>(e, std::forward<Args>(args)...);
}

template<typename C, typename... Args> decltype(auto) EmplaceOrReplace(entt::registry &r, entt::entity e, Args &&...args) {
    Capture<C>(r, e);
    return r.emplace_or_replace<C>(e, std::forward<Args>(args)...);
}

template<typename C, typename... Args> decltype(auto) GetOrEmplace(entt::registry &r, entt::entity e, Args &&...args) {
    Capture<C>(r, e);
    return r.get_or_emplace<C>(e, std::forward<Args>(args)...);
}

template<typename C, typename... Args> decltype(auto) Replace(entt::registry &r, entt::entity e, Args &&...args) {
    Capture<C>(r, e);
    return r.replace<C>(e, std::forward<Args>(args)...);
}

template<typename C, typename... Fn> decltype(auto) Patch(entt::registry &r, entt::entity e, Fn &&...fn) {
    Capture<C>(r, e);
    return r.patch<C>(e, std::forward<Fn>(fn)...);
}

template<typename... C> auto Remove(entt::registry &r, entt::entity e) {
    (Capture<C>(r, e), ...);
    return r.remove<C...>(e);
}

template<typename... C> void Erase(entt::registry &r, entt::entity e) {
    (Capture<C>(r, e), ...);
    r.erase<C...>(e);
}

template<typename... C> void Clear(entt::registry &r) {
    ([&] {
        for (const auto e : r.view<C>()) Capture<C>(r, e);
        r.clear<C>();
    }(),
     ...);
}

// Use the returned reference only until the next History::Pin or navigation.
template<typename C> C &Mutable(entt::registry &r, entt::entity e) {
    Capture<C>(r, e);
    return r.get<C>(e);
}

template<typename C> C *TryMutable(entt::registry &r, entt::entity e) {
    if (auto *value = r.try_get<C>(e)) {
        Capture<C>(r, e);
        return value;
    }
    return nullptr;
}
} // namespace project
