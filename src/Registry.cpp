#include "Registry.h"

#include <entt/entity/registry.hpp>

#include <cstdint>
#include <format>
#include <ranges>

std::string IdString(entt::entity e) { return std::format("0x{:08x}", uint32_t(e)); }
std::string GetName(const entt::registry &r, entt::entity e) {
    if (e == entt::null) return "null";

    if (const auto *name = r.try_get<Name>(e)) {
        if (!name->Value.empty()) return name->Value;
    }
    return IdString(e);
}
std::string CreateName(const entt::registry &r, std::string_view prefix) {
    const std::string prefix_str{prefix};
    for (uint32_t i = 0; i < std::numeric_limits<uint32_t>::max(); ++i) {
        const auto view = r.view<const Name>();
        const auto name = i == 0 ? prefix_str : std::format("{}_{}", prefix, i);
        if (!std::ranges::any_of(view, [&](auto e) { return view.get<Name>(e).Value == name; })) {
            return name;
        }
    }
    assert(false);
    return prefix_str;
}

entt::entity FindActiveEntity(const entt::registry &registry) {
    auto all_active = registry.view<Active>();
    assert(all_active.size() <= 1);
    return all_active.empty() ? entt::null : *all_active.begin();
}

entt::entity GetParentEntity(const entt::registry &r, entt::entity e) {
    if (e == entt::null) return entt::null;

    if (const auto *node = r.try_get<SceneNode>(e)) {
        return node->Parent == entt::null ? e : GetParentEntity(r, node->Parent);
    }
    return e;
}
