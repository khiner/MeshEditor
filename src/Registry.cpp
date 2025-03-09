#include "Registry.h"

#include <entt/entity/registry.hpp>

#include <cstdint>
#include <format>
#include <ranges>

std::string IdString(entt::entity entity) { return std::format("0x{:08x}", uint32_t(entity)); }
std::string GetName(const entt::registry &r, entt::entity entity) {
    if (entity == entt::null) return "null";

    if (const auto *name = r.try_get<Name>(entity)) {
        if (!name->Value.empty()) return name->Value;
    }
    return IdString(entity);
}
std::string CreateName(const entt::registry &r, std::string_view prefix) {
    const std::string prefix_str{prefix};
    for (uint32_t i = 0; i < std::numeric_limits<uint32_t>::max(); ++i) {
        const auto view = r.view<const Name>();
        const auto name = i == 0 ? prefix_str : std::format("{}_{}", prefix, i);
        if (!std::ranges::any_of(view, [&](auto entity) { return view.get<Name>(entity).Value == name; })) {
            return name;
        }
    }
    assert(false);
    return prefix_str;
}
