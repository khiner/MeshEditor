#include "Entity.h"

#include <entt/entity/registry.hpp>

#include <format>

static_assert(null_entity == entt::null, "null_entity does not match entt::null");

std::string IdString(entt::entity e) { return std::format("0x{:08x}", uint32_t(e)); }
std::string GetName(const entt::registry &r, entt::entity e) {
    if (e == entt::null) return "null";

    if (const auto *name = r.try_get<Name>(e)) {
        if (!name->Value.empty()) return name->Value;
    }
    return IdString(e);
}

entt::entity FindActiveEntity(const entt::registry &registry) {
    auto all_active = registry.view<Active>();
    assert(all_active.size() <= 1);
    return all_active.empty() ? entt::null : *all_active.begin();
}
