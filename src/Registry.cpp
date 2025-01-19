#include "Registry.h"

#include <entt/entity/registry.hpp>

#include <cstdint>
#include <format>

std::string IdString(entt::entity entity) { return std::format("0x{:08x}", uint32_t(entity)); }
std::string GetName(const entt::registry &r, entt::entity entity) {
    if (entity == entt::null) return "null";

    if (const auto *name = r.try_get<Name>(entity)) {
        if (!name->Value.empty()) return name->Value;
    }
    return IdString(entity);
}