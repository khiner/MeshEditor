#pragma once

#include <entt/entity/fwd.hpp>

#include <string>
#include <string_view>

struct Name {
    std::string Value;
};

std::string IdString(entt::entity);
std::string GetName(const entt::registry &, entt::entity); // Returns name if present, otherwise hex ID.
std::string CreateName(const entt::registry &, std::string_view prefix = "Entity");
entt::entity GetParentEntity(const entt::registry &, entt::entity);
