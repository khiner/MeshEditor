#pragma once

#include <entt/entity/fwd.hpp>

#include <string>

struct Name {
    std::string Value;
};

std::string IdString(entt::entity);
std::string GetName(const entt::registry &, entt::entity); // Returns name if present, otherwise hex ID.
entt::entity GetParentEntity(const entt::registry &, entt::entity);
