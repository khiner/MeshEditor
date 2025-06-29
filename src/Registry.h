#pragma once

#include <entt/entity/entity.hpp>

#include <string>
#include <string_view>
#include <vector>

struct Name {
    std::string Value;
};

// Invariants:
// * Zero or more entities can be Selected.
// * The (single) most recently Selected entity is Active.
struct Selected {};
struct Active {};

struct Visible {}; // Visible in the scene
struct Frozen {}; // Disable entity transform changes

struct SceneNode {
    entt::entity Parent{entt::null};
    std::vector<entt::entity> Children{};
    uint32_t ModelBufferIndex{0}; // Index in the contiguous model buffer.
};

std::string IdString(entt::entity);
std::string GetName(const entt::registry &, entt::entity); // Returns name if present, otherwise hex ID.
std::string CreateName(const entt::registry &, std::string_view prefix = "Entity");

entt::entity FindActiveEntity(const entt::registry &); // If no active entity, returns entt::null.
entt::entity GetParentEntity(const entt::registry &, entt::entity); // If no parent, returns the provided entity.
