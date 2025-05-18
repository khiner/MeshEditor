#pragma once

#include <entt/entity/entity.hpp>

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

struct Name {
    std::string Value;
};

struct Selected {}; // Entity is selected (multiple can be selected)
// Active selected entity
// Invariants:
//   * <=1 entity is active at a time.
//   * If an entity is Active, it is also Selected.
//   * Most recently Selected entity is Active.
struct Active {};
struct Visible {}; // Visible in the scene
struct Frozen {}; // Disable entity transform changes

struct SceneNode {
    entt::entity Parent{entt::null};
    std::vector<entt::entity> Children;
    // `ModelIndices` maps entities to their index in the models buffer. Includes parent.
    // It's only present in parent nodes.
    // This enables contiguous storage of models in the buffer, with erases and appends but no inserts.
    std::unordered_map<entt::entity, uint32_t> ModelIndices;
};

std::string IdString(entt::entity);
std::string GetName(const entt::registry &, entt::entity); // Returns name if present, otherwise hex ID.
std::string CreateName(const entt::registry &, std::string_view prefix = "Entity");

entt::entity FindActiveEntity(const entt::registry &);
entt::entity GetParentEntity(const entt::registry &, entt::entity);
