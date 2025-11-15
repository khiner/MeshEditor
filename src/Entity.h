#pragma once

#include "entt_fwd.h"

#include <string>
#include <string_view>

struct Name {
    std::string Value;
};

// Invariants:
// * Zero or more entities can be Selected.
// * The (single) most recently Selected entity is Active.
// NOTE: An entity can be both Selected and Active, and can also be Active but not Selected.
struct Selected {};
struct Active {};

struct Frozen {}; // Disable entity transform changes

std::string IdString(entt::entity);
std::string GetName(const entt::registry &, entt::entity); // Returns name if present, otherwise hex ID.
std::string CreateName(const entt::registry &, std::string_view prefix = "Entity");

entt::entity FindActiveEntity(const entt::registry &); // If no active entity, returns entt::null.
