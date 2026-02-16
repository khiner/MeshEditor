#pragma once

#include "entt_fwd.h"

#include <string>
#include <string_view>

struct Name {
    std::string Value;
};

// Invariants:
// * Zero or more entities can be Selected.
// * An entity can be both Selected and Active. If an entity is Active, it must also be Selected.
// * There may be Selected entities but no Active entity.
struct Selected {};
struct Active {};

struct Frozen {}; // Disable scale changes (translate/rotate still allowed)

enum class ObjectType : uint8_t {
    Empty,
    Mesh,
    Armature,
    Camera,
    Light,
};

struct ObjectKind {
    ObjectType Value{ObjectType::Empty};
};

constexpr std::string_view ObjectTypeName(ObjectType type) {
    switch (type) {
        case ObjectType::Empty: return "Empty";
        case ObjectType::Mesh: return "Mesh";
        case ObjectType::Armature: return "Armature";
        case ObjectType::Camera: return "Camera";
        case ObjectType::Light: return "Light";
    }
}

std::string IdString(entt::entity);
std::string GetName(const entt::registry &, entt::entity); // Returns name if present, otherwise hex ID.
std::string CreateName(const entt::registry &, std::string_view prefix = "Entity");

entt::entity FindActiveEntity(const entt::registry &); // If no active entity, returns entt::null.
