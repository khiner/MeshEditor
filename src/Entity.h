#pragma once

#include <entt/entity/fwd.hpp>

#include <string>
#include <string_view>

struct Name {
    std::string Value;
};

// Invariants:
// * Zero or more entities can be Selected.
// * At most one entity can be Active. Active and Selected are independent.
// * Active persists until explicitly replaced by a new pick/select action.
struct Selected {};
struct Active {};

// Sub-elements are not independently selectable in Object mode.
// Picking/selection routes to Parent. Origin dot drawn only on Parent.
// Examples: armature bones, future duplivert instances.
struct SubElementOf {
    entt::entity Parent;
};

struct ScaleLocked {}; // Disable scale changes (translate/rotate still allowed)

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

entt::entity FindActiveEntity(const entt::registry &); // If no active entity, returns entt::null.

// Mesh-data entity behind an instance. GetMeshEntity returns null for non-mesh instances.
entt::entity GetMeshEntity(const entt::registry &, entt::entity);
entt::entity GetActiveMeshEntity(const entt::registry &);
entt::entity FindMeshEntity(const entt::registry &, entt::entity); // Instance's mesh entity, else the entity itself.
