#pragma once

#include "state/Entity.h"

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

struct Name {
    std::string Value;
};

// Initialize/tear down the derived index of live entity names.
void InitEntityNames(state::Scene &);
void DeinitEntityNames(state::Scene &);
void RebuildEntityNames(state::Scene &);
void ReserveEntityNames(state::Scene &, size_t additional);

// Choose an unused editor name and attach it to an entity.
Name &EmplaceUniqueName(state::Scene &, state::Entity, std::string_view prefix);

// A scene: a named, possibly empty grouping of objects.
struct Scene {
    std::string Name;
};
// Tag on the currently shown scene.
struct ActiveScene {};
// Scenes this object is in. Absent when there's only one scene (everything's in it).
struct SceneMembership {
    std::vector<state::Entity> Scenes;
};

// Invariants:
// * Zero or more entities can be Selected.
// * At most one entity can be Active. Active and Selected are independent.
// * Active persists until explicitly replaced by a new pick/select action.
struct Selected {};
struct Active {};

// Most recently selected element within a mesh (remembered even when not selected).
struct MeshActiveElement {
    uint32_t Handle;
};

// Sub-elements are not independently selectable in Object mode.
// Picking/selection routes to Parent. Origin dot drawn only on Parent.
// Examples: armature bones, future duplivert instances.
struct SubElementOf {
    state::Entity Parent;
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

// Canonical entity order for consumers whose output depends on traversal order.
// Component storage order changes with insertion, deletion, and snapshot reconstruction.
template<typename Compare = std::ranges::less>
std::vector<state::Entity> SortedEntities(std::ranges::input_range auto &&entities, Compare compare = {}) {
    auto sorted = entities | std::ranges::to<std::vector<state::Entity>>();
    std::ranges::sort(sorted, compare);
    return sorted;
}

std::string IdString(state::Entity);
std::string GetName(const state::Scene &, state::Entity); // Returns name if present, otherwise hex ID.

state::Entity FindActiveEntity(const state::Scene &); // If no active entity, returns state::Null.

// Mesh-data entity behind an instance. GetMeshEntity returns null for non-mesh instances.
state::Entity GetMeshEntity(const state::Scene &, state::Entity);
state::Entity GetActiveMeshEntity(const state::Scene &);
state::Entity FindMeshEntity(const state::Scene &, state::Entity); // Instance's mesh entity, else the entity itself.
