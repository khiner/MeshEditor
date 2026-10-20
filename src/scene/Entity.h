#pragma once

#include "scene/SceneSchema.h"
#include "state/Entity.h"

#include <algorithm>
#include <ranges>
#include <vector>

struct Name {
    std::string Value;
};

// Initialize/tear down the derived index of live entity names.
void InitEntityNames(state::Registry &);
void DeinitEntityNames(state::Registry &);
void RebuildEntityNames(state::Registry &);
void ReserveEntityNames(state::Registry &, size_t additional);

// Choose an unused editor name and attach it to an entity.
// A taken prefix gets a suffix past every live "{prefix}_{N}".
Name &EmplaceUniqueName(state::Registry &, state::Entity, std::string_view prefix);
// The name without one trailing `_<digits>` suffix, which a unique name or a duplicate renumbers.
std::string_view NameStem(std::string_view);

// A scene: a possibly empty grouping of objects, named by its Name.
// A document holds at least one.
struct Scene {};
// A document holds at most this many scenes, one per bit of the physics scene word.
inline constexpr uint32_t MaxScenes = 64;
// The scenes a root node is in, never empty.
// A node with a parent is in its root's scenes.
struct SceneMembership {
    std::vector<state::Entity> Scenes;
};

// The scene when it holds a Scene, else the lowest-index scene, else null.
state::Entity ResolveScene(const state::Registry &, state::Entity scene);
// The frame's scene resolved to a live scene, which actions and the settle pass act on.
// A registry without frame inputs acts on its lowest-index scene.
state::Entity FrameScene(const state::Registry &);

// Invariants:
// * Zero or more entities can be Selected.
// * At most one entity can be Active.
// Active and Selected are independent.
// * Active persists until explicitly replaced by a new pick/select action.
struct Selected {};
struct Active {};

// Most recently selected element within a mesh (remembered even when not selected).
struct MeshActiveElement {
    uint32_t Handle;
};

// Sub-elements are not independently selectable in Object mode.
// Picking/selection routes to Parent.
// Origin dot drawn only on Parent.
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

// Makes the entity an object of the type, a member of the frame's scene when it has no parent.
void EmplaceObject(state::Registry &, state::Entity, ObjectType);

// Canonical entity order for consumers whose output depends on traversal order.
// Component storage order changes with insertion, deletion, and snapshot reconstruction.
template<typename Compare = std::ranges::less>
std::vector<state::Entity> SortedEntities(std::ranges::input_range auto &&entities, Compare compare = {}) {
    auto sorted = entities | std::ranges::to<std::vector<state::Entity>>();
    std::ranges::sort(sorted, compare);
    return sorted;
}

// Return whether entities contains e; require entity-index order.
inline bool ContainsInIndexOrder(std::span<const state::Entity> entities, state::Entity e) {
    const auto it = std::ranges::lower_bound(entities, state::Index(e), {}, state::Index);
    return it != entities.end() && *it == e;
}

std::string IdString(state::Entity);
std::string GetName(const state::Registry &, state::Entity); // Returns name if present, otherwise hex ID.

// The active entity when it is in the scene, else state::Null.
state::Entity FindActiveEntity(const state::Registry &, state::Entity scene);
// The active entity when it is in the frame's scene, else state::Null.
state::Entity FindActiveEntity(const state::Registry &);

// Mesh-data entity behind an instance.
// GetMeshEntity returns null for non-mesh instances.
state::Entity GetMeshEntity(const state::Registry &, state::Entity);
state::Entity GetActiveMeshEntity(const state::Registry &);
state::Entity FindMeshEntity(const state::Registry &, state::Entity); // Instance's mesh entity, else the entity itself.
