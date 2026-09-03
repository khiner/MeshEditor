#pragma once

#include <entt/entity/fwd.hpp>

#include <unordered_map>
#include <unordered_set>
#include <vector>

struct ElementRange;

bool AllSelectedAreMeshes(const entt::registry &);
bool IsBoneEditMode(const entt::registry &, entt::entity viewport);
bool CanDuplicate(const entt::registry &, entt::entity viewport);
bool CanDuplicateLinked(const entt::registry &, entt::entity viewport);
bool CanDelete(const entt::registry &, entt::entity viewport);
std::vector<ElementRange> GetElementRangesForSelected(const entt::registry &, entt::entity viewport);

// Returns the containing armature, or null if the entity is unrelated to an armature.
entt::entity FindArmatureObject(const entt::registry &, entt::entity);
// Returns null if no bone is active.
entt::entity FindActiveBone(const entt::registry &);

// Returns selected transform roots, using bones in pose or edit mode and objects otherwise.
std::vector<entt::entity> RootSelectedForTransform(const entt::registry &, entt::entity viewport);

struct EditTransformContext {
    std::unordered_map<entt::entity, entt::entity> TransformInstances;
};

namespace selection {
using PrimaryEditInstanceMap = std::unordered_map<entt::entity, entt::entity>;
struct PrimaryEditInstanceMaps {
    PrimaryEditInstanceMap All, Transformable;
};

// Returns one selected mesh instance per mesh, preferring the active instance.
PrimaryEditInstanceMap ComputePrimaryEditInstances(const entt::registry &, bool include_scale_locked = true);
PrimaryEditInstanceMaps ComputePrimaryEditInstanceMaps(const entt::registry &);

bool HasScaleLockedInstance(const entt::registry &, entt::entity);
std::unordered_set<entt::entity> GetSelectedMeshEntities(const entt::registry &);
} // namespace selection
