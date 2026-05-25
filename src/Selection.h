#pragma once

#include <entt/entity/fwd.hpp>

#include <unordered_map>
#include <unordered_set>
#include <vector>

struct ElementRange;

// Scene/selection predicates and queries used by the UI and action handlers.
bool AllSelectedAreMeshes(const entt::registry &);
bool IsBoneEditMode(const entt::registry &, entt::entity viewport);
bool CanDuplicate(const entt::registry &, entt::entity viewport);
bool CanDuplicateLinked(const entt::registry &, entt::entity viewport);
bool CanDelete(const entt::registry &, entt::entity viewport);
std::vector<ElementRange> GetBitsetRangesForSelected(const entt::registry &);

struct EditTransformContext {
    std::unordered_map<entt::entity, entt::entity> TransformInstances; // excludes frozen, for transforms
};

namespace selection {

// Returns representative edit instance per selected mesh: active instance if selected, else first selected instance.
// Only includes Mesh-type objects (excludes Cameras, etc.).
std::unordered_map<entt::entity, entt::entity> ComputePrimaryEditInstances(const entt::registry &, bool include_scale_locked = true);

bool HasScaleLockedInstance(const entt::registry &, entt::entity);
std::unordered_set<entt::entity> GetSelectedMeshEntities(const entt::registry &);

} // namespace selection
