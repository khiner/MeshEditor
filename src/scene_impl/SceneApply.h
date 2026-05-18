#pragma once

#include <entt/entity/fwd.hpp>

#include <vector>

struct ElementRange;

namespace scene_apply {

// Pure inspectors over the registry — safe for UI/query paths without touching Scene-owned GPU state.
bool CanDuplicate(const entt::registry &, entt::entity scene_entity);
bool CanDuplicateLinked(const entt::registry &, entt::entity scene_entity);
bool CanDelete(const entt::registry &, entt::entity scene_entity);
entt::entity GetMeshEntity(const entt::registry &, entt::entity);
entt::entity GetActiveMeshEntity(const entt::registry &);
entt::entity LookThroughCameraEntity(const entt::registry &);
std::vector<ElementRange> GetBitsetRangesForSelected(const entt::registry &);

} // namespace scene_apply
