#pragma once

#include "Action.h"
#include "SceneOps.h" // MeshInstanceCreateInfo

#include "gpu/Element.h"
#include "gpu/InteractionMode.h"

#include <entt/entity/fwd.hpp>

#include <expected>
#include <filesystem>

// Apply(Action) is registry-only. GPU side-effects are deferred via PendingX components that ProcessComponentEvents observes.
// Apply(FallibleAction) (LoadGltf, SaveGltf, LoadRealImpact) runs GPU work synchronously and reports failure inline.
void Apply(entt::registry &, entt::entity scene_entity, const action::Action &);
std::expected<void, std::string> Apply(entt::registry &, entt::entity scene_entity, const action::FallibleAction &);

bool SetInteractionMode(entt::registry &, entt::entity scene_entity, InteractionMode);
std::pair<entt::entity, entt::entity> ImportMesh(
    entt::registry &, entt::entity scene_entity,
    const std::filesystem::path &, MeshInstanceCreateInfo
);
void Destroy(entt::registry &, entt::entity scene_entity, entt::entity);

bool CanDuplicate(const entt::registry &, entt::entity scene_entity);
bool CanDuplicateLinked(const entt::registry &, entt::entity scene_entity);
bool CanDelete(const entt::registry &, entt::entity scene_entity);
entt::entity GetMeshEntity(const entt::registry &, entt::entity);
entt::entity GetActiveMeshEntity(const entt::registry &);
entt::entity LookThroughCameraEntity(const entt::registry &);
entt::entity FindMeshEntity(const entt::registry &, entt::entity);
std::vector<ElementRange> GetBitsetRangesForSelected(const entt::registry &);
bool IsBoneEditMode(const entt::registry &, entt::entity scene_entity);
bool AllSelectedAreMeshes(const entt::registry &);
