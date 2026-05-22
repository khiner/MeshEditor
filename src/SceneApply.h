#pragma once

#include "Action.h"
#include "SceneOps.h" // MeshInstanceCreateInfo, ElementRange

#include "gpu/Element.h"
#include "gpu/InteractionMode.h"

#include <entt/entity/fwd.hpp>

#include <expected>
#include <filesystem>
#include <span>

struct ElementRange;

// Apply(Action) is registry-only. GPU side-effects are deferred via PendingX components that ProcessComponentEvents observes.
// Apply(FallibleAction) (LoadGltf, SaveGltf, LoadRealImpact) runs GPU work synchronously and reports failure inline.
void Apply(entt::registry &, entt::entity scene_entity, const action::Action &);
std::expected<void, std::string> Apply(entt::registry &, entt::entity scene_entity, const action::FallibleAction &);

void SetStudioEnvironment(entt::registry &, entt::entity scene_entity, uint32_t index);

// Dispatches the GPU compute pass that rewrites per-element state buffers from the SelectionBitset.
// Blocking — waits on the one-shot fence before returning.
void DispatchUpdateSelectionStates(entt::registry &, entt::entity scene_entity, std::span<const ElementRange>, Element);
// Runs DispatchUpdateSelectionStates, then derives the dependent edge/face/vertex state buffers CPU-side.
void ApplySelectionStateUpdate(entt::registry &, entt::entity scene_entity, std::span<const ElementRange>, Element);

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
