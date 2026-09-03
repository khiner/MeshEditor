#pragma once

#include "SlottedRange.h"
#include "gpu/EditSelectionOperation.h"
#include "gpu/EditSharpnessOperation.h"
#include "gpu/Element.h"
#include "numeric/vec2.h"
#include "selection/BoneSelection.h"

#include <entt/entity/fwd.hpp>

#include <optional>
#include <span>
#include <utility>
#include <vector>

struct ElementRange;

struct SelectionHit {
    entt::entity Entity;
    std::optional<BoneSel> Part{};
    bool operator==(const SelectionHit &) const = default;
};

// Resolves raw GPU hits to logical targets, collapsing bone parts and object sub-elements.
std::vector<SelectionHit> ResolveHits(entt::registry &, const std::vector<entt::entity> &raw, bool bone_mode, bool merge_parts = false);

// Returns box hits in object-id order.
std::vector<entt::entity> RunBoxSelect(entt::registry &, std::pair<uvec2, uvec2> box_px);

// Element-level box selection: renders IDs into the authoritative masks and derives the other domains on the GPU.
void RunBoxSelectElements(entt::registry &, entt::entity viewport, std::span<const ElementRange> ranges, Element, std::pair<uvec2, uvec2> box_px, bool is_additive);
void PublishBoxSelectElementStats(entt::registry &, entt::entity viewport);
void FinalizeBoxSelectElements(entt::registry &, entt::entity viewport);

// Returns click hits sorted by distance, depth, and object id, then advances the 8-bit epoch tag.
std::vector<entt::entity> RunObjectPick(entt::registry &, uint32_t &object_pick_epoch_tag, uvec2 mouse_px, uint32_t radius_px = 0);

// Pick the nearest sound-vertex of an instance under the cursor.
std::optional<uint32_t> RunSoundVerticesVertexPick(entt::registry &, entt::entity instance_entity, uvec2 mouse_px);

// Runs an element-level pick, mutation, derivation, and summary on the GPU.
// Returns the hit only for CPU editor mirrors.
std::optional<std::pair<entt::entity, uint32_t>> RunEditElementClick(entt::registry &, entt::entity viewport, std::span<const ElementRange> ranges, Element, uvec2 mouse_px, bool toggle);

void ApplyEditSelectionCommand(entt::registry &, entt::entity viewport, std::span<const ElementRange>, Element, EditSelectionOperation);
void ApplyEditSelectionLists(entt::registry &, entt::entity viewport, std::span<const std::pair<entt::entity, SlottedRange>>, Element);
void ApplyEditSharpness(entt::registry &, entt::entity viewport, std::span<const entt::entity> mesh_entities, EditSharpnessOperation, bool value = false, float angle = 0.f);
// Publish the mesh-local selection aggregates consumed by UI and transform tools.
void RefreshElementSelectionStats(entt::registry &, entt::entity mesh_entity);
void RefreshElementSelectionSharpness(entt::registry &, entt::entity mesh_entity);
