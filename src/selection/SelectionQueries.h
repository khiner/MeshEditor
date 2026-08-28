#pragma once

#include "gpu/Element.h"
#include "gpu/EditSelectionOperation.h"
#include "gpu/EditSharpnessOperation.h"
#include "numeric/vec2.h"
#include "selection/BoneSelection.h"
#include "SlottedRange.h"

#include <entt/entity/fwd.hpp>

#include <optional>
#include <span>
#include <utility>
#include <vector>

struct ElementRange;

// A logical selection target: an entity, plus (in bone mode) which part of a bone was hit.
struct SelectionHit {
    entt::entity Entity;
    std::optional<BoneSel> Part{};
    bool operator==(const SelectionHit &) const = default;
};

// Map raw GPU pick/box-select instances to logical selection targets.
// In bone mode, body + joint spheres collapse to one entry per bone.
// merge_parts true merges multiple parts to nullopt (= all parts), and false keeps the first (closest) part.
// In object mode, bones fall through to SubElementOf like any other sub-element, collapsing to the armature.
std::vector<SelectionHit> ResolveHits(entt::registry &, const std::vector<entt::entity> &raw, bool bone_mode, bool merge_parts = false);

// Box selection: returns object-id-sorted entities hit by the box.
std::vector<entt::entity> RunBoxSelect(entt::registry &, entt::entity viewport, std::pair<uvec2, uvec2> box_px);

// Element-level box selection: renders IDs into the authoritative masks and derives the other domains on the GPU.
void RunBoxSelectElements(entt::registry &, entt::entity viewport, std::span<const ElementRange> ranges, Element, std::pair<uvec2, uvec2> box_px, bool is_additive);
void PublishBoxSelectElementStats(entt::registry &, entt::entity viewport);
void FinalizeBoxSelectElements(entt::registry &, entt::entity viewport);

// Object click pick. Returns hit entities sorted by (distance, depth, object id). Advances `object_pick_epoch_tag` (8-bit, wraps with periodic key reset).
std::vector<entt::entity> RunObjectPick(entt::registry &, entt::entity viewport, uint32_t &object_pick_epoch_tag, uvec2 mouse_px, uint32_t radius_px = 0);

// Pick the nearest sound-vertex of an instance under the cursor.
std::optional<uint32_t> RunSoundVerticesVertexPick(entt::registry &, entt::entity viewport, entt::entity instance_entity, uvec2 mouse_px);

// Element-level click transaction: pick, mutate, derive, and summarize on the GPU; returns the hit only for CPU editor mirrors.
std::optional<std::pair<entt::entity, uint32_t>> RunEditElementClick(entt::registry &, entt::entity viewport, std::span<const ElementRange> ranges, Element, uvec2 mouse_px, bool toggle);

void ApplyEditSelectionCommand(entt::registry &, entt::entity viewport, std::span<const ElementRange>, Element, EditSelectionOperation);
void ApplyEditSelectionLists(entt::registry &, entt::entity viewport, std::span<const std::pair<entt::entity, SlottedRange>>, Element);
void ApplyEditSharpness(entt::registry &, entt::entity viewport, std::span<const entt::entity> mesh_entities, EditSharpnessOperation, bool value = false, float angle = 0.f);
// Publish the mesh-local selection aggregates consumed by UI and transform tools.
void RefreshElementSelectionStats(entt::registry &, entt::entity mesh_entity);
void RefreshElementSelectionSharpness(entt::registry &, entt::entity mesh_entity);
