#pragma once

#include "gpu/Element.h"
#include "numeric/vec2.h"
#include "selection/BoneSelection.h"

#include <entt/entity/fwd.hpp>

#include <optional>
#include <span>
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
// merge_parts: true merges multiple parts to nullopt (= all parts); false keeps the first (closest) part.
// In object mode, bones fall through to SubElementOf like any other sub-element, collapsing to the armature.
std::vector<SelectionHit> ResolveHits(entt::registry &, const std::vector<entt::entity> &raw, bool bone_mode, bool merge_parts = false);

// Vulkan-free GPU pick/box-select queries, so interaction/UI consumers need not
// include the (vulkan-heavy) SelectionGpu header.

// Box selection: returns object-id-sorted entities hit by the box.
std::vector<entt::entity> RunBoxSelect(entt::registry &, entt::entity viewport, std::pair<uvec2, uvec2> box_px);

// Element-level box selection: renders element IDs into the bitset over the box region, then GPU-updates the element state buffers.
void RunBoxSelectElements(entt::registry &, entt::entity viewport, std::span<const ElementRange> ranges, Element, std::pair<uvec2, uvec2> box_px, bool is_additive);

// Object click pick. Returns hit entities sorted by (distance, depth, object id). Advances `object_pick_epoch_tag` (8-bit, wraps with periodic key reset).
std::vector<entt::entity> RunObjectPick(entt::registry &, entt::entity viewport, uint32_t &object_pick_epoch_tag, uvec2 mouse_px, uint32_t radius_px = 0);

// Pick the nearest sound-vertex of an instance under the cursor.
std::optional<uint32_t> RunSoundVerticesVertexPick(entt::registry &, entt::entity viewport, entt::entity instance_entity, uvec2 mouse_px);

// Element-level click pick. Returns the (mesh_entity, element_index) under the cursor.
std::optional<std::pair<entt::entity, uint32_t>> RunElementPickFromRanges(entt::registry &, entt::entity viewport, std::span<const ElementRange> ranges, Element, uvec2 mouse_px);

// Dispatches the GPU compute pass that rewrites per-element state buffers from the SelectionBitset. Blocks on the one-shot fence.
void DispatchUpdateSelectionStates(entt::registry &, std::span<const ElementRange>, Element);
// Runs DispatchUpdateSelectionStates, then derives the dependent edge/face/vertex state buffers CPU-side.
void ApplySelectionStateUpdate(entt::registry &, entt::entity viewport, std::span<const ElementRange>, Element);
