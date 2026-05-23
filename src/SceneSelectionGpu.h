#pragma once

#include "gpu/Element.h"
#include "numeric/vec2.h"

#include <entt/entity/fwd.hpp>
#include <vulkan/vulkan.hpp>

#include <functional>
#include <optional>
#include <span>
#include <vector>

struct ElementRange;
struct DrawListBuilder;
struct SelectionDrawInfo;

using SelectionBuildFn = std::function<std::vector<SelectionDrawInfo>(DrawListBuilder &)>;

// Render the on-demand selection-fragment pass. `build_fn` populates the draw list given the silhouette-prefilled builder.
void RenderSelectionPassWith(entt::registry &, entt::entity viewport, bool render_depth, const SelectionBuildFn &, vk::Semaphore signal_semaphore = {}, bool render_silhouette = true);

// Replays the cached selection draw list (built by RecordRenderCommandBuffer). Clears SelectionStale on success.
void RenderSelectionPass(entt::registry &, entt::entity viewport, vk::Semaphore signal_semaphore = {});

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
