#pragma once

#include "SlottedRange.h"
#include "gpu/EditSelectionOperation.h"
#include "gpu/EditSelectionSummary.h"
#include "gpu/EditSharpnessOperation.h"
#include "gpu/Element.h"
#include "metal/Bindless.h"
#include "numeric/vec2.h"

#include "state/Entity.h"

#include <optional>
#include <span>
#include <utility>
#include <vector>

struct ElementRange;

// Bindless slots of the object and element pick buffers.
struct SelectionSlots {
    explicit SelectionSlots(mtl::BindlessSet &slots)
        : Owner(slots),
          ObjectPickKey(Owner.Allocate(SlotType::Buffer)), ElementPickKey(Owner.Allocate(SlotType::Buffer)), ElementPickId(Owner.Allocate(SlotType::Buffer)),
          ObjectPickSeenBits(Owner.Allocate(SlotType::Buffer)), ObjectBoxBitset(Owner.Allocate(SlotType::Buffer)) {}

    mtl::SlotOwner Owner;
    uint32_t ObjectPickKey, ElementPickKey, ElementPickId, ObjectPickSeenBits, ObjectBoxBitset;
};

// Returns box hits in object-id order.
std::vector<state::Entity> RunBoxSelect(state::Scene &, std::pair<uvec2, uvec2> box_px);

// Element-level box selection: renders IDs into the authoritative masks and derives the other domains on the GPU.
void RunBoxSelectElements(state::Scene &, state::Entity viewport, std::span<const ElementRange> ranges, Element, std::pair<uvec2, uvec2> box_px, bool is_additive);

// Returns click hits sorted by distance, depth, and object id, then advances the 8-bit epoch tag.
std::vector<state::Entity> RunObjectPick(state::Scene &, uvec2 mouse_px, uint32_t radius_px = 0);

// Pick the nearest sound-vertex of an instance under the cursor.
std::optional<uint32_t> RunSoundVerticesVertexPick(state::Scene &, state::Entity instance_entity, uvec2 mouse_px);

// Runs an element-level pick, mutation, derivation, and summary on the GPU.
// Returns the hit only for CPU editor mirrors.
std::optional<std::pair<state::Entity, uint32_t>> RunEditElementClick(state::Scene &, state::Entity viewport, std::span<const ElementRange> ranges, Element, uvec2 mouse_px, bool toggle);

void ApplyEditSelectionCommand(state::Scene &, std::span<const ElementRange>, Element, EditSelectionOperation);
void ApplyEditSelectionLists(state::Scene &, std::span<const std::pair<state::Entity, SlottedRange>>, Element);
void ApplyEditSharpness(state::Scene &, state::Entity viewport, std::span<const state::Entity> mesh_entities, EditSharpnessOperation, bool value = false, float angle = 0.f);
// Read the shared GPU summary after selection work has completed. Other element domains have no current summary.
const EditSelectionSummary *GetElementSelectionSummary(const state::Scene &, state::Entity mesh_entity, Element);
