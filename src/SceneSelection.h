#pragma once

#include "entt_fwd.h"
#include "gpu/Element.h"

#include <cstdint>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Mesh;

namespace scene_selection {

// Set all bits in [offset, offset+count), clearing any gap bits in the last word.
void SelectAll(uint32_t *bits, uint32_t offset, uint32_t count);
// Count selected bits in [offset, offset+count).
uint32_t CountSelected(const uint32_t *bits, uint32_t offset, uint32_t count);
// Return local (0-based) handles of all set bits in [offset, offset+count).
std::vector<uint32_t> ScanBitsetRange(const uint32_t *bits, uint32_t offset, uint32_t count);
std::vector<uint32_t> ConvertSelectionElement(std::span<const uint32_t> handles, const Mesh &, Element from_element, Element to_element);

// Returns representative edit instance per selected mesh: active instance if selected, else first selected instance.
// Only includes Mesh-type objects (excludes Cameras, etc.).
std::unordered_map<entt::entity, entt::entity> ComputePrimaryEditInstances(const entt::registry &, bool include_scale_locked = true);

bool HasScaleLockedInstance(const entt::registry &, entt::entity);
std::unordered_set<entt::entity> GetSelectedMeshEntities(const entt::registry &);
uint32_t GetElementCount(const Mesh &, Element);

// Mesh vertices targeted by a sound-object sample operation (Add/Replace/Remove).
// Excite mode: single active vertex on the sound object's mesh.
// Edit mode: selected vertices on the sound object's mesh (edges/faces converted via ConvertSelectionElement).
// `selection_bits` is the raw SelectionBitset pointer; ignored outside Edit mode.
std::vector<uint32_t> GetSampleOpVertices(
    const entt::registry &, entt::entity scene_entity, entt::entity sound_entity,
    const uint32_t *selection_bits
);

} // namespace scene_selection
