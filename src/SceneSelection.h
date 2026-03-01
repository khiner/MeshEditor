#pragma once

#include "entt_fwd.h"
#include "gpu/Element.h"
#include "scene_impl/SceneInternalTypes.h"

#include <cstdint>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Mesh;

namespace scene_selection {

// Count selected bits in [offset, offset+count).
uint32_t CountSelected(const uint32_t *bits, uint32_t offset, uint32_t count);
// Return local (0-based) handles of all set bits in [offset, offset+count).
std::vector<uint32_t> ScanBitsetRange(const uint32_t *bits, uint32_t offset, uint32_t count);
// Convert selected elements from one type to another by reading `from_range` bits,
// converting topology, and writing into `to_range`. Both ranges are in the same bitset.
void ConvertSelectionBitset(uint32_t *bits, MeshSelectionBitsetRange from_range, MeshSelectionBitsetRange to_range, const Mesh &, Element from_element, Element to_element);
// Get the set of vertex handles for the selection in the bitset (converts from any element type).
std::unordered_set<uint32_t> GetSelectedVertices(const uint32_t *bits, MeshSelectionBitsetRange range, const Mesh &, Element element);

std::unordered_set<uint32_t> ConvertSelectionElement(std::span<const uint32_t> handles, const Mesh &, Element from_element, Element to_element);

// Returns representative edit instance per selected mesh: active instance if selected, else first selected instance.
// Only includes Mesh-type objects (excludes Cameras, etc. that also have MeshInstance).
std::unordered_map<entt::entity, entt::entity> ComputePrimaryEditInstances(const entt::registry &, bool include_frozen = true);

bool HasFrozenInstance(const entt::registry &, entt::entity mesh_entity);
std::unordered_set<entt::entity> GetSelectedMeshEntities(const entt::registry &);
uint32_t GetElementCount(const Mesh &, Element);

} // namespace scene_selection
