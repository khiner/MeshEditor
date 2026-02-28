#pragma once

#include "entt_fwd.h"
#include "gpu/Element.h"

#include <unordered_map>
#include <unordered_set>

struct Mesh;
struct MeshSelection;

namespace scene_selection {

std::unordered_set<uint32_t> ConvertSelectionElement(const MeshSelection &, const Mesh &, Element from_element, Element to_element);

// Returns representative edit instance per selected mesh: active instance if selected, else first selected instance.
// Only includes Mesh-type objects (excludes Cameras, etc. that also have MeshInstance).
std::unordered_map<entt::entity, entt::entity> ComputePrimaryEditInstances(const entt::registry &, bool include_frozen = true);

bool HasFrozenInstance(const entt::registry &, entt::entity mesh_entity);
std::unordered_set<entt::entity> GetSelectedMeshEntities(const entt::registry &);
uint32_t GetElementCount(const Mesh &, Element);

} // namespace scene_selection
