#pragma once

#include "state/Entity.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

struct ElementRange;

bool AllSelectedAreMeshes(const state::Scene &);
bool IsBoneEditMode(const state::Scene &, state::Entity viewport);
bool CanDuplicate(const state::Scene &, state::Entity viewport);
bool CanDuplicateLinked(const state::Scene &, state::Entity viewport);
bool CanDelete(const state::Scene &, state::Entity viewport);
std::vector<ElementRange> GetElementRangesForSelected(const state::Scene &, state::Entity viewport);

// Returns the containing armature, or null if the entity is unrelated to an armature.
state::Entity FindArmatureObject(const state::Scene &, state::Entity);
// Returns null if no bone is active.
state::Entity FindActiveBone(const state::Scene &);

// Returns selected transform roots, using bones in pose or edit mode and objects otherwise.
std::vector<state::Entity> RootSelectedForTransform(const state::Scene &, state::Entity viewport);

struct EditTransformContext {
    std::unordered_map<state::Entity, state::Entity> TransformInstances;
};

namespace selection {
using PrimaryEditInstanceMap = std::unordered_map<state::Entity, state::Entity>;
struct PrimaryEditInstanceMaps {
    PrimaryEditInstanceMap All, Transformable;
};

// Returns one selected mesh instance per mesh, preferring the active instance, then the lowest entity ID.
PrimaryEditInstanceMap ComputePrimaryEditInstances(const state::Scene &, bool include_scale_locked = true);
PrimaryEditInstanceMaps ComputePrimaryEditInstanceMaps(const state::Scene &);

bool HasScaleLockedInstance(const state::Scene &, state::Entity);
std::unordered_set<state::Entity> GetSelectedMeshEntities(const state::Scene &);
} // namespace selection
