#pragma once

#include "gpu/Element.h"
#include "numeric/vec3.h"
#include "selection/BoneSelection.h"

#include "state/Entity.h"

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct ElementRange;
struct StartPivot;

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
vec3 EditSelectionCenter(const state::Scene &, Element edit_mode);
// The pivot a transform of the current selection turns and scales about: the selection's center, with the active target's rotation.
StartPivot TransformPivot(state::Scene &, state::Entity viewport);

// Exclusive select: clears Selected/Active, then selects `e` (null clears everything).
void Select(state::Scene &, state::Entity);
// Exclusive bone select: clears BoneSelection/BoneActive, then selects `e` (null clears everything).
void SelectBone(state::Scene &, state::Entity);

struct SelectionHit {
    state::Entity Entity;
    std::optional<BoneSel> Part{};
    bool operator==(const SelectionHit &) const = default;
};

// Resolves raw GPU hits to logical targets, collapsing bone parts and object sub-elements.
std::vector<SelectionHit> ResolveHits(state::Scene &, const std::vector<state::Entity> &raw, bool bone_mode, bool merge_parts = false);

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
