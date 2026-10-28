#pragma once

#include "gpu/Element.h"
#include "numeric/vec3.h"
#include "selection/BoneSelection.h"

#include "state/Entity.h"

#include <span>
#include <variant>
#include <vector>

using numeric::vec3;

struct ElementRange;
struct StartPivot;

// The selected nodes in the frame's scene, in entity index order.
std::vector<state::Entity> SelectedIn(const state::Registry &);

bool IsBoneEditMode(const state::Registry &);
bool CanDuplicate(const state::Registry &);
bool CanDuplicateLinked(const state::Registry &);
bool CanDelete(const state::Registry &);
std::vector<ElementRange> GetElementRangesForSelected(const state::Registry &, std::optional<Element> = {});

// Returns the containing armature, or null if the entity is unrelated to an armature.
state::Entity FindArmatureObject(const state::Registry &, state::Entity);
// Returns null if no bone is active.
state::Entity FindActiveBone(const state::Registry &);

// The mean world position of the selected vertices of each mesh's transformable primary instance.
vec3 EditSelectionCenter(const state::Registry &);
// The pivot a transform of the current selection turns and scales about: the selection's center, with the active target's rotation.
StartPivot TransformPivot(const state::Registry &);

// Exclusive select: clears Selected/Active, then selects `e` (null clears everything).
void Select(state::Registry &, state::Entity);
// Exclusive bone select: clears BoneSelection/BoneActive, then selects `e` (null clears everything).
void SelectBone(state::Registry &, state::Entity);

struct ObjectKind;
struct BoneIndex;
struct ObjectHit {
    state::EntityRef<ObjectKind> Entity;
    bool operator==(const ObjectHit &) const = default;
};
struct BoneHit {
    state::EntityRef<BoneIndex> Entity;
    BoneSel Part{BoneSel::Body};
    bool operator==(const BoneHit &) const = default;
};
using SelectionHit = std::variant<ObjectHit, BoneHit>;
inline state::Entity HitEntity(const SelectionHit &hit) {
    return std::visit([](const auto &h) { return h.Entity.Value; }, hit);
}
inline std::optional<BoneSel> HitPart(const SelectionHit &hit) {
    if (const auto *bone = std::get_if<BoneHit>(&hit)) return bone->Part;
    return {};
}

struct ElementHit {
    state::Entity MeshEntity{state::Null};
    uint32_t Handle{};
};

// Resolves raw GPU hits to logical targets, collapsing bone parts and object sub-elements.
std::vector<SelectionHit> ResolveHits(state::Registry &, std::span<const state::Entity> raw, bool bone_mode, bool merge_parts = false);

namespace selection {
using PrimaryEditInstanceMap = std::unordered_map<state::Entity, state::Entity>;
} // namespace selection
