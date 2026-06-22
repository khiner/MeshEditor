#include "selection/Selection.h"
#include "armature/ArmatureComponents.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "render/Instance.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "selection/SelectionBitset.h"
#include "selection/SelectionComponents.h"
#include "viewport/InteractionComponents.h"

#include <entt/entity/registry.hpp>

namespace selection {

std::unordered_map<entt::entity, entt::entity> ComputePrimaryEditInstances(const entt::registry &r, bool include_scale_locked) {
    std::unordered_map<entt::entity, entt::entity> primaries;
    const auto active = FindActiveEntity(r);
    for (const auto [e, instance, ok, ri] : r.view<const Instance, const Selected, const ObjectKind, const RenderInstance>().each()) {
        if (ok.Value != ObjectType::Mesh) continue;
        if (!include_scale_locked && r.all_of<ScaleLocked>(e)) continue;
        auto &primary = primaries[instance.Entity];
        if (primary == entt::entity{} || e == active) primary = e;
    }
    return primaries;
}

bool HasScaleLockedInstance(const entt::registry &r, entt::entity e) {
    for (const auto [_, instance] : r.view<const Instance, const ScaleLocked>().each()) {
        if (instance.Entity == e) return true;
    }
    return false;
}

std::unordered_set<entt::entity> GetSelectedMeshEntities(const entt::registry &r) {
    std::unordered_set<entt::entity> entities;
    for (const auto [e, instance] : r.view<const Instance, const Selected>().each()) {
        if (HasMesh(r, instance.Entity)) entities.emplace(instance.Entity);
    }
    return entities;
}

} // namespace selection

entt::entity FindArmatureObject(const entt::registry &r, entt::entity e) {
    if (e == entt::null) return entt::null;
    if (r.all_of<ArmatureObject>(e)) return e;
    if (const auto *sub = r.try_get<SubElementOf>(e); sub && r.all_of<ArmatureObject>(sub->Parent)) return sub->Parent;
    return entt::null;
}

entt::entity FindActiveBone(const entt::registry &r) {
    entt::entity result = entt::null;
    for (const auto e : r.view<BoneActive>()) {
        assert(result == entt::null && "Multiple BoneActive entities");
        result = e;
    }
    return result;
}

bool IsBoneEditMode(const entt::registry &r, entt::entity viewport) {
    if (r.get<const Interaction>(viewport).Mode != InteractionMode::Edit) return false;
    return FindArmatureObject(r, FindActiveEntity(r)) != entt::null;
}

std::vector<entt::entity> RootSelectedForTransform(const entt::registry &r, entt::entity viewport) {
    const auto mode = r.get<const Interaction>(viewport).Mode;
    const auto arm_obj = FindArmatureObject(r, FindActiveEntity(r));
    const bool bone_edit_mode = mode == InteractionMode::Edit && arm_obj != entt::null;
    const bool bone_mode = bone_edit_mode || (mode == InteractionMode::Pose && arm_obj != entt::null);
    const auto is_parent_selected = [&](entt::entity e) {
        const auto *node = r.try_get<const SceneNode>(e);
        if (!node || node->Parent == entt::null) return false;
        return bone_mode ? r.all_of<BoneSelection>(node->Parent) : r.all_of<Selected>(node->Parent);
    };
    std::vector<entt::entity> root_selected;
    // Edit mode: all selected bones are roots (rest-pose edits don't propagate during drag).
    // Pose/object mode: drop children whose parent is also selected (the parent's transform propagates).
    if (bone_edit_mode) {
        for (const auto e : r.view<const BoneSelection>()) root_selected.emplace_back(e);
    } else if (bone_mode) {
        for (const auto e : r.view<const BoneSelection>())
            if (!is_parent_selected(e)) root_selected.emplace_back(e);
    } else {
        for (const auto e : r.view<const Selected>())
            if (!is_parent_selected(e)) root_selected.emplace_back(e);
    }
    return root_selected;
}

bool CanDuplicate(const entt::registry &r, entt::entity viewport) {
    if (r.get<const Interaction>(viewport).Mode == InteractionMode::Pose) return false;
    if (IsBoneEditMode(r, viewport)) return !r.view<BoneSelection>().empty();
    return !r.view<Selected>().empty();
}
bool CanDuplicateLinked(const entt::registry &r, entt::entity viewport) { return CanDuplicate(r, viewport) && !IsBoneEditMode(r, viewport); }
bool CanDelete(const entt::registry &r, entt::entity viewport) { return CanDuplicate(r, viewport); }

bool AllSelectedAreMeshes(const entt::registry &r) {
    for (const auto [e, ok] : r.view<const Selected, const ObjectKind>().each()) {
        if (ok.Value != ObjectType::Mesh) return false;
    }
    return true;
}

std::vector<ElementRange> GetBitsetRangesForSelected(const entt::registry &r) {
    std::vector<ElementRange> ranges;
    for (const auto mesh_entity : selection::GetSelectedMeshEntities(r)) {
        if (const auto *br = r.try_get<const MeshSelectionBitsetRange>(mesh_entity); br && br->Count > 0) {
            ranges.emplace_back(mesh_entity, br->Offset, br->Count);
        }
    }
    return ranges;
}
