#include "selection/Selection.h"
#include "armature/ArmatureComponents.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "render/Instance.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "selection/SelectionBitset.h"
#include "selection/SelectionComponents.h"
#include "viewport/InteractionComponents.h"

#include "state/Scene.h"

bool IsBoneEditMode(const state::Scene &r, state::Entity viewport) {
    if (r.get<const Interaction>(viewport).Mode != InteractionMode::Edit) return false;
    return FindArmatureObject(r, FindActiveEntity(r)) != state::Null;
}

std::vector<state::Entity> RootSelectedForTransform(const state::Scene &r, state::Entity viewport) {
    const auto mode = r.get<const Interaction>(viewport).Mode;
    const auto arm_obj = FindArmatureObject(r, FindActiveEntity(r));
    const bool bone_edit_mode = mode == InteractionMode::Edit && arm_obj != state::Null;
    const bool bone_mode = bone_edit_mode || (mode == InteractionMode::Pose && arm_obj != state::Null);
    const auto is_parent_selected = [&](state::Entity e) {
        const auto *node = r.try_get<const SceneNode>(e);
        if (!node || node->Parent == state::Null) return false;
        return bone_mode ? r.all_of<BoneSelection>(node->Parent) : r.all_of<Selected>(node->Parent);
    };
    std::vector<state::Entity> root_selected;
    // Rest-pose edits do not propagate during a drag, so every selected edit-mode bone is a root.
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

bool CanDuplicate(const state::Scene &r, state::Entity viewport) {
    if (r.get<const Interaction>(viewport).Mode == InteractionMode::Pose) return false;
    if (IsBoneEditMode(r, viewport)) return !r.view<BoneSelection>().empty();
    return !r.view<Selected>().empty();
}
bool CanDuplicateLinked(const state::Scene &r, state::Entity viewport) { return CanDuplicate(r, viewport) && !IsBoneEditMode(r, viewport); }
bool CanDelete(const state::Scene &r, state::Entity viewport) { return CanDuplicate(r, viewport); }

bool AllSelectedAreMeshes(const state::Scene &r) {
    for (const auto [e, ok] : r.view<const Selected, const ObjectKind>().each()) {
        if (ok.Value != ObjectType::Mesh) return false;
    }
    return true;
}

std::vector<ElementRange> GetElementRangesForSelected(const state::Scene &r, state::Entity viewport) {
    const auto element = r.get<const EditMode>(viewport).Value;
    const auto &meshes = r.ctx().get<const MeshStore>();
    std::vector<ElementRange> ranges;
    for (const auto mesh_entity : selection::GetSelectedMeshEntities(r)) {
        if (!r.all_of<MeshElementSelection>(mesh_entity)) continue;
        const auto mesh = GetMesh(r, mesh_entity);
        if (const auto count = selection::GetElementCount(mesh, element); count > 0) {
            ranges.emplace_back(mesh_entity, meshes.GetSelectionBitOffset(mesh.GetStoreId(), element), count);
        }
    }
    return ranges;
}
