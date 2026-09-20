#include "selection/Selection.h"

#include "armature/ArmatureComponents.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "scene/WorldTransform.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportInteractionState.h"

#include "state/Scene.h"

using std::ranges::contains, std::ranges::find;

namespace selection {
namespace {
void ForEachEditInstance(const state::Scene &r, auto &&f) {
    const auto active = FindActiveEntity(r);
    for (const auto [e, instance, ok, ri] : r.view<const Instance, const Selected, const ObjectKind, const RenderInstance>().each()) {
        if (ok.Value == ObjectType::Mesh) f(instance.Entity, e, active, r.all_of<ScaleLocked>(e));
    }
}

void ChoosePrimary(PrimaryEditInstanceMap &primaries, state::Entity mesh, state::Entity instance, state::Entity active) {
    auto &primary = primaries.try_emplace(mesh, instance).first->second;
    if (instance == active || (primary != active && instance < primary)) primary = instance;
}
} // namespace

PrimaryEditInstanceMap ComputePrimaryEditInstances(const state::Scene &r, bool include_scale_locked) {
    PrimaryEditInstanceMap primaries;
    ForEachEditInstance(r, [&](state::Entity mesh, state::Entity instance, state::Entity active, bool scale_locked) {
        if (include_scale_locked || !scale_locked) ChoosePrimary(primaries, mesh, instance, active);
    });
    return primaries;
}

PrimaryEditInstanceMaps ComputePrimaryEditInstanceMaps(const state::Scene &r) {
    PrimaryEditInstanceMaps result;
    ForEachEditInstance(r, [&](state::Entity mesh, state::Entity instance, state::Entity active, bool scale_locked) {
        ChoosePrimary(result.All, mesh, instance, active);
        if (!scale_locked) ChoosePrimary(result.Transformable, mesh, instance, active);
    });
    return result;
}

bool HasScaleLockedInstance(const state::Scene &r, state::Entity e) {
    for (const auto [_, instance] : r.view<const Instance, const ScaleLocked>().each()) {
        if (instance.Entity == e) return true;
    }
    return false;
}

std::unordered_set<state::Entity> GetSelectedMeshEntities(const state::Scene &r) {
    std::unordered_set<state::Entity> entities;
    for (const auto [e, instance] : r.view<const Instance, const Selected>().each()) {
        if (HasMesh(r, instance.Entity)) entities.emplace(instance.Entity);
    }
    return entities;
}
} // namespace selection

state::Entity FindArmatureObject(const state::Scene &r, state::Entity e) {
    if (e == state::Null) return state::Null;
    if (r.all_of<ArmatureObject>(e)) return e;
    if (const auto *sub = r.try_get<SubElementOf>(e); sub && r.all_of<ArmatureObject>(sub->Parent)) return sub->Parent;
    return state::Null;
}

state::Entity FindActiveBone(const state::Scene &r) {
    state::Entity result = state::Null;
    for (const auto e : r.view<BoneActive>()) {
        assert(result == state::Null && "Multiple BoneActive entities");
        result = e;
    }
    return result;
}

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

vec3 EditSelectionCenter(const state::Scene &r, Element edit_mode) {
    vec3 center{};
    uint32_t vertex_count = 0;
    for (const auto &[mesh_entity, instance_entity] : selection::ComputePrimaryEditInstances(r, false)) {
        const auto *stats = GetElementSelectionSummary(r, mesh_entity, edit_mode);
        if (!stats || stats->SelectedVertexCount == 0) continue;
        const auto &world = r.get<const WorldTransform>(instance_entity);
        center += float(stats->SelectedVertexCount) * world.P + Rotate(world.R, world.S * stats->PositionSum);
        vertex_count += stats->SelectedVertexCount;
    }
    return vertex_count > 0 ? center / float(vertex_count) : center;
}

StartPivot TransformPivot(state::Scene &r, state::Entity viewport) {
    const auto mode = r.get<const Interaction>(viewport).Mode;
    const auto active_entity = FindActiveEntity(r);
    const bool bone_edit_mode = IsBoneEditMode(r, viewport);
    const bool bone_mode = bone_edit_mode || (mode == InteractionMode::Pose && FindArmatureObject(r, active_entity) != state::Null);
    const auto active = bone_mode ? FindActiveBone(r) : active_entity;
    StartPivot pivot{.R = active != state::Null ? r.get<const WorldTransform>(active).R : quat{1, 0, 0, 0}};
    if (mode == InteractionMode::Edit && !bone_edit_mode) {
        const auto element = r.get<const EditMode>(viewport).Value;
        // Derive the summary now, since a restore leaves it stale until the next render.
        if (r.Context.get<const GpuSceneState>().EditSelectionDirty) ApplyEditSelectionCommand(r, GetElementRangesForSelected(r, viewport), element, EditSelectionOperation::Derive);
        pivot.P = EditSelectionCenter(r, element);
        return pivot;
    }
    vec3 sum{};
    uint32_t count = 0;
    // A bone contributes its head for a selected root and its tail for a selected tip, so a whole bone contributes its midpoint.
    for (const auto e : RootSelectedForTransform(r, viewport)) {
        const auto &wt = r.get<const WorldTransform>(e);
        const auto *parts = bone_edit_mode ? r.try_get<const BoneSelection>(e) : nullptr;
        if (!parts || parts->Root) {
            sum += wt.P;
            ++count;
        }
        if (parts && parts->Tip) {
            sum += wt.P + Rotate(wt.R, vec3{0, r.get<const BoneDisplayScale>(e).Value, 0});
            ++count;
        }
    }
    pivot.P = count > 0 ? sum / float(count) : vec3{};
    return pivot;
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
    const auto &meshes = r.Context.get<const MeshStore>();
    std::vector<ElementRange> ranges;
    for (const auto mesh_entity : selection::GetSelectedMeshEntities(r)) {
        if (!r.all_of<MeshElementSelection>(mesh_entity)) continue;
        const auto mesh = GetMesh(r, mesh_entity);
        if (const auto count = mesh.ElementCount(element); count > 0) {
            ranges.emplace_back(mesh_entity, meshes.GetSelectionBitOffset(mesh.GetStoreId(), element), count);
        }
    }
    return ranges;
}

void Select(state::Scene &r, state::Entity e) {
    r.clear<Selected>();
    if (e != state::Null) {
        r.clear<Active>();
        r.emplace<Active>(e);
        r.emplace<Selected>(e);
    }
}

void SelectBone(state::Scene &r, state::Entity e) {
    r.clear<BoneSelection>();
    if (e != state::Null) {
        r.clear<BoneActive>();
        r.emplace<BoneActive>(e);
        r.emplace<BoneSelection>(e);
    }
}

std::vector<SelectionHit> ResolveHits(state::Scene &r, const std::vector<state::Entity> &raw, bool bone_mode, bool merge_parts) {
    std::vector<SelectionHit> hits;
    for (const auto e : raw) {
        if (bone_mode && r.all_of<BoneIndex>(e)) {
            if (auto it = find(hits, e, &SelectionHit::Entity); it == hits.end()) hits.emplace_back(e, BoneSel::Body);
            else if (merge_parts) it->Part = {};
        } else if (bone_mode && r.all_of<BoneSubPartOf>(e)) {
            const auto &sub = r.get<BoneSubPartOf>(e);
            if (auto it = find(hits, sub.BoneEntity, &SelectionHit::Entity); it == hits.end()) hits.emplace_back(sub.BoneEntity, sub.IsTip ? BoneSel::Tip : BoneSel::Root);
            else if (merge_parts) it->Part = {};
        } else if (!bone_mode) {
            if (const auto target = r.all_of<SubElementOf>(e) ? r.get<SubElementOf>(e).Parent : e; !contains(hits, target, &SelectionHit::Entity)) hits.emplace_back(target);
        }
    }
    return hits;
}
