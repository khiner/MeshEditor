#include "selection/Selection.h"

#include "armature/ArmatureComponents.h"
#include "mesh/MeshStore.h"
#include "render/Instance.h"
#include "scene/Entity.h"
#include "scene/WorldTransform.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportInteractionState.h"

#include "state/Scene.h"

using std::ranges::find;

namespace selection {
bool HasScaleLockedInstance(const state::Scene &r, state::Entity e) {
    for (const auto [_, instance] : r.view<const Instance, const ScaleLocked>().each()) {
        if (instance.Entity == e) return true;
    }
    return false;
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

vec3 EditSelectionCenter(const state::Scene &r, state::Entity viewport) {
    const auto element = r.get<const EditMode>(viewport).Value;
    vec3 center{};
    uint32_t vertex_count = 0;
    for (const auto &[mesh_entity, instance_entity] : r.get<const EditPrimaries>(viewport).Transformable) {
        const auto *stats = GetElementSelectionSummary(r, mesh_entity, element);
        if (!stats || stats->SelectedVertexCount == 0) continue;
        const auto &world = *WorldTransformOf(r, instance_entity);
        center += float(stats->SelectedVertexCount) * world.P + Rotate(world.R, world.S * stats->PositionSum);
        vertex_count += stats->SelectedVertexCount;
    }
    return vertex_count > 0 ? center / float(vertex_count) : center;
}

StartPivot TransformPivot(const state::Scene &r, state::Entity viewport) {
    const auto &roots = r.get<const TransformRoots>(viewport);
    // Selection, position edits and history restore publish the persistent element aggregates the edit center reads.
    const bool mesh_edit = r.get<const Interaction>(viewport).Mode == InteractionMode::Edit && !IsBoneEditMode(r, viewport);
    return {mesh_edit ? EditSelectionCenter(r, viewport) : roots.PivotPosition, roots.PivotRotation};
}

bool CanDuplicate(const state::Scene &r, state::Entity viewport) {
    if (r.get<const Interaction>(viewport).Mode == InteractionMode::Pose) return false;
    if (IsBoneEditMode(r, viewport)) return !r.view<BoneSelection>().empty();
    return !r.view<Selected>().empty();
}
bool CanDuplicateLinked(const state::Scene &r, state::Entity viewport) { return CanDuplicate(r, viewport) && !IsBoneEditMode(r, viewport); }
bool CanDelete(const state::Scene &r, state::Entity viewport) { return CanDuplicate(r, viewport); }

std::vector<ElementRange> GetElementRangesForSelected(const state::Scene &r, state::Entity viewport) {
    const auto element = r.get<const EditMode>(viewport).Value;
    const auto &meshes = r.Context.get<const MeshStore>();
    std::vector<ElementRange> ranges;
    for (const auto mesh_entity : r.get<const SelectionFlags>(viewport).Meshes) {
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
    std::vector<uint64_t> seen((r.EntityCapacity() + 63) / 64);
    // Marks the target seen and returns whether it was new.
    const auto first = [&](state::Entity target) {
        const auto index = state::Index(target);
        const auto bit = uint64_t{1} << index % 64;
        if (seen[index / 64] & bit) return false;
        seen[index / 64] |= bit;
        return true;
    };
    const auto merge = [&](state::Entity target) {
        if (merge_parts) find(hits, target, &SelectionHit::Entity)->Part = {};
    };
    for (const auto e : raw) {
        if (bone_mode && r.all_of<BoneIndex>(e)) {
            if (first(e)) hits.emplace_back(e, BoneSel::Body);
            else merge(e);
        } else if (bone_mode && r.all_of<BoneSubPartOf>(e)) {
            const auto &sub = r.get<BoneSubPartOf>(e);
            if (first(sub.BoneEntity)) hits.emplace_back(sub.BoneEntity, sub.IsTip ? BoneSel::Tip : BoneSel::Root);
            else merge(sub.BoneEntity);
        } else if (!bone_mode) {
            if (const auto target = r.all_of<SubElementOf>(e) ? r.get<SubElementOf>(e).Parent : e; first(target)) hits.emplace_back(target);
        }
    }
    return hits;
}
