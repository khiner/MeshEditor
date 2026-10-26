#include "selection/Selection.h"

#include "armature/ArmatureComponents.h"
#include "mesh/Mesh.h"
#include "render/Instance.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportInteractionState.h"

#include "state/Registry.h"

using std::ranges::find;

std::vector<state::Entity> SelectedIn(const state::Registry &r) {
    const auto scene = FrameScene(r);
    return r.view<const Selected>() | std::views::filter([&](state::Entity e) { return InScene(r, e, scene); }) | std::ranges::to<std::vector>();
}

state::Entity FindArmatureObject(const state::Registry &r, state::Entity e) {
    if (e == state::Null) return state::Null;
    if (r.all_of<ArmatureObject>(e)) return e;
    if (const auto *sub = r.try_get<SubElementOf>(e); sub && r.all_of<ArmatureObject>(sub->Parent)) return sub->Parent;
    return state::Null;
}

state::Entity FindActiveBone(const state::Registry &r) {
    state::Entity result = state::Null;
    for (const auto e : r.view<BoneActive>()) {
        assert(result == state::Null && "Multiple BoneActive entities");
        result = e;
    }
    return result;
}

bool IsBoneEditMode(const state::Registry &r) {
    if (r.get<const Interaction>(state::Document).Mode != InteractionMode::Edit) return false;
    return FindArmatureObject(r, FindActiveEntity(r)) != state::Null;
}

vec3 EditSelectionCenter(const state::Registry &r) {
    const auto element = r.get<const EditMode>(state::Document).Value;
    vec3 center{};
    uint32_t vertex_count = 0;
    for (const auto &[mesh_entity, instance_entity] : r.get<const EditPrimaries>(state::Document).Transformable) {
        const auto *stats = GetElementSelectionSummary(r, mesh_entity, element);
        if (!stats || stats->SelectedVertexCount == 0) continue;
        const auto &world = *WorldTransformOf(r, instance_entity);
        center += float(stats->SelectedVertexCount) * world.P + Rotate(world.R, world.S * stats->PositionSum);
        vertex_count += stats->SelectedVertexCount;
    }
    return vertex_count > 0 ? center / float(vertex_count) : center;
}

StartPivot TransformPivot(const state::Registry &r) {
    const auto &roots = r.get<const TransformRoots>(state::Document);
    // Selection, position edits and history restore publish the persistent element aggregates the edit center reads.
    const bool mesh_edit = r.get<const Interaction>(state::Document).Mode == InteractionMode::Edit && !IsBoneEditMode(r);
    return {mesh_edit ? EditSelectionCenter(r) : roots.PivotPosition, roots.PivotRotation};
}

bool CanDuplicate(const state::Registry &r) {
    if (r.get<const Interaction>(state::Document).Mode == InteractionMode::Pose) return false;
    if (IsBoneEditMode(r)) return !r.view<BoneSelection>().empty();
    return !SelectedIn(r).empty();
}
bool CanDuplicateLinked(const state::Registry &r) { return CanDuplicate(r) && !IsBoneEditMode(r); }
bool CanDelete(const state::Registry &r) { return CanDuplicate(r); }

std::vector<ElementRange> GetElementRangesForSelected(const state::Registry &r) {
    const auto element = r.get<const EditMode>(state::Document).Value;
    std::vector<ElementRange> ranges;
    for (const auto mesh_entity : r.get<const SelectionFlags>(state::Document).Meshes) {
        if (!r.all_of<MeshElementSelection>(mesh_entity)) continue;
        const auto mesh = GetMesh(r, mesh_entity);
        if (const auto count = mesh.ElementCount(element); count > 0) {
            ranges.emplace_back(mesh_entity, count);
        }
    }
    return ranges;
}

void Select(state::Registry &r, state::Entity e) {
    r.clear<Selected>();
    if (e != state::Null) {
        r.clear<Active>();
        r.emplace<Active>(e);
        r.emplace<Selected>(e);
    }
}

void SelectBone(state::Registry &r, state::Entity e) {
    r.clear<BoneSelection>();
    if (e != state::Null) {
        r.clear<BoneActive>();
        r.emplace<BoneActive>(e);
        r.emplace<BoneSelection>(e);
    }
}

std::vector<SelectionHit> ResolveHits(state::Registry &r, std::span<const state::Entity> raw, bool bone_mode, bool merge_parts) {
    std::vector<SelectionHit> hits;
    std::vector<uint64_t> seen((r.EntityCapacity() + 63) / 64);
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
