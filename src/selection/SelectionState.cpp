#include "selection/SelectionState.h"

#include "SortUnique.h"
#include "animation/AnimationTimeline.h"
#include "animation/Clips.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "selection/SelectionGpu.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportInteractionState.h"

using state::Change;

ModeScope ScopeOf(const state::Scene &r, state::Entity viewport) {
    const auto mode = r.get<const Interaction>(viewport).Mode;
    const bool armature = FindArmatureObject(r, FindActiveEntity(r)) != state::Null;
    const bool bone_edit = mode == InteractionMode::Edit && armature;
    return {mode, bone_edit, bone_edit || (mode == InteractionMode::Pose && armature), mode == InteractionMode::Edit && !bone_edit};
}

namespace {
void ComputeRoots(const state::Scene &r, const ModeScope &scope, TransformRoots &out) {
    // Views run in entity index order, so the candidates are searchable by index.
    auto selected = scope.Bone ? r.view<const BoneSelection>() | std::ranges::to<std::vector>() : r.view<const Selected>() | std::ranges::to<std::vector>();
    if (scope.BoneEdit) {
        out.Roots = std::move(selected);
        return;
    }
    out.Roots.clear();
    for (const auto e : selected) {
        auto parent = ParentOrNull(r, e);
        while (parent != state::Null && !ContainsInIndexOrder(selected, parent)) parent = ParentOrNull(r, parent);
        if (parent == state::Null) out.Roots.push_back(e);
    }
}

void ComputePivot(const state::Scene &r, const ModeScope &scope, TransformRoots &out) {
    const auto active = scope.Bone ? FindActiveBone(r) : FindActiveEntity(r);
    const auto *active_world = active != state::Null ? WorldTransformOf(r, active) : nullptr;
    out.PivotRotation = active_world ? active_world->R : quat{1, 0, 0, 0};
    vec3 sum{};
    uint32_t count = 0;
    // A bone contributes its head for a selected root and its tail for a selected tip, so a whole bone contributes its midpoint.
    // A selected entity outside the scene graph, like a mesh data entity, has no position to contribute.
    for (const auto e : out.Roots) {
        const auto *world = WorldTransformOf(r, e);
        if (!world) continue;
        const auto *parts = scope.BoneEdit ? r.try_get<const BoneSelection>(e) : nullptr;
        if (!parts || parts->Root) {
            sum += world->P;
            ++count;
        }
        if (parts && parts->Tip) {
            sum += world->P + Rotate(world->R, vec3{0, r.get<const BoneDisplayScale>(e).Value, 0});
            ++count;
        }
    }
    out.PivotPosition = count > 0 ? sum / float(count) : vec3{};
}

// Whether a world transform the pivot reads changed this settle.
bool PivotMoved(const state::Scene &r, const ModeScope &scope, const TransformRoots &roots) {
    const auto &moved = reactive(r, Change::WorldTransform);
    if (moved.empty()) return false;
    if (const auto active = scope.Bone ? FindActiveBone(r) : FindActiveEntity(r); active != state::Null && moved.contains(active)) return true;
    if (moved.size() < roots.Roots.size()) return std::ranges::any_of(moved, [&](state::Entity e) { return ContainsInIndexOrder(roots.Roots, e); });
    return std::ranges::any_of(roots.Roots, [&](state::Entity e) { return moved.contains(e); });
}

SelectionSharpness Sharpness(bool any_smooth, bool any_sharp, bool mixed) {
    if (mixed || (any_smooth && any_sharp)) return SelectionSharpness::Mixed;
    if (any_smooth) return SelectionSharpness::Smooth;
    return any_sharp ? SelectionSharpness::Sharp : SelectionSharpness::None;
}

void ComputeFlags(const state::Scene &r, state::Entity viewport, const ModeScope &scope, const EditPrimaries &primaries, SelectionFlags &out) {
    const auto &meshes = r.Context.get<const MeshStore>();
    out.Meshes.clear();
    out.AllMeshes = std::ranges::none_of(r.view<const Selected, const ObjectKind>().each(), [](const auto &entry) { return std::get<1>(entry).Value != ObjectType::Mesh; });
    out.AnyVisible = out.AnyHidden = false;
    for (const auto [e, instance] : r.view<const Selected, const Instance>().each()) {
        if (HasMesh(r, instance.Entity)) out.Meshes.push_back(instance.Entity);
        if (r.all_of<SubElementOf>(e)) continue;
        if (r.all_of<Hidden>(e)) out.AnyHidden = true;
        else out.AnyVisible = true;
    }
    SortUnique(out.Meshes);

    const bool edit_locked = scope.Mode == InteractionMode::Edit && std::ranges::any_of(r.view<const Instance, const ScaleLocked>().each(), [&](const auto &entry) { return std::ranges::binary_search(out.Meshes, std::get<1>(entry).Entity); });
    out.Transformable = !edit_locked;
    out.Scalable = !edit_locked && r.view<const Selected, const ScaleLocked>().empty();

    // A fully smooth mesh has no sharp faces, while partial sharpness is mixed.
    // Every live face marks its mesh's face root sharp or smooth, so a root with neither belongs to a mesh without faces.
    bool any_smooth = false, any_sharp = false, any_partial = false;
    for (const auto mesh_entity : out.Meshes) {
        const auto faces = meshes.GetSelectionRoot(r.get<const MeshHandle>(mesh_entity).StoreId, Element::Face).Flags;
        const bool sharp = faces & SelectionLiveSharp, smooth = faces & SelectionLiveSmooth;
        if (!sharp && !smooth) continue;
        any_smooth |= !sharp;
        any_sharp |= sharp;
        any_partial |= sharp && smooth;
        if ((any_smooth && any_sharp) || any_partial) break;
    }
    out.FaceSharpness = Sharpness(any_smooth, any_sharp, any_partial);

    const auto element = r.get<const EditMode>(viewport).Value;
    bool element_sharp = false, element_smooth = false;
    if (scope.MeshEdit) {
        for (const auto &[mesh_entity, _] : primaries.All) {
            const auto *summary = GetElementSelectionSummary(r, mesh_entity, element);
            if (!summary) continue;
            element_sharp |= (summary->SharpnessFlags & 1u) != 0u;
            element_smooth |= (summary->SharpnessFlags & 2u) != 0u;
            if (element_sharp && element_smooth) break;
        }
    }
    out.ElementSharpness = Sharpness(element_smooth, element_sharp, false);

    out.HasTransformTarget = [&] {
        if (scope.Bone) return !r.view<const BoneSelection>().empty();
        if (r.view<const Selected>().empty()) return false;
        if (!scope.MeshEdit) return true;
        for (const auto [e, instance] : r.view<const Instance, const Selected>(state::Exclude<ScaleLocked>).each()) {
            const auto *stats = GetElementSelectionSummary(r, instance.Entity, element);
            if (stats && stats->SelectedCount > 0) return true;
        }
        return false;
    }();
}

void ComputeKeyframes(const state::Scene &r, state::Entity viewport, SelectedKeyframes &out) {
    const float fps = r.get<const TimelineRange>(viewport).Fps;
    auto &frames = out.Frames;
    frames.clear();
    const auto append = [&](state::Entity e, auto &&include) {
        const auto *clips = r.try_get<const AnimationClips>(e);
        const auto *clip = clips ? animation::ActiveClip(r, viewport, *clips) : nullptr;
        if (!clip) return;
        for (const auto &channel : clip->Channels)
            if (include(channel))
                for (const float t : channel.Times) frames.emplace_back(1.f + t * fps);
    };
    const auto all = [](const AnimationChannel &) { return true; };
    if (!r.view<const AnimationClips>().empty()) {
        for (const auto e : r.view<const Selected>()) {
            append(e, all);
            if (const auto *armature = r.try_get<const ArmatureObject>(e)) append(armature->Entity, all);
        }
    }
    // Material keys show for the material slot the active mesh displays.
    if (const auto active = FindActiveEntity(r); active != state::Null) {
        const auto *instance = r.try_get<const Instance>(active);
        if (const auto material = instance ? DisplayedMaterial(r, instance->Entity) : std::nullopt) append(viewport, [&](const AnimationChannel &channel) { return channel.Target.Component == state::Key<MaterialStore>() && channel.Target.Index == *material; });
    }
    SortUnique(frames);
}

void ComputeSelectedAncestors(const state::Scene &r, OutlinerRows &out) {
    out.SelectedAncestors.clear();
    const auto mark = [&](state::Entity selected) {
        // A marked ancestor already marked its own ancestors.
        for (auto parent = ParentOrNull(r, selected); parent != state::Null && out.SelectedAncestors.insert(parent).second;) parent = ParentOrNull(r, parent);
    };
    for (const auto e : r.view<const Selected>()) mark(e);
    for (const auto e : r.view<const BoneSelection>()) mark(e);
}
} // namespace

EditPrimaries ComputeEditPrimaries(const state::Scene &r) {
    const auto active = FindActiveEntity(r);
    const auto choose = [active](selection::PrimaryEditInstanceMap &primaries, state::Entity mesh, state::Entity instance) {
        auto &primary = primaries.try_emplace(mesh, instance).first->second;
        if (instance == active || (primary != active && instance < primary)) primary = instance;
    };
    EditPrimaries primaries;
    for (const auto [e, instance, kind, _] : r.view<const Instance, const Selected, const ObjectKind, const RenderInstance>(state::Exclude<Hidden>).each()) {
        if (kind.Value != ObjectType::Mesh) continue;
        choose(primaries.All, instance.Entity, e);
        if (!r.all_of<ScaleLocked>(e)) choose(primaries.Transformable, instance.Entity, e);
    }
    return primaries;
}

std::optional<uint32_t> OutlinerRows::Find(state::Entity e) const {
    const auto index = state::Index(e);
    if (index >= RowOf.size() || RowOf[index] >= Rows.size() || Rows[RowOf[index]].Entity != e) return {};
    return RowOf[index];
}

void BuildOutlinerRows(const state::Scene &r, OutlinerRows &out) {
    out.Rows.clear();
    out.RowOf.assign(r.EntityCapacity(), UINT32_MAX);
    const auto roots = SortedEntities(r.view<const Name>() | std::views::filter([&](state::Entity e) { return ParentOrNull(r, e) == state::Null; }));
    // Depth-first, so each open row's subtree follows it with its children in sibling order.
    std::vector<std::pair<state::Entity, uint32_t>> pending;
    std::vector<state::Entity> children;
    for (const auto root : roots | std::views::reverse) pending.emplace_back(root, 0u);
    while (!pending.empty()) {
        const auto [e, depth] = pending.back();
        pending.pop_back();
        const auto *links = r.try_get<const SceneChildren>(e);
        const bool has_children = links && links->FirstChild != state::Null;
        out.RowOf[state::Index(e)] = uint32_t(out.Rows.size());
        out.Rows.push_back({e, depth, has_children});
        if (!has_children || !out.Open.contains(e)) continue;
        children.assign_range(Children{&r, e});
        for (const auto child : children | std::views::reverse) pending.emplace_back(child, depth + 1u);
    }
}

void UpdateSelectionState(state::Scene &r, state::Entity viewport) {
    const bool selection = AnyChanged(r, Change::Selected, Change::ActiveInstance, Change::BoneSelection);
    const bool interaction = AnyChanged(r, Change::InteractionMode);
    const bool parents = AnyChanged(r, Change::SceneParent);
    const bool instances = AnyChanged(r, Change::InstanceVisibility, Change::RenderInstanceCreated, Change::RenderInstanceDestroyed, Change::ScaleLocked);
    const auto scope = ScopeOf(r, viewport);

    const bool primaries_changed = selection || interaction || instances;
    if (primaries_changed) {
        auto &primaries = r.edit<EditPrimaries>(viewport);
        if (scope.Mode == InteractionMode::Edit) primaries = ComputeEditPrimaries(r);
        else if (!primaries.All.empty()) primaries = {};
    }

    const bool roots_changed = selection || interaction || parents;
    if (roots_changed) ComputeRoots(r, scope, r.edit<TransformRoots>(viewport));
    // The pivot holds still through a gesture, which reads the pivot it started with.
    if (roots_changed || AnyChanged(r, Change::TransformEnd) ||
        (!r.all_of<StartPivot>(viewport) && PivotMoved(r, scope, r.get<const TransformRoots>(viewport)))) {
        ComputePivot(r, scope, r.edit<TransformRoots>(viewport));
    }

    if (primaries_changed || AnyChanged(r, Change::EditMode, Change::MeshGeometry) || r.Context.get<const GpuSceneState>().EditSelectionDirty) {
        ComputeFlags(r, viewport, scope, r.get<const EditPrimaries>(viewport), r.edit<SelectionFlags>(viewport));
    }

    if (selection || AnyChanged(r, Change::AnimationEdited, Change::KeyframeSources, Change::MeshMaterial, Change::ActiveMaterialVariant)) {
        ComputeKeyframes(r, viewport, r.edit<SelectedKeyframes>(viewport));
    }

    if (AnyChanged(r, Change::SceneHierarchy, Change::Names)) BuildOutlinerRows(r, r.edit<OutlinerRows>(viewport));
    if (selection || parents) ComputeSelectedAncestors(r, r.edit<OutlinerRows>(viewport));
}
