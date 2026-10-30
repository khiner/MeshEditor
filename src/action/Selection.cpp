#include "action/Selection.h"
#include "Variant.h"
#include "action/Errors.h"
#include "action/Nodes.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshStore.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "selection/Selection.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "viewport/InteractionComponents.h"
#include "viewport/Viewport.h"

namespace action::selection {
void Apply(state::Registry &r, const Action &action) {
    auto merge_bone_sel = [&](state::Entity e, const std::optional<BoneSel> &part, bool additive) {
        const auto sel = part ? BoneSelection::From(*part) : BoneSelection{};
        const auto *cur = r.try_get<BoneSelection>(e);
        r.emplace_or_replace<BoneSelection>(e, additive && cur ? *cur | sel : sel);
    };
    // Click selection ends an active box-select drag.
    auto end_box_select_interaction = [&] { r.remove<AdditiveBoxSelectBaseline>(state::Document); };
    // Object selection names scene-graph objects, which the gizmo and pivot read through their transforms.
    const auto object = [&](state::Entity e) {
        if (e == state::Null || r.all_of<Transform>(e)) return true;
        Fail(r, "Only an object can be selected");
        return false;
    };

    std::visit(
        overloaded{
            [&](Select a) {
                if (!object(a.Entity)) return;
                end_box_select_interaction();
                ::Select(r, a.Entity);
            },
            [&](ToggleSelected a) {
                if (a.Entity == state::Null || !object(a.Entity)) return;
                end_box_select_interaction();
                if (r.all_of<Selected>(a.Entity)) r.remove<Selected>(a.Entity);
                else r.emplace_or_replace<Selected>(a.Entity);
            },
            [&](SelectBone a) {
                end_box_select_interaction();
                ::SelectBone(r, a.Entity);
                if (a.Part) merge_bone_sel(a.Entity, a.Part, a.Additive);
            },
            [&](ExtendActive a) {
                if (a.Entity == state::Null || !object(a.Entity)) return;
                end_box_select_interaction();
                r.clear<Active>();
                r.emplace<Active>(a.Entity);
                r.emplace_or_replace<Selected>(a.Entity);
            },
            [&](ExtendBoneActive a) {
                end_box_select_interaction();
                r.clear<BoneActive>();
                r.emplace<BoneActive>(a.Entity);
                if (!r.all_of<BoneSelection>(a.Entity)) r.emplace<BoneSelection>(a.Entity, false, false, false);
                if (a.Part) merge_bone_sel(a.Entity, a.Part, a.Additive);
            },
            [&](DeselectAll) {
                end_box_select_interaction();
                const auto interaction_mode = r.get<const Interaction>(state::Document).Mode;
                if (interaction_mode == InteractionMode::Pose || IsBoneEditMode(r)) {
                    r.clear<BoneSelection>();
                } else if (interaction_mode == InteractionMode::Edit) {
                    const auto element = r.get<const EditMode>(state::Document).Value;
                    const auto ranges = GetElementRangesForSelected(r);
                    for (const auto &range : ranges) {
                        r.remove<MeshActiveElement>(range.MeshEntity);
                    }
                    ApplyEditSelectionCommand(r, ranges, element, EditSelectionOperation::Clear);
                    for (const auto &range : ranges) nodes::CaptureActiveSelection(r, range.MeshEntity);
                } else {
                    r.clear<Selected>();
                }
            },
            // GPU selection resolves the rectangle after action application, against the viewport the gesture was emitted in.
            [&](const ApplyBoxSelect &a) { r.emplace_or_replace<PendingBoxSelect>(state::Document, a.Box, a.Additive, FindActiveViewport(r)); },
            [&](const Pick &a) {
                const auto &pick = a.Hit;
                const bool shift = a.Shift;
                const bool bone_mode = r.get<const Interaction>(state::Document).Mode == InteractionMode::Pose || IsBoneEditMode(r);
                const auto entity = pick ? HitEntity(*pick) : state::Null;
                const auto part = pick ? HitPart(*pick) : std::nullopt;
                if (pick && (bone_mode != part.has_value() || (bone_mode ? !r.all_of<BoneIndex>(entity) : !object(entity)))) return;
                const auto active = bone_mode ? FindActiveBone(r) : FindActiveEntity(r);
                if (pick && shift) {
                    if (active == entity && !bone_mode) Apply(r, ToggleSelected{entity});
                    else if (bone_mode) Apply(r, ExtendBoneActive{entity, part, true});
                    else Apply(r, ExtendActive{entity});
                } else if (pick || !shift) {
                    if (pick && bone_mode) Apply(r, SelectBone{entity, part, false});
                    else if (pick) Apply(r, Select{entity});
                    else Apply(r, DeselectAll{});
                }
            },
            [&](const ApplyEditElementClick &a) {
                end_box_select_interaction();
                const auto edit_mode = r.get<const EditMode>(state::Document).Value;
                const auto ranges = GetElementRangesForSelected(r);
                const auto hit = ApplyEditElementPick(r, ranges, edit_mode, a.Hit, a.Toggle);
                if (!a.Toggle) {
                    for (const auto &range : ranges) r.remove<MeshActiveElement>(range.MeshEntity);
                }
                if (hit) {
                    const auto mesh_entity = hit->MeshEntity;
                    const auto &summary = r.Resources.get<const MeshStore>().GetSelectionSummary(*DrawnStoreId(r, mesh_entity));
                    if (summary.ActiveHandle == InvalidOffset) r.remove<MeshActiveElement>(mesh_entity);
                    else r.emplace_or_replace<MeshActiveElement>(mesh_entity, summary.ActiveHandle);
                }
                for (const auto &range : ranges) nodes::CaptureActiveSelection(r, range.MeshEntity);
            },
            [&](const ApplyTreeSelection &a) {
                using Clear = ApplyTreeSelection::ClearKind;
                for (const auto e : a.ToSelect()) {
                    if (!r.all_of<BoneIndex>(e) && !object(e)) return;
                }
                if (a.Clear == Clear::BonesOnly) r.clear<BoneSelection>();
                else if (a.Clear == Clear::All) r.clear<Selected, BoneSelection>();
                for (const auto e : a.ToSelect()) {
                    if (r.all_of<BoneIndex>(e)) r.emplace_or_replace<BoneSelection>(e);
                    else if (!r.all_of<Selected>(e)) r.emplace<Selected>(e);
                }
                for (const auto e : a.ToDeselect()) {
                    if (r.all_of<BoneIndex>(e)) {
                        if (r.all_of<BoneSelection>(e)) r.remove<BoneSelection>(e);
                    } else if (r.all_of<Selected>(e)) {
                        r.remove<Selected>(e);
                    }
                }
                if (a.NavToActive != state::Null) {
                    const bool is_bone = r.all_of<BoneIndex>(a.NavToActive);
                    if (is_bone ? r.all_of<BoneSelection>(a.NavToActive) : r.all_of<Selected>(a.NavToActive)) {
                        if (is_bone) {
                            r.clear<BoneActive>();
                            r.emplace<BoneActive>(a.NavToActive);
                        } else {
                            r.clear<Active>();
                            r.emplace<Active>(a.NavToActive);
                        }
                    }
                }
            },
            [&](SelectAll) {
                const auto interaction_mode = r.get<const Interaction>(state::Document).Mode;
                const auto active_entity = FindActiveEntity(r);
                const auto arm_obj_entity = FindArmatureObject(r, active_entity);
                const bool bone_select = interaction_mode == InteractionMode::Pose || (interaction_mode == InteractionMode::Edit && arm_obj_entity != state::Null);
                if (bone_select) {
                    if (arm_obj_entity == state::Null) return;
                    const auto &arm_obj = r.get<const ArmatureObject>(arm_obj_entity);
                    r.clear<BoneActive, BoneSelection>();
                    for (const auto bone_entity : arm_obj.BoneEntities) r.emplace<BoneSelection>(bone_entity);
                    if (!arm_obj.BoneEntities.empty()) r.emplace<BoneActive>(arm_obj.BoneEntities.back());
                } else if (interaction_mode == InteractionMode::Edit) {
                    const auto element = r.get<const EditMode>(state::Document).Value;
                    const auto ranges = GetElementRangesForSelected(r);
                    ApplyEditSelectionCommand(r, ranges, element, EditSelectionOperation::Fill);
                    for (const auto &range : ranges) nodes::CaptureActiveSelection(r, range.MeshEntity);
                } else if (interaction_mode == InteractionMode::Object) {
                    r.clear<Selected>();
                    const auto scene = FrameScene(r);
                    for (const auto e : r.view<const ObjectKind>()) {
                        if (InScene(r, e, scene)) r.emplace<Selected>(e);
                    }
                }
            },
        },
        action
    );
}
} // namespace action::selection
