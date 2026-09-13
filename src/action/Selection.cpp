#include "action/Selection.h"
#include "Variant.h"
#include "armature/ArmatureComponents.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "project/Registry.h"
#include "scene/Entity.h"
#include "selection/Selection.h"
#include "selection/SelectionBitset.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionOps.h"
#include "selection/SelectionQueries.h"
#include "viewport/InteractionComponents.h"

#include <entt/entity/registry.hpp>

namespace action::selection {
void Apply(entt::registry &r, entt::entity viewport, const Action &action) {
    auto merge_bone_sel = [&](entt::entity e, const std::optional<BoneSel> &part, bool additive) {
        const auto sel = part ? BoneSelection::From(*part) : BoneSelection{};
        const auto *cur = r.try_get<BoneSelection>(e);
        project::EmplaceOrReplace<BoneSelection>(r, e, additive && cur ? *cur | sel : sel);
    };
    // Click selection ends an active box-select drag.
    auto end_box_select_interaction = [&] { project::Remove<AdditiveBoxSelectBaseline>(r, viewport); };

    std::visit(
        overloaded{
            [&](Select a) { end_box_select_interaction(); ::Select(r, a.Entity); },
            [&](ToggleSelected a) {
                end_box_select_interaction();
                if (a.Entity == entt::null) return;
                if (r.all_of<Selected>(a.Entity)) project::Remove<Selected>(r, a.Entity);
                else project::EmplaceOrReplace<Selected>(r, a.Entity);
            },
            [&](SelectBone a) {
                end_box_select_interaction();
                ::SelectBone(r, a.Entity);
                if (a.Part) merge_bone_sel(a.Entity, a.Part, a.Additive);
            },
            [&](ExtendActive a) {
                end_box_select_interaction();
                project::Clear<Active>(r);
                project::Emplace<Active>(r, a.Entity);
                project::EmplaceOrReplace<Selected>(r, a.Entity);
            },
            [&](ExtendBoneActive a) {
                end_box_select_interaction();
                project::Clear<BoneActive>(r);
                project::Emplace<BoneActive>(r, a.Entity);
                if (!r.all_of<BoneSelection>(a.Entity)) project::Emplace<BoneSelection>(r, a.Entity, false, false, false);
                if (a.Part) merge_bone_sel(a.Entity, a.Part, a.Additive);
            },
            [&](const SetBoneSelectionPart &a) { merge_bone_sel(a.Entity, a.Part, a.Additive); },
            [&](DeselectAll) {
                end_box_select_interaction();
                const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
                if (interaction_mode == InteractionMode::Pose || IsBoneEditMode(r, viewport)) {
                    project::Clear<BoneSelection>(r);
                } else if (interaction_mode == InteractionMode::Edit) {
                    const auto element = r.get<const EditMode>(viewport).Value;
                    const auto ranges = GetElementRangesForSelected(r, viewport);
                    for (const auto &range : ranges) {
                        project::Remove<MeshActiveElement>(r, range.MeshEntity);
                    }
                    ApplyEditSelectionCommand(r, viewport, ranges, element, EditSelectionOperation::Clear);
                } else {
                    project::Clear<Selected>(r);
                }
            },
            [&](SnapshotBoxSelectBaseline) {
                const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
                const auto active_entity = FindActiveEntity(r);
                const bool active_is_armature = FindArmatureObject(r, active_entity) != entt::null;
                AdditiveBoxSelectBaseline baseline;
                if (interaction_mode == InteractionMode::Edit && !active_is_armature) {
                    // Preserve the initial domain masks throughout the drag.
                } else if (interaction_mode == InteractionMode::Pose || (interaction_mode == InteractionMode::Edit && active_is_armature)) {
                    for (const auto e : r.view<BoneSelection>()) baseline.BoneSelections.emplace_back(e, r.get<BoneSelection>(e));
                } else if (interaction_mode == InteractionMode::Object) {
                    for (const auto e : r.view<Selected>()) baseline.SelectedEntities.emplace_back(e);
                }
                project::EmplaceOrReplace<AdditiveBoxSelectBaseline>(r, viewport, std::move(baseline));
            },
            [&](ClearBoxSelectBaseline) {
                project::Remove<AdditiveBoxSelectBaseline>(r, viewport);
            },
            // GPU selection resolves the rectangle after action application.
            [&](const ApplyBoxSelect &a) { project::EmplaceOrReplace<PendingBoxSelect>(r, viewport, a.BoxPx, a.Additive, *a.View); },
            // GPU selection resolves the pixel after action application.
            [&](const Pick &a) { project::EmplaceOrReplace<PendingPick>(r, viewport, a.MousePx, a.Shift, false, *a.View); },
            [&](const PickCycle &a) { project::EmplaceOrReplace<PendingPick>(r, viewport, a.MousePx, a.Shift, true, *a.View); },
            [&](const ApplyEditElementClick &a) {
                end_box_select_interaction();
                project::EmplaceOrReplace<PendingEditElementClick>(r, viewport, a.MousePx, a.Toggle, *a.View);
            },
            [&](const ApplyTreeSelection &a) {
                using Clear = ApplyTreeSelection::ClearKind;
                if (a.Clear == Clear::BonesOnly) project::Clear<BoneSelection>(r);
                else if (a.Clear == Clear::All) project::Clear<Selected, BoneSelection>(r);
                for (const auto e : a.ToSelect()) {
                    if (r.all_of<BoneIndex>(e)) project::EmplaceOrReplace<BoneSelection>(r, e);
                    else if (!r.all_of<Selected>(e)) project::Emplace<Selected>(r, e);
                }
                for (const auto e : a.ToDeselect()) {
                    if (r.all_of<BoneIndex>(e)) {
                        if (r.all_of<BoneSelection>(e)) project::Remove<BoneSelection>(r, e);
                    } else if (r.all_of<Selected>(e)) project::Remove<Selected>(r, e);
                }
                if (a.NavToActive != entt::null) {
                    const bool is_bone = r.all_of<BoneIndex>(a.NavToActive);
                    if (is_bone ? r.all_of<BoneSelection>(a.NavToActive) : r.all_of<Selected>(a.NavToActive)) {
                        if (is_bone) {
                            project::Clear<BoneActive>(r);
                            project::Emplace<BoneActive>(r, a.NavToActive);
                        } else {
                            project::Clear<Active>(r);
                            project::Emplace<Active>(r, a.NavToActive);
                        }
                    }
                }
            },
            [&](SelectAll) {
                const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
                const auto active_entity = FindActiveEntity(r);
                const auto arm_obj_entity = FindArmatureObject(r, active_entity);
                const bool bone_select = interaction_mode == InteractionMode::Pose || (interaction_mode == InteractionMode::Edit && arm_obj_entity != entt::null);
                if (bone_select) {
                    if (arm_obj_entity == entt::null) return;
                    const auto &arm_obj = r.get<const ArmatureObject>(arm_obj_entity);
                    project::Clear<BoneActive, BoneSelection>(r);
                    for (const auto bone_entity : arm_obj.BoneEntities) project::Emplace<BoneSelection>(r, bone_entity);
                    if (!arm_obj.BoneEntities.empty()) project::Emplace<BoneActive>(r, arm_obj.BoneEntities.back());
                } else if (interaction_mode == InteractionMode::Edit) {
                    const auto element = r.get<const EditMode>(viewport).Value;
                    const auto ranges = GetElementRangesForSelected(r, viewport);
                    ApplyEditSelectionCommand(r, viewport, ranges, element, EditSelectionOperation::Fill);
                } else if (interaction_mode == InteractionMode::Object) {
                    project::Clear<Active, Selected>(r);
                    entt::entity last{entt::null};
                    for (const auto [e, _] : r.view<const ObjectKind>().each()) {
                        project::Emplace<Selected>(r, e);
                        last = e;
                    }
                    if (last != entt::null) project::Emplace<Active>(r, last);
                }
            },
        },
        action
    );
}
} // namespace action::selection
