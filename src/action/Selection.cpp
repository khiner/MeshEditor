#include "action/Selection.h"
#include "Variant.h"
#include "armature/ArmatureComponents.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "scene/Entity.h"
#include "selection/Selection.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "state/Scene.h"
#include "viewport/InteractionComponents.h"

namespace action::selection {
void Apply(state::Scene &r, state::Entity viewport, const Action &action) {
    auto merge_bone_sel = [&](state::Entity e, const std::optional<BoneSel> &part, bool additive) {
        const auto sel = part ? BoneSelection::From(*part) : BoneSelection{};
        const auto *cur = r.try_get<BoneSelection>(e);
        r.emplace_or_replace<BoneSelection>(e, additive && cur ? *cur | sel : sel);
    };
    // Click selection ends an active box-select drag.
    auto end_box_select_interaction = [&] { r.remove<AdditiveBoxSelectBaseline>(viewport); };

    std::visit(
        overloaded{
            [&](Select a) { end_box_select_interaction(); ::Select(r, a.Entity); },
            [&](ToggleSelected a) {
                end_box_select_interaction();
                if (a.Entity == state::Null) return;
                if (r.all_of<Selected>(a.Entity)) r.remove<Selected>(a.Entity);
                else r.emplace_or_replace<Selected>(a.Entity);
            },
            [&](SelectBone a) {
                end_box_select_interaction();
                ::SelectBone(r, a.Entity);
                if (a.Part) merge_bone_sel(a.Entity, a.Part, a.Additive);
            },
            [&](ExtendActive a) {
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
                const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
                if (interaction_mode == InteractionMode::Pose || IsBoneEditMode(r, viewport)) {
                    r.clear<BoneSelection>();
                } else if (interaction_mode == InteractionMode::Edit) {
                    const auto element = r.get<const EditMode>(viewport).Value;
                    const auto ranges = GetElementRangesForSelected(r, viewport);
                    for (const auto &range : ranges) {
                        r.remove<MeshActiveElement>(range.MeshEntity);
                    }
                    ApplyEditSelectionCommand(r, ranges, element, EditSelectionOperation::Clear);
                } else {
                    r.clear<Selected>();
                }
            },
            // GPU selection resolves the rectangle after action application.
            [&](const ApplyBoxSelect &a) { r.emplace_or_replace<PendingBoxSelect>(viewport, a.Box, a.Additive, *a.View); },
            // GPU selection resolves the pixel after action application.
            [&](const Pick &a) { r.emplace_or_replace<PendingPick>(viewport, a.Mouse, a.Shift, false, *a.View); },
            [&](const PickCycle &a) { r.emplace_or_replace<PendingPick>(viewport, a.Mouse, a.Shift, true, *a.View); },
            [&](const ApplyEditElementClick &a) {
                end_box_select_interaction();
                r.emplace_or_replace<PendingEditElementClick>(viewport, a.Mouse, a.Toggle, *a.View);
            },
            [&](const ApplyTreeSelection &a) {
                using Clear = ApplyTreeSelection::ClearKind;
                if (a.Clear == Clear::BonesOnly) r.clear<BoneSelection>();
                else if (a.Clear == Clear::All) r.clear<Selected, BoneSelection>();
                for (const auto e : a.ToSelect()) {
                    if (r.all_of<BoneIndex>(e)) r.emplace_or_replace<BoneSelection>(e);
                    else if (!r.all_of<Selected>(e)) r.emplace<Selected>(e);
                }
                for (const auto e : a.ToDeselect()) {
                    if (r.all_of<BoneIndex>(e)) {
                        if (r.all_of<BoneSelection>(e)) r.remove<BoneSelection>(e);
                    } else if (r.all_of<Selected>(e)) r.remove<Selected>(e);
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
                const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
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
                    const auto element = r.get<const EditMode>(viewport).Value;
                    const auto ranges = GetElementRangesForSelected(r, viewport);
                    ApplyEditSelectionCommand(r, ranges, element, EditSelectionOperation::Fill);
                } else if (interaction_mode == InteractionMode::Object) {
                    r.clear<Selected>();
                    for (const auto e : r.view<const ObjectKind>()) r.emplace<Selected>(e);
                }
            },
        },
        action
    );
}
} // namespace action::selection
