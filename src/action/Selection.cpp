#include "action/Selection.h"
#include "../Selection.h"
#include "../SelectionBitset.h"
#include "Armature.h"
#include "ArmatureOps.h"
#include "Entity.h"
#include "InteractionComponents.h"
#include "SelectionComponents.h"
#include "SelectionOps.h"
#include "Variant.h"

#include <entt/entity/registry.hpp>

namespace action::selection {
void Apply(entt::registry &r, entt::entity viewport, const Action &action) {
    auto apply_box_select = [&]<typename Tag>(bool additive, auto restore_baseline, auto apply_hits) {
        r.clear<Tag>();
        if (additive) {
            if (const auto *baseline = r.try_get<const AdditiveBoxSelectBaseline>(viewport)) restore_baseline(*baseline);
        }
        apply_hits();
    };
    auto merge_bone_sel = [&](entt::entity e, const std::optional<BoneSel> &part, bool additive) {
        const auto sel = part ? BoneSelection::From(*part) : BoneSelection{};
        const auto *cur = r.try_get<BoneSelection>(e);
        r.emplace_or_replace<BoneSelection>(e, additive && cur ? *cur | sel : sel);
    };
    // AdditiveBoxSelectBaseline is meaningful only during an active box-select drag;
    // a click-selection always ends one, so its handler owns the cleanup.
    auto end_box_select_interaction = [&] { r.remove<AdditiveBoxSelectBaseline>(viewport); };

    std::visit(
        overloaded{
            [&](Select a) { end_box_select_interaction(); ::Select(r, a.Entity); },
            [&](ToggleSelected a) { end_box_select_interaction(); ::ToggleSelected(r, a.Entity); },
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
            [&](const SetBoneSelectionPart &a) { merge_bone_sel(a.Entity, a.Part, a.Additive); },
            [&](DeselectAll) {
                end_box_select_interaction();
                const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
                const bool bone_mode = interaction_mode == InteractionMode::Pose || IsBoneEditMode(r, viewport);
                if (bone_mode) r.clear<BoneSelection>();
                else r.clear<Selected>();
            },
            [&](SnapshotBoxSelectBaseline) {
                const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
                const auto active_entity = FindActiveEntity(r);
                const bool active_is_armature = FindArmatureObject(r, active_entity) != entt::null;
                const bool bone_mode = interaction_mode == InteractionMode::Pose || (interaction_mode == InteractionMode::Edit && active_is_armature);
                AdditiveBoxSelectBaseline baseline;
                if (interaction_mode == InteractionMode::Edit && !active_is_armature) {
                    if (const auto ranges = GetBitsetRangesForSelected(r); !ranges.empty()) {
                        const auto element_count = std::ranges::fold_left(ranges, uint32_t{0}, [](uint32_t total, const auto &range) { return std::max(total, range.Offset + range.Count); });
                        const uint32_t bitset_words = (element_count + 31) / 32;
                        const auto bits = r.get<const SelectionBitsetRef>(viewport).Value;
                        baseline.ElementBitset.assign(bits.begin(), bits.begin() + bitset_words);
                    }
                } else if (bone_mode) {
                    for (const auto e : r.view<BoneSelection>()) baseline.BoneSelections.emplace_back(e, r.get<BoneSelection>(e));
                } else if (interaction_mode == InteractionMode::Object) {
                    for (const auto e : r.view<Selected>()) baseline.SelectedEntities.push_back(e);
                }
                r.emplace_or_replace<AdditiveBoxSelectBaseline>(viewport, std::move(baseline));
            },
            [&](ClearBoxSelectBaseline) { r.remove<AdditiveBoxSelectBaseline>(viewport); },
            [&](const ApplyBoxSelectObjectHits &a) {
                apply_box_select.template operator()<Selected>(
                    a.Additive,
                    [&](const AdditiveBoxSelectBaseline &b) {
                        for (const auto e : b.SelectedEntities) {
                            if (r.valid(e)) r.emplace_or_replace<Selected>(e);
                        }
                    },
                    [&] {
                        for (const auto e : a.Hits) r.emplace_or_replace<Selected>(e);
                    }
                );
            },
            [&](const ApplyEditElementClick &a) {
                end_box_select_interaction();
                r.emplace_or_replace<PendingEditElementClick>(viewport, a.MousePx, a.Toggle);
            },
            [&](const ApplyTreeSelection &a) {
                using Clear = ApplyTreeSelection::ClearKind;
                if (a.Clear == Clear::BonesOnly) r.clear<BoneSelection>();
                else if (a.Clear == Clear::All) r.clear<Selected, BoneSelection>();
                for (const auto e : a.ToSelect) {
                    if (r.all_of<BoneIndex>(e)) r.emplace_or_replace<BoneSelection>(e);
                    else if (!r.all_of<Selected>(e)) r.emplace<Selected>(e);
                }
                for (const auto e : a.ToDeselect) {
                    if (r.all_of<BoneIndex>(e)) {
                        if (r.all_of<BoneSelection>(e)) r.remove<BoneSelection>(e);
                    } else if (r.all_of<Selected>(e)) r.remove<Selected>(e);
                }
                if (a.NavToActive != entt::null) {
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
            [&](const ApplyBoxSelectBoneHits &a) {
                apply_box_select.template operator()<BoneSelection>(
                    a.Additive,
                    [&](const AdditiveBoxSelectBaseline &b) {
                        for (const auto &[e, sel] : b.BoneSelections) {
                            if (r.valid(e)) r.emplace_or_replace<BoneSelection>(e, sel);
                        }
                    },
                    [&] {
                        for (const auto &[entity, part] : a.Hits) merge_bone_sel(entity, part, a.Additive);
                    }
                );
            },
            [&](SelectAll) {
                const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
                const auto active_entity = FindActiveEntity(r);
                const auto arm_obj_entity = FindArmatureObject(r, active_entity);
                const bool bone_select = interaction_mode == InteractionMode::Pose || (interaction_mode == InteractionMode::Edit && arm_obj_entity != entt::null);
                if (bone_select) {
                    if (arm_obj_entity == entt::null) return;
                    const auto &arm_obj = r.get<const ArmatureObject>(arm_obj_entity);
                    r.clear<BoneActive, BoneSelection>();
                    for (const auto bone_entity : arm_obj.BoneEntities) r.emplace<BoneSelection>(bone_entity);
                    if (!arm_obj.BoneEntities.empty()) r.emplace<BoneActive>(arm_obj.BoneEntities.back());
                } else if (interaction_mode == InteractionMode::Edit) {
                    const auto ranges = GetBitsetRangesForSelected(r);
                    auto *bits = r.get<SelectionBitsetRef>(viewport).Value.data();
                    for (const auto &range : ranges) ::selection::SelectAll(bits, range.Offset, range.Count);
                    if (!ranges.empty()) r.emplace_or_replace<SelectionBitsDirty>(viewport);
                } else if (interaction_mode == InteractionMode::Object) {
                    r.clear<Active, Selected>();
                    entt::entity last{entt::null};
                    for (const auto [e, _] : r.view<const ObjectKind>().each()) {
                        r.emplace<Selected>(e);
                        last = e;
                    }
                    if (last != entt::null) r.emplace<Active>(last);
                }
            },
        },
        action
    );
}
} // namespace action::selection
