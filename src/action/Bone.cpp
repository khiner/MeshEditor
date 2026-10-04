#include "action/Bone.h"
#include "TransformMath.h"
#include "Variant.h"
#include "animation/AnimationData.h"
#include "animation/AnimationTimeline.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "numeric/VectorMath.h"
#include "object/ObjectOps.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "scene/WorldTransform.h"
#include "selection/BoneSelection.h"
#include "selection/Selection.h"
#include "state/Scene.h"
#include "viewport/ViewportInteractionState.h"

#include <format>
#include <unordered_set>

namespace {
// Finalize armature structure after AddBone/RemoveBone. Resets pose state, re-resolves animation indices, and forces a re-evaluation of the current frame.
void RebuildBoneStructure(state::Scene &r, state::Entity viewport, state::Entity arm_data_entity) {
    auto &armature = r.edit<Armature>(arm_data_entity);
    armature.FinalizeStructure();
    armature.RecomputeRestWorld();

    for (const auto [_, arm_obj] : r.view<const ArmatureObject>().each()) {
        if (arm_obj.Entity != arm_data_entity) continue;
        for (const auto b : arm_obj.BoneEntities)
            if (!r.all_of<BoneDelta>(b)) r.emplace<BoneDelta>(b);
    }
    if (auto *ps = r.try_edit<ArmaturePoseState>(arm_data_entity)) {
        ps->BoneUserOffset.assign(armature.Bones.size(), Transform{});
        ps->BonePoseWorld.assign(armature.Bones.size(), I4);
    }
    r.edit<LastEvaluatedFrame>(viewport).Value = -1;
}

// `display_scales` holds every bone's display scale, in bone order.
state::Entity CreateSingleBoneInstance(state::Scene &r, state::Entity arm_obj_entity, BoneId bone_id, std::span<const float> display_scales) {
    auto &arm_obj = r.edit<ArmatureObject>(arm_obj_entity);
    const auto &armature = r.get<const Armature>(arm_obj.Entity);
    const auto new_index = *armature.FindBoneIndex(bone_id);
    const auto parent_index = armature.Bones[new_index].ParentIndex;
    const auto parent_entity = parent_index == InvalidBoneIndex ? arm_obj_entity : arm_obj.BoneEntities[parent_index];
    const auto bone_entity = ::CreateBoneEntity(r, arm_obj_entity, armature, new_index, parent_entity, display_scales[new_index]);
    if (arm_obj.JointEntity != state::Null && r.valid(arm_obj.JointEntity)) {
        ::CreateBoneJoints(r, arm_obj_entity, bone_entity, arm_obj.JointEntity);
    }
    arm_obj.BoneEntities.emplace_back(bone_entity);
    return bone_entity;
}
} // namespace

namespace action::bone {
void Apply(state::Scene &r, state::Entity viewport, const Action &action) {
    std::visit(
        overloaded{
            [&](Add) {
                const auto active_entity = FindActiveEntity(r);
                const auto arm_obj_entity = FindArmatureObject(r, active_entity);
                if (arm_obj_entity == state::Null) return;

                auto &armature = r.edit<Armature>(r.get<ArmatureObject>(arm_obj_entity).Entity);
                const auto &arm_wt = *WorldTransformOf(r, arm_obj_entity);
                const auto new_id = armature.AddBone("Bone", {}, {.P = (Conjugate(Normalize(arm_wt.R)) * -arm_wt.P) / arm_wt.S});
                RebuildBoneStructure(r, viewport, r.get<ArmatureObject>(arm_obj_entity).Entity);

                const auto bone_entity = CreateSingleBoneInstance(r, arm_obj_entity, new_id, ComputeBoneDisplayScales(armature));
                SelectBone(r, bone_entity);
                r.emplace_or_replace<BoneSelection>(bone_entity, false, true, false);
            },
            [&](Extrude) {
                const auto arm_obj_entity = FindArmatureObject(r, FindActiveEntity(r));
                if (arm_obj_entity == state::Null) return;

                auto &arm_obj = r.edit<ArmatureObject>(arm_obj_entity);
                auto &armature = r.edit<Armature>(arm_obj.Entity);

                // Classify: tip or body selected → extrude from tip (child); root-only → extrude from root (sibling).
                // For root extrude, skip if parent bone's tip is also selected.
                std::vector<BoneId> new_bone_ids;
                std::vector<uint32_t> updated_parent_indices;
                for (const auto e : r.view<BoneSelection, BoneIndex>()) {
                    if (r.get<SubElementOf>(e).Parent != arm_obj_entity) continue;
                    const auto idx = r.get<BoneIndex>(e).Index;
                    const auto &bone = armature.Bones[idx];
                    const auto *parts = r.try_get<const BoneSelection>(e);
                    const bool from_tip = !(parts && parts->Root && !parts->Tip && !parts->Body);
                    if (!from_tip) {
                        if (bone.ParentIndex != InvalidBoneIndex) {
                            const auto *pp = r.try_get<const BoneSelection>(arm_obj.BoneEntities[bone.ParentIndex]);
                            if (pp && pp->Tip) continue;
                        }
                        const auto parent = bone.ParentBoneId == InvalidBoneId ? std::optional<BoneId>{} : std::optional{bone.ParentBoneId};
                        new_bone_ids.emplace_back(armature.AddBone("", parent, bone.RestLocal));
                    } else {
                        new_bone_ids.emplace_back(armature.AddBone("", bone.Id, {.P = vec3{0, r.get<BoneDisplayScale>(e).Value, 0}}));
                        updated_parent_indices.emplace_back(idx);
                    }
                }
                if (new_bone_ids.empty()) return;

                RebuildBoneStructure(r, viewport, arm_obj.Entity);
                r.clear<BoneSelection, BoneActive>();

                const auto display_scales = ComputeBoneDisplayScales(armature);
                for (const auto id : new_bone_ids) {
                    const auto bone_entity = CreateSingleBoneInstance(r, arm_obj_entity, id, display_scales);
                    r.replace<BoneDisplayScale>(bone_entity, 0.f);
                    r.emplace<BoneSelection>(bone_entity, false, true, false);
                    r.emplace_or_replace<BoneActive>(bone_entity);
                }
                for (const auto idx : updated_parent_indices) {
                    r.replace<BoneDisplayScale>(arm_obj.BoneEntities[idx], display_scales[idx]);
                }
                r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate);
            },
            [&](DuplicateSelected) {
                const auto arm_obj_entity = FindArmatureObject(r, FindActiveEntity(r));
                if (arm_obj_entity == state::Null) return;

                auto &armature = r.edit<Armature>(r.get<ArmatureObject>(arm_obj_entity).Entity);
                std::unordered_set<std::string> names;
                for (const auto &bone : armature.Bones) names.insert(bone.Name);
                auto unique_name = [&](std::string_view base) {
                    for (uint32_t i = 1;; ++i) {
                        if (auto c = std::format("{}.{:03d}", base, i); names.insert(c).second) return c;
                    }
                };
                std::unordered_map<BoneId, BoneId> orig_to_new;
                std::vector<std::pair<state::Entity, BoneId>> duplicated; // {original entity, new bone id}
                for (const auto e : r.view<BoneSelection, BoneIndex>()) {
                    if (r.get<SubElementOf>(e).Parent != arm_obj_entity) continue;
                    const auto &bone = armature.Bones[r.get<BoneIndex>(e).Index];
                    const auto parent = bone.ParentBoneId == InvalidBoneId ? std::optional<BoneId>{} : std::optional{bone.ParentBoneId};
                    const auto new_id = armature.AddBone(unique_name(bone.Name), parent, bone.RestLocal);
                    orig_to_new[bone.Id] = new_id;
                    duplicated.emplace_back(e, new_id);
                }
                // Remap: if both a bone and its parent were duplicated, point duplicate child to duplicate parent.
                for (const auto &dup : duplicated) {
                    auto &nb = armature.Bones[*armature.FindBoneIndex(dup.second)];
                    if (auto it = orig_to_new.find(nb.ParentBoneId); it != orig_to_new.end()) nb.ParentBoneId = it->second;
                }
                if (duplicated.empty()) return;

                RebuildBoneStructure(r, viewport, r.get<ArmatureObject>(arm_obj_entity).Entity);
                r.clear<BoneSelection, BoneActive>();

                const auto display_scales = ComputeBoneDisplayScales(armature);
                state::Entity last_bone{};
                for (const auto &[orig_entity, new_id] : duplicated) {
                    last_bone = CreateSingleBoneInstance(r, arm_obj_entity, new_id, display_scales);
                    r.replace<BoneDisplayScale>(last_bone, r.get<const BoneDisplayScale>(orig_entity).Value);
                    r.emplace<BoneSelection>(last_bone);
                }
                r.emplace<BoneActive>(last_bone);

                r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate);
            },
            [&](const ClearSelectedTransforms &a) {
                if (FindArmatureObject(r, FindActiveEntity(r)) == state::Null) return;
                for (const auto b : r.view<const BoneSelection, const BoneDelta>()) {
                    r.patch<BoneDelta>(b, [&](auto &delta) {
                        if (a.Position) delta.Value.P = {};
                        if (a.Rotation) delta.Value.R = {};
                        if (a.Scale) delta.Value.S = vec3{1};
                    });
                }
            },
            [&](DeleteSelected) {
                const auto active_entity = FindActiveEntity(r);
                const auto arm_obj_entity = FindArmatureObject(r, active_entity);
                if (arm_obj_entity == state::Null) return;

                auto &arm_obj = r.edit<ArmatureObject>(arm_obj_entity);
                auto &armature = r.edit<Armature>(arm_obj.Entity);
                const auto to_delete = CollectBonesForDeletion(r, arm_obj_entity);
                if (to_delete.empty()) return;

                std::vector<state::Entity> destroyed;
                for (const auto idx : to_delete) {
                    const auto bone_entity = arm_obj.BoneEntities[idx];
                    const auto &bone = armature.Bones[idx];
                    const auto grandparent = bone.ParentIndex == InvalidBoneIndex ? arm_obj_entity : arm_obj.BoneEntities[bone.ParentIndex];

                    if (const auto *joints = r.try_get<const BoneJointEntities>(bone_entity)) {
                        if (joints->Head != state::Null) destroyed.push_back(joints->Head);
                        if (joints->Tail != state::Null) destroyed.push_back(joints->Tail);
                    }

                    std::vector<state::Entity> children;
                    for (const auto child : Children{&r, bone_entity}) children.emplace_back(child);
                    ClearParents(r, children);
                    for (const auto child : children) {
                        const auto ct = *ComposedLocal(r, child);
                        const auto t = ComposeLocalTransforms(bone.RestLocal, ct);
                        PatchEditedLocal(r, child, [&](auto &local) { local = Transform{t.P, t.R, r.all_of<ScaleLocked>(child) ? ct.S : t.S}; });
                        SetParent(r, child, grandparent);
                    }

                    ClearParents(r, std::span{&bone_entity, 1u});
                    destroyed.push_back(bone_entity);
                }
                r.destroy(destroyed);

                for (const auto idx : to_delete) {
                    armature.RemoveBone(armature.Bones[idx].Id);
                    arm_obj.BoneEntities.erase(arm_obj.BoneEntities.begin() + idx);
                }

                RebuildBoneStructure(r, viewport, arm_obj.Entity);

                for (uint32_t i = 0; i < arm_obj.BoneEntities.size(); ++i) {
                    const auto e = arm_obj.BoneEntities[i];
                    if (r.get<const BoneIndex>(e).Index != i) r.replace<BoneIndex>(e, i);
                }

                if (arm_obj.BoneEntities.empty()) DestroyArmatureData(r, arm_obj_entity);
                ::Select(r, arm_obj_entity);
            },
            [&](const SetEditHeadTailRoll &a) {
                const auto e = FindActiveBone(r);
                r.patch<PosedLocal>(e, [&](auto &posed) { posed.Value.P = a.LocalP; posed.Value.R = a.LocalR; });
                r.replace<BoneDisplayScale>(e, a.DisplayScale);
            },
            [&](const SetConstraintTarget &a) {
                r.patch<BoneConstraints>(FindActiveBone(r), [&](auto &cs) { cs.Stack[a.Index].TargetEntity = a.Target; });
            },
            [&](const SetConstraintInfluence &a) {
                r.patch<BoneConstraints>(FindActiveBone(r), [&](auto &cs) { cs.Stack[a.Index].Influence = a.Influence; });
            },
            [&](const BakeConstraintChildOfInverse &a) {
                const auto bone = FindActiveBone(r);
                r.patch<BoneConstraints>(bone, [&](auto &cs) {
                    const auto target = cs.Stack[a.Index].TargetEntity;
                    const auto *twt = WorldTransformOf(r, target);
                    const auto *bwt = WorldTransformOf(r, bone);
                    // Bake inverse(target_world) * bone_world so the current relative pose becomes the new rest.
                    if (twt && bwt) std::get<ChildOfData>(cs.Stack[a.Index].Data).InverseMatrix = Inverse(ToMatrix(*twt)) * ToMatrix(*bwt);
                });
            },
            [&](const ClearConstraintChildOfInverse &a) {
                r.patch<BoneConstraints>(FindActiveBone(r), [&](auto &cs) { std::get<ChildOfData>(cs.Stack[a.Index].Data).InverseMatrix = I4; });
            },
            [&](const DeleteConstraint &a) {
                r.patch<BoneConstraints>(FindActiveBone(r), [&](auto &cs) { cs.Stack.erase(cs.Stack.begin() + a.Index); });
            },
            [&](const AddConstraint &a) {
                const auto e = FindActiveBone(r);
                if (!r.all_of<BoneConstraints>(e)) r.emplace<BoneConstraints>(e);
                r.patch<BoneConstraints>(e, [&](auto &cs) {
                    cs.Stack.emplace_back(a.Kind == BoneConstraintKind::ChildOf ? BoneConstraint{.Data = ChildOfData{}} : BoneConstraint{.Data = CopyTransformsData{}});
                });
            },
        },
        action
    );
}
} // namespace action::bone
