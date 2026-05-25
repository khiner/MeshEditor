#include "action/Bone.h"
#include "AnimationTimeline.h"
#include "Armature.h"
#include "ArmatureOps.h"
#include "BoneSelection.h"
#include "ObjectOps.h"
#include "SceneGraph.h"
#include "SceneGraphOps.h"
#include "SelectionOps.h"
#include "TransformMath.h"
#include "Variant.h"
#include "ViewportInteractionState.h"
#include "WorldTransform.h"

#include <entt/entity/registry.hpp>

namespace {
void RebuildBoneStructure(entt::registry &r, entt::entity viewport, entt::entity arm_data_entity) {
    RebuildArmatureStructure(r, arm_data_entity);
    r.get<LastEvaluatedFrame>(viewport).Value = -1;
}

entt::entity CreateSingleBoneInstance(entt::registry &r, entt::entity arm_obj_entity, BoneId bone_id) {
    auto &arm_obj = r.get<ArmatureObject>(arm_obj_entity);
    const auto &armature = r.get<const Armature>(arm_obj.Entity);
    const auto new_index = *armature.FindBoneIndex(bone_id);
    const auto parent_index = armature.Bones[new_index].ParentIndex;
    const auto parent_entity = parent_index == InvalidBoneIndex ? arm_obj_entity : arm_obj.BoneEntities[parent_index];
    const auto bone_entity = ::CreateBoneEntity(r, arm_obj_entity, armature, new_index, parent_entity);
    if (arm_obj.JointEntity != null_entity && r.valid(arm_obj.JointEntity)) {
        ::CreateBoneJoints(r, arm_obj_entity, bone_entity, arm_obj.JointEntity);
    }
    arm_obj.BoneEntities.emplace_back(bone_entity);
    return bone_entity;
}
} // namespace

namespace action::bone {
void Apply(entt::registry &r, entt::entity viewport, const Action &action) {
    std::visit(
        overloaded{
            [&](Add) {
                const auto active_entity = FindActiveEntity(r);
                const auto arm_obj_entity = FindArmatureObject(r, active_entity);
                if (arm_obj_entity == entt::null) return;

                auto &armature = r.get<Armature>(r.get<ArmatureObject>(arm_obj_entity).Entity);
                const auto &arm_wt = r.get<WorldTransform>(arm_obj_entity);
                const auto new_id = armature.AddBone("Bone", {}, {.P = (glm::conjugate(glm::normalize(arm_wt.R)) * -arm_wt.P) / arm_wt.S});
                RebuildBoneStructure(r, viewport, r.get<ArmatureObject>(arm_obj_entity).Entity);

                const auto bone_entity = CreateSingleBoneInstance(r, arm_obj_entity, new_id);
                SelectBone(r, bone_entity);
                r.emplace_or_replace<BoneSelection>(bone_entity, false, true, false);
            },
            [&](Extrude) {
                const auto arm_obj_entity = FindArmatureObject(r, FindActiveEntity(r));
                if (arm_obj_entity == entt::null) return;

                auto &arm_obj = r.get<ArmatureObject>(arm_obj_entity);
                auto &armature = r.get<Armature>(arm_obj.Entity);
                auto result = ExtrudeBones(r, armature, arm_obj_entity);
                if (result.NewBoneIds.empty()) return;

                RebuildBoneStructure(r, viewport, arm_obj.Entity);
                r.clear<BoneSelection, BoneActive>();

                for (const auto id : result.NewBoneIds) {
                    const auto bone_entity = CreateSingleBoneInstance(r, arm_obj_entity, id);
                    r.replace<BoneDisplayScale>(bone_entity, 0.f);
                    r.emplace<BoneSelection>(bone_entity, false, true, false);
                    r.emplace_or_replace<BoneActive>(bone_entity);
                }
                for (const auto idx : result.UpdatedParentIndices) {
                    r.replace<BoneDisplayScale>(arm_obj.BoneEntities[idx], ComputeBoneDisplayScale(armature, idx));
                }
                r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate);
            },
            [&](DuplicateSelected) {
                const auto arm_obj_entity = FindArmatureObject(r, FindActiveEntity(r));
                if (arm_obj_entity == entt::null) return;

                auto &armature = r.get<Armature>(r.get<ArmatureObject>(arm_obj_entity).Entity);
                auto result = DuplicateBones(r, armature, arm_obj_entity);
                if (result.Duplicated.empty()) return;

                RebuildBoneStructure(r, viewport, r.get<ArmatureObject>(arm_obj_entity).Entity);
                r.clear<BoneSelection, BoneActive>();

                entt::entity last_bone{};
                for (const auto &[orig_entity, new_id] : result.Duplicated) {
                    last_bone = CreateSingleBoneInstance(r, arm_obj_entity, new_id);
                    r.replace<BoneDisplayScale>(last_bone, r.get<const BoneDisplayScale>(orig_entity).Value);
                    r.emplace<BoneSelection>(last_bone);
                }
                r.emplace<BoneActive>(last_bone);

                r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate);
            },
            [&](const ClearSelectedTransforms &a) { ClearSelectedBoneTransforms(r, a.Position, a.Rotation, a.Scale); },
            [&](DeleteSelected) {
                const auto active_entity = FindActiveEntity(r);
                const auto arm_obj_entity = FindArmatureObject(r, active_entity);
                if (arm_obj_entity == entt::null) return;

                auto &arm_obj = r.get<ArmatureObject>(arm_obj_entity);
                auto &armature = r.get<Armature>(arm_obj.Entity);
                const auto to_delete = CollectBonesForDeletion(r, arm_obj_entity);
                if (to_delete.empty()) return;

                for (const auto idx : to_delete) {
                    const auto bone_entity = arm_obj.BoneEntities[idx];
                    const auto &bone = armature.Bones[idx];
                    const auto grandparent = bone.ParentIndex == InvalidBoneIndex ? arm_obj_entity : arm_obj.BoneEntities[bone.ParentIndex];

                    if (auto *joints = r.try_get<BoneJointEntities>(bone_entity)) {
                        if (joints->Head != null_entity) {
                            Hide(r, joints->Head);
                            r.destroy(joints->Head);
                        }
                        if (joints->Tail != null_entity) {
                            Hide(r, joints->Tail);
                            r.destroy(joints->Tail);
                        }
                        r.remove<BoneJointEntities>(bone_entity);
                    }

                    std::vector<entt::entity> children;
                    for (const auto child : Children{&r, bone_entity}) children.emplace_back(child);
                    for (const auto child : children) {
                        const auto &ct = r.get<const Transform>(child);
                        const auto t = ComposeLocalTransforms(bone.RestLocal, ct);
                        r.emplace_or_replace<Transform>(child, Transform{t.P, t.R, r.all_of<ScaleLocked>(child) ? ct.S : t.S});
                        ClearParent(r, child);
                        SetParent(r, child, grandparent);
                    }

                    ClearParent(r, bone_entity);
                    Hide(r, bone_entity);
                    r.destroy(bone_entity);
                }

                for (const auto idx : to_delete) {
                    armature.RemoveBone(armature.Bones[idx].Id);
                    arm_obj.BoneEntities.erase(arm_obj.BoneEntities.begin() + idx);
                }

                RebuildBoneStructure(r, viewport, arm_obj.Entity);

                for (uint32_t i = 0; i < arm_obj.BoneEntities.size(); ++i) r.get<BoneIndex>(arm_obj.BoneEntities[i]).Index = i;

                if (arm_obj.BoneEntities.empty()) DestroyArmatureData(r, arm_obj_entity);
                ::Select(r, arm_obj_entity);
            },
            [&](const SetEditHeadTailRoll &a) {
                const auto e = FindActiveBone(r);
                r.patch<Transform>(e, [&](auto &t) { t.P = a.LocalP; t.R = a.LocalR; });
                r.get<BoneDisplayScale>(e).Value = a.DisplayScale;
            },
            [&](const SetConstraintTarget &a) {
                r.patch<BoneConstraints>(FindActiveBone(r), [&](auto &cs) { cs.Stack[a.Index].TargetEntity = a.Target; });
            },
            [&](const SetConstraintInfluence &a) {
                r.patch<BoneConstraints>(FindActiveBone(r), [&](auto &cs) { cs.Stack[a.Index].Influence = a.Influence; });
            },
            [&](const SetConstraintChildOfInverse &a) {
                r.patch<BoneConstraints>(FindActiveBone(r), [&](auto &cs) { std::get<ChildOfData>(cs.Stack[a.Index].Data).InverseMatrix = *a.Inverse; });
            },
            [&](const DeleteConstraint &a) {
                r.patch<BoneConstraints>(FindActiveBone(r), [&](auto &cs) { cs.Stack.erase(cs.Stack.begin() + a.Index); });
            },
            [&](const AddConstraint &a) {
                const auto e = FindActiveBone(r);
                if (!r.all_of<BoneConstraints>(e)) r.emplace<BoneConstraints>(e);
                r.patch<BoneConstraints>(e, [&](auto &cs) {
                    cs.Stack.push_back(a.Kind == BoneConstraintKind::ChildOf ? BoneConstraint{.Data = ChildOfData{}} : BoneConstraint{.Data = CopyTransformsData{}});
                });
            },
        },
        action
    );
}
} // namespace action::bone
