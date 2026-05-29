#include "action/Bone.h"
#include "TransformMath.h"
#include "Variant.h"
#include "animation/AnimationData.h"
#include "animation/AnimationTimeline.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "object/ObjectOps.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "scene/WorldTransform.h"
#include "selection/BoneSelection.h"
#include "selection/Selection.h"
#include "selection/SelectionOps.h"
#include "viewport/ViewportInteractionState.h"

#include <entt/entity/registry.hpp>

#include <format>

namespace {
// Finalize armature structure after AddBone/RemoveBone. Resets pose state, re-resolves animation indices,
// and forces a re-evaluation of the current frame.
void RebuildBoneStructure(entt::registry &r, entt::entity viewport, entt::entity arm_data_entity) {
    auto &armature = r.get<Armature>(arm_data_entity);
    armature.FinalizeStructure();
    armature.RecomputeRestWorld();

    if (auto *ps = r.try_get<ArmaturePoseState>(arm_data_entity)) {
        ps->BonePoseDelta.assign(armature.Bones.size(), Transform{});
        ps->BoneUserOffset.assign(armature.Bones.size(), Transform{});
        ps->BonePoseWorld.assign(armature.Bones.size(), I4);
    }
    if (auto *anim = r.try_get<ArmatureAnimation>(arm_data_entity)) {
        for (auto &clip : anim->Clips) armature.ResolveAnimationIndices(clip);
    }
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

                for (const auto id : new_bone_ids) {
                    const auto bone_entity = CreateSingleBoneInstance(r, arm_obj_entity, id);
                    r.replace<BoneDisplayScale>(bone_entity, 0.f);
                    r.emplace<BoneSelection>(bone_entity, false, true, false);
                    r.emplace_or_replace<BoneActive>(bone_entity);
                }
                for (const auto idx : updated_parent_indices) {
                    r.replace<BoneDisplayScale>(arm_obj.BoneEntities[idx], ComputeBoneDisplayScale(armature, idx));
                }
                r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate);
            },
            [&](DuplicateSelected) {
                const auto arm_obj_entity = FindArmatureObject(r, FindActiveEntity(r));
                if (arm_obj_entity == entt::null) return;

                auto &armature = r.get<Armature>(r.get<ArmatureObject>(arm_obj_entity).Entity);
                auto unique_name = [&](std::string_view base) {
                    for (uint32_t i = 1;; ++i) {
                        if (auto c = std::format("{}.{:03d}", base, i);
                            std::ranges::none_of(armature.Bones, [&](const auto &b) { return b.Name == c; })) return c;
                    }
                };
                std::unordered_map<BoneId, BoneId> orig_to_new;
                std::vector<std::pair<entt::entity, BoneId>> duplicated; // {original entity, new bone id}
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

                entt::entity last_bone{};
                for (const auto &[orig_entity, new_id] : duplicated) {
                    last_bone = CreateSingleBoneInstance(r, arm_obj_entity, new_id);
                    r.replace<BoneDisplayScale>(last_bone, r.get<const BoneDisplayScale>(orig_entity).Value);
                    r.emplace<BoneSelection>(last_bone);
                }
                r.emplace<BoneActive>(last_bone);

                r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate);
            },
            [&](const ClearSelectedTransforms &a) {
                const auto arm_obj_entity = FindArmatureObject(r, FindActiveEntity(r));
                if (arm_obj_entity == entt::null) return;

                const auto &arm_obj = r.get<const ArmatureObject>(arm_obj_entity);
                const auto &armature = r.get<const Armature>(arm_obj.Entity);
                for (const auto b : r.view<const BoneSelection, const BoneIndex>()) {
                    const auto idx = r.get<const BoneIndex>(b).Index;
                    const auto &rest = armature.Bones[idx].RestLocal;
                    r.patch<Transform>(b, [&](auto &t) {
                        if (a.Position) t.P = rest.P;
                        if (a.Rotation) t.R = rest.R;
                        if (a.Scale) t.S = rest.S;
                    });
                }
            },
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
                r.replace<BoneDisplayScale>(e, a.DisplayScale);
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
