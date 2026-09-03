#pragma once

#include "BoneConstraint.h"
#include "armature/BoneId.h"
#include "gpu/Transform.h"
#include "numeric/mat3.h"

#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

inline constexpr uint32_t InvalidBoneIndex{std::numeric_limits<uint32_t>::max()};

struct AnimationClip;

struct ArmatureBone {
    BoneId Id;
    BoneId ParentBoneId;
    std::optional<uint32_t> JointNodeIndex;
    std::string Name;
    Transform RestLocal;
    mat4 RestWorld{I4}, InvRestWorld{I4};
    uint32_t ParentIndex{InvalidBoneIndex}, FirstChild{InvalidBoneIndex}, NextSibling{InvalidBoneIndex};
};

struct ArmatureImportedSkin {
    uint32_t SkinIndex;
    std::optional<uint32_t> SkeletonNodeIndex, AnchorNodeIndex;
    std::string Name;
    std::vector<uint32_t> OrderedJointNodeIndices;
    std::vector<mat4> InverseBindMatrices;
};

struct Armature {
    uint32_t Version{1};
    BoneId NextBoneId{1};
    bool Dirty{false};

    std::vector<ArmatureBone> Bones;
    std::unordered_map<BoneId, uint32_t> BoneIdToIndex;
    std::vector<std::vector<uint32_t>> JointOrderToBoneIndex;
    std::vector<ArmatureImportedSkin> Skins;

    BoneId AllocateBoneId();
    std::optional<uint32_t> FindBoneIndex(BoneId) const;
    BoneId AddBone(std::string_view name, std::optional<BoneId> parent_bone_id, const Transform &rest_local, std::optional<uint32_t> joint_node_index = {});
    bool RemoveBone(BoneId bone_id);
    void FinalizeStructure();
    void RebuildCaches();
    void ResolveAnimationIndices(AnimationClip &) const;
    void RecomputeRestWorld();
    void RecomputeInverseBindMatrices();
};

std::vector<uint32_t> CollectBonesForDeletion(const entt::registry &, entt::entity arm_obj_entity);

Transform ComposeWithDelta(const Transform &rest, const Transform &delta);

Transform AbsoluteToDelta(const Transform &rest, const Transform &absolute);

// For each keyed channel, interpolate the absolute glTF keyframe value and convert to rest-relative delta.
void EvaluateAnimationDeltas(const AnimationClip &, float time, std::span<const ArmatureBone>, std::span<Transform> deltas);

// For each joint j of skin `skin_slot`: out[j] = bone_pose_world[bone_for_j] * inverse_bind[j], or I4 if the joint maps to no bone.
void ComputeDeformMatrices(const Armature &, uint32_t skin_slot, std::span<const mat4> bone_pose_world, std::span<mat4> out);

// Blend `pre_local` toward the transform implied by `target_world` at `c.Influence`.
// Performs calculations in armature-local space after `armature_world_inv` converts the target from world space.
// Preserves scale from `pre_local`.
Transform ApplyBoneConstraint(const BoneConstraint &, const Transform &pre_local, const mat4 &parent_pose_world, const mat4 &armature_world_inv, const mat4 &target_world);

// Returns the minimum nonzero distance to a child for non-leaf bones.
// Returns the parent scale or 1.0 for leaf bones.
float ComputeBoneDisplayScale(const Armature &, uint32_t bone_index);

// Returns a basis whose Y axis follows `direction` with `roll` radians of axial rotation.
mat3 BoneVecRollToMat3(vec3 direction, float roll);
void BoneMat3ToVecRoll(const mat3 &m, vec3 &direction, float &roll);
