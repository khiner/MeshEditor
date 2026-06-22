#pragma once

#include "BoneConstraint.h"
#include "armature/BoneId.h"
#include "gpu/Transform.h"
#include "numeric/mat3.h"

#include <span>
#include <string>
#include <string_view>

inline constexpr uint32_t InvalidBoneIndex{std::numeric_limits<uint32_t>::max()};

struct AnimationClip;

struct ArmatureBone {
    BoneId Id;
    BoneId ParentBoneId; // Stable
    std::optional<uint32_t> JointNodeIndex; // Optional imported glTF joint node binding
    std::string Name;
    Transform RestLocal;
    mat4 RestWorld{I4}, InvRestWorld{I4};
    uint32_t ParentIndex{InvalidBoneIndex}, FirstChild{InvalidBoneIndex}, NextSibling{InvalidBoneIndex}; // Dense runtime caches
};

struct ArmatureImportedSkin {
    uint32_t SkinIndex;
    std::optional<uint32_t> SkeletonNodeIndex, AnchorNodeIndex;
    std::vector<uint32_t> OrderedJointNodeIndices;
    std::vector<mat4> InverseBindMatrices;
};

struct Armature {
    uint32_t Version{1}; // Increments when structure changes
    BoneId NextBoneId{1}; // Monotonic allocator state
    bool Dirty{false}; // True when structural edits need finalization.

    std::vector<ArmatureBone> Bones;
    std::unordered_map<BoneId, uint32_t> BoneIdToIndex;
    std::vector<uint32_t> JointOrderToBoneIndex; // Precomputed: skin joint order -> bone array index (InvalidBoneIndex if unmapped).
    std::optional<ArmatureImportedSkin> ImportedSkin;

    BoneId AllocateBoneId();
    std::optional<uint32_t> FindBoneIndex(BoneId) const;
    BoneId AddBone(std::string_view name, std::optional<BoneId> parent_bone_id, const Transform &rest_local, std::optional<uint32_t> joint_node_index = {});
    bool RemoveBone(BoneId bone_id); // Returns true if bone was found and removed.
    void FinalizeStructure(); // Rebuild derived caches and increment version. Call after AddBone/RemoveBone.
    void RebuildCaches(); // Rebuild derived caches (BoneIdToIndex, dense topology, RestWorld) from canonical bone data. No version bump.
    void ResolveAnimationIndices(AnimationClip &) const; // Resolve TargetBoneId -> BoneIndex for all bone channels.
    void RecomputeRestWorld(); // Recompute RestWorld/InvRestWorld from RestLocal (no reordering).
    void RecomputeInverseBindMatrices(); // Update IBMs from current RestWorld after rest pose edits.
};

// Canonical bone structure, rest pose, and imported skin (not the pose itself, which lives in the bone entity Transforms).

std::vector<uint32_t> CollectBonesForDeletion(const entt::registry &, entt::entity arm_obj_entity);

// Compose a rest-pose transform with a delta
Transform ComposeWithDelta(const Transform &rest, const Transform &delta);

// Absolute local transform to delta relative to rest
Transform AbsoluteToDelta(const Transform &rest, const Transform &absolute);

// For each keyed channel, interpolate the absolute glTF keyframe value and convert to rest-relative delta.
void EvaluateAnimationDeltas(const AnimationClip &, float time, std::span<const ArmatureBone>, std::span<Transform> deltas);

// For each skin joint j: out[j] = bone_pose_world[bone_for_j] * inverse_bind[j], or I4 if the joint maps to no bone.
void ComputeDeformMatrices(const Armature &, std::span<const mat4> bone_pose_world, std::span<const mat4> inverse_bind, std::span<mat4> out);

// Blend `pre_local` toward the transform implied by `target_world` at `c.Influence`.
// Math is armature-local; `armature_world_inv` converts target from world. Scale is preserved from `pre_local`.
Transform ApplyBoneConstraint(const BoneConstraint &, const Transform &pre_local, const mat4 &parent_pose_world, const mat4 &armature_world_inv, const mat4 &target_world);

// Non-leaf: minimum distance to any child (ignoring near-zero). Leaf: inherit parent's scale, or 1.0.
float ComputeBoneDisplayScale(const Armature &, uint32_t bone_index);

// Direction is the normalized bone Y axis (head->tail). Roll is twist around that axis.
mat3 BoneVecRollToMat3(vec3 direction, float roll);
void BoneMat3ToVecRoll(const mat3 &m, vec3 &direction, float &roll);
