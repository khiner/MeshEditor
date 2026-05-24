#pragma once

#include "AnimationData.h"
#include "Range.h"
#include "entt_fwd.h"
#include "numeric/mat3.h"
#include "numeric/mat4.h"

inline constexpr uint32_t InvalidBoneIndex{std::numeric_limits<uint32_t>::max()};

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
    void ResolveAnimationIndices(AnimationClip &) const; // Resolve TargetBoneId -> BoneIndex for all bone channels.
    void RecomputeRestWorld(); // Recompute RestWorld/InvRestWorld from RestLocal (no reordering).
    void RecomputeInverseBindMatrices(); // Update IBMs from current RestWorld after rest pose edits.
};

// Scene-facing armature object (name/transform/selection).
// References shared Armature (bones, IDs, imported skin metadata).
struct ArmatureObject {
    entt::entity Entity;
    std::vector<entt::entity> BoneEntities; // Bone instance entities, indexed by bone index.
    entt::entity JointEntity{null_entity}; // Shared joint sphere entity (billboard disc vertices).
};

// Tag on the shared joint sphere entity (excluded from silhouette and normal mesh iteration).
struct BoneJoint {};

// Component on joint sphere instance entities. Maps picks back to the bone.
struct BoneSubPartOf {
    entt::entity BoneEntity;
    bool IsTip; // false = root/head, true = tip/tail
};

// Component on bone entities pointing to their head/tail joint sphere instance entities.
struct BoneJointEntities {
    entt::entity Head{null_entity}, Tail{null_entity};
};

// The armature keeps its own Selected/Active throughout Edit/Pose mode.
struct BoneActive {};

// Needs deferred bone GPU instance state sync (selected/active colors).
struct BoneInstanceStateDirty {};

// Modifiers are optional, non-destructive object-level links that affect evaluated mesh output.
struct ArmatureModifier {
    entt::entity ArmatureEntity, ArmatureObjectEntity;
};

struct BoneIndex {
    uint32_t Index; // into Armature::Bones
};

// Display-only scale for bone sphere rendering. Kept separate from the ECS Scale component
// (which must stay vec3{1} so parent scale doesn't displace FK child positions).
struct BoneDisplayScale {
    float Value; // bone head-to-tail length; baked into mesh scale at GPU write time
};

// This is primarily for uncommon glTF joint+mesh nodes (direct per-bone attachment),
// not the common skinned-mesh ArmatureModifier deformation path.
struct BoneAttachment {
    entt::entity ArmatureEntity;
    BoneId Bone;
};

// Pose constraint stack on bone entities.
struct CopyTransformsData {};
struct ChildOfData {
    mat4 InverseMatrix{I4}; // Stored "parent-inverse" like Blender's Child Of: inverse(target_world) * owner_world at bind time.
};

struct BoneConstraint {
    entt::entity TargetEntity{null_entity};
    float Influence{1.f};
    std::variant<CopyTransformsData, ChildOfData> Data{CopyTransformsData{}};
};
struct BoneConstraints {
    std::vector<BoneConstraint> Stack;
};

struct ArmaturePoseState {
    std::vector<Transform> BonePoseDelta; // Animation delta from rest (identity = at rest). Persistent across frames.
    std::vector<Transform> BoneUserOffset; // Additive user offset per bone (identity = no offset). Applied on top of animation.
    std::vector<mat4> BonePoseWorld; // Per-bone pose world in armature-local space, post-constraint. Scratch, reused across frames.
    Range GpuDeformRange; // Allocation in shared ArmatureDeformBuffer arena. Count == 0 means not yet allocated.
};
struct ArmatureAnimation {
    std::vector<AnimationClip> Clips;
    uint32_t ActiveClipIndex{0};
};

struct MorphWeightAnimation {
    std::vector<MorphWeightClip> Clips;
    uint32_t ActiveClipIndex{0};
};
struct MorphWeightState {
    std::vector<float> Weights; // Current evaluated weights (CPU), size == target count
    Range GpuWeightRange; // Allocation in MorphWeightBuffer
};

// Returns armature entity if entity is an armature or a sub-element of one
entt::entity FindArmatureObject(const entt::registry &, entt::entity);

// Returns entt::null if no bone is active
entt::entity FindActiveBone(const entt::registry &);

void SelectBone(entt::registry &, entt::entity); // Exclusive

// Reset selected bones' transforms to rest pose.
void ClearSelectedBoneTransforms(entt::registry &, bool position, bool rotation, bool scale);

// Finalize armature structure after AddBone/RemoveBone. Resets pose state and re-resolves animation indices.
void RebuildArmatureStructure(entt::registry &, entt::entity arm_data_entity);

struct ExtrudeResult {
    std::vector<BoneId> NewBoneIds;
    std::vector<uint32_t> UpdatedParentIndices;
};
ExtrudeResult ExtrudeBones(entt::registry &, Armature &, entt::entity arm_obj_entity);

struct DuplicateResult {
    struct Entry {
        entt::entity OriginalEntity;
        BoneId NewId;
    };
    std::vector<Entry> Duplicated;
};
DuplicateResult DuplicateBones(entt::registry &, Armature &, entt::entity arm_obj_entity);

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
