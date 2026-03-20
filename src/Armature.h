#pragma once

#include "AnimationData.h"
#include "Transform.h"
#include "entt_fwd.h"
#include "numeric/mat3.h"
#include "numeric/mat4.h"
#include "vulkan/Range.h"

#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

inline constexpr uint32_t InvalidBoneIndex{std::numeric_limits<uint32_t>::max()};

struct ArmatureBone {
    BoneId Id;
    BoneId ParentBoneId; // Stable structural parent relation
    std::optional<uint32_t> JointNodeIndex; // Optional imported glTF joint node binding
    std::string Name;
    Transform RestLocal;
    mat4 RestWorld{I4}, InvRestWorld{I4};
    uint32_t ParentIndex{InvalidBoneIndex}, FirstChild{InvalidBoneIndex}, NextSibling{InvalidBoneIndex}; // Dense runtime caches
};

struct ArmatureImportedSkin {
    uint32_t SkinIndex;
    std::optional<uint32_t> SkeletonNodeIndex;
    std::optional<uint32_t> AnchorNodeIndex;
    std::vector<uint32_t> OrderedJointNodeIndices;
    std::vector<mat4> InverseBindMatrices;
};

struct Armature {
    uint32_t Version{1}; // Increments when structure changes
    BoneId NextBoneId{1}; // Monotonic allocator state
    bool Dirty{false}; // True when structural edits need finalization.

    std::vector<ArmatureBone> Bones;
    std::unordered_map<BoneId, uint32_t> BoneIdToIndex;
    std::unordered_map<uint32_t, BoneId> JointNodeIndexToBoneId; // Derived cache from ArmatureBone::JointNodeIndex
    std::optional<ArmatureImportedSkin> ImportedSkin;

    BoneId AllocateBoneId();
    std::optional<uint32_t> FindBoneIndex(BoneId) const;
    std::optional<uint32_t> FindJointNodeIndex(BoneId) const;
    std::optional<BoneId> FindBoneIdByJointNodeIndex(uint32_t) const;
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
    entt::entity Head{null_entity};
    entt::entity Tail{null_entity};
};

// Bone-level selection — independent of object-level Selected/Active.
// The armature keeps its own Selected/Active throughout Edit/Pose mode.
struct BoneActive {};

// Bone selection part identifier.
enum class BoneSel : uint8_t {
    Root,
    Tip,
    Body
};

// Presence means the bone is selected. Fields indicate which parts are selected.
struct BoneSelection {
    bool Root{true}, Tip{true}, Body{true};

    static BoneSelection From(BoneSel part) { return {part != BoneSel::Tip, part != BoneSel::Root, part == BoneSel::Body}; }
    BoneSelection operator|(BoneSelection o) const {
        const bool r = Root || o.Root, t = Tip || o.Tip;
        return {r, t, Body || o.Body || (r && t)};
    }
    bool Has(BoneSel part) const { return part == BoneSel::Root ? Root : part == BoneSel::Tip ? Tip :
                                                                                                Body; }
};

// Armature modifier linkage for mesh instances deformed by an armature.
// Modifiers are optional, non-destructive object-level links that affect evaluated
// mesh output (for armatures: skinned vertex deformation).
struct ArmatureModifier {
    entt::entity ArmatureEntity;
    entt::entity ArmatureObjectEntity;
};

// Component on bone instance entities. Index into Armature::Bones.
struct BoneIndex {
    uint32_t Index;
};

// Display-only scale for bone sphere rendering. Kept separate from the ECS Scale component
// (which must stay vec3{1} so parent scale doesn't displace FK child positions).
// Applied only at GPU write time in UpdateModelBuffer.
struct BoneDisplayScale {
    float Value; // bone head-to-tail length; baked into mesh scale at GPU write time
};

// Component on scene objects attached to a specific armature bone.
// Source skin/node references remain recoverable via Armature::ImportedSkin and BoneId mappings.
// This is primarily for uncommon glTF joint+mesh nodes (direct per-bone attachment),
// not the common skinned-mesh deformation path handled by ArmatureModifier.
struct BoneAttachment {
    entt::entity ArmatureEntity;
    BoneId Bone;
};

// Component on armature data entities with imported skin data.
struct ArmaturePoseState {
    std::vector<Transform> BonePoseDelta; // Animation delta from rest (identity = at rest). Persistent across frames.
    std::vector<Transform> BoneUserOffset; // Additive user offset per bone (identity = no offset). Applied on top of animation.
    Range GpuDeformRange; // Allocation in shared ArmatureDeformBuffer arena
    bool Dirty{true};
};

// Component on armature data entities with animation data.
struct ArmatureAnimation {
    std::vector<AnimationClip> Clips;
    uint32_t ActiveClipIndex{0};
};

// Component on mesh instance entities with morph target animation data.
struct MorphWeightAnimation {
    std::vector<MorphWeightClip> Clips;
    uint32_t ActiveClipIndex{0};
};

// Component on mesh instance entities with morph targets (animated or static).
// DefaultWeights and TargetCount are mesh-level data in MeshStore::Entry.
struct MorphWeightState {
    std::vector<float> Weights; // Current evaluated weights (CPU), size == target count
    Range GpuWeightRange; // Allocation in MorphWeightBuffer
};

// Given an instance entity, return its armature object entity (if it is one, or if it's a sub-element of one).
entt::entity FindArmatureObject(const entt::registry &, entt::entity);

// Return the single BoneActive entity, or entt::null if none.
entt::entity FindActiveBone(const entt::registry &);

// Compose a rest-pose transform with a delta (Blender-style pose channel model).
// delta identity {P=0, R=identity, S=1} produces rest unchanged.
Transform ComposeWithDelta(const Transform &rest, const Transform &delta);

// Convert an absolute local transform to a delta relative to rest.
Transform AbsoluteToDelta(const Transform &rest, const Transform &absolute);

// Evaluate animation into delta space. For each keyed channel, interpolates the absolute glTF
// keyframe value and converts to a rest-relative delta. Unkeyed components are left unchanged.
void EvaluateAnimationDeltas(const AnimationClip &, float time, std::span<const ArmatureBone>, std::span<Transform> deltas);

// Compute final deform matrices from rest poses + pose deltas + user offsets + inverse bind matrices.
// Effective delta per bone = ComposeWithDelta(pose_delta, user_offset). Writes directly into `out` (mapped GPU memory).
void ComputeDeformMatrices(const Armature &, std::span<const Transform> pose_deltas, std::span<const Transform> user_offsets, std::span<const mat4> inverse_bind, std::span<mat4> out);

// Bone direction/roll ↔ rotation matrix conversion (ported from Blender).
// Direction is the normalized bone Y axis (head→tail). Roll is twist around that axis.
mat3 BoneVecRollToMat3(vec3 direction, float roll);
void BoneMat3ToVecRoll(const mat3 &m, vec3 &direction, float &roll);
