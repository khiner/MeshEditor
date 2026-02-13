#pragma once

#include "Transform.h"
#include "entt_fwd.h"
#include "numeric/mat4.h"
#include "vulkan/BufferArena.h"

#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using BoneId = uint32_t; // Stable identifier - never reused

inline constexpr BoneId InvalidBoneId{0};
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

struct ArmatureData {
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
    void SetJointNodeMapping(BoneId, uint32_t joint_node_index);
    BoneId AddBone(std::string_view name, std::optional<BoneId> parent_bone_id, const Transform &rest_local, std::optional<uint32_t> joint_node_index);
    void FinalizeStructure(); // Validate/canonicalize + rebuild derived runtime caches.
};

// Scene-facing armature object (name/transform/selection).
// References shared ArmatureData (bones, IDs, imported skin metadata).
struct ArmatureObject {
    entt::entity DataEntity;
};

// Armature modifier linkage for mesh instances deformed by an armature.
// Modifiers are optional, non-destructive object-level links that affect evaluated
// mesh output (for armatures: skinned vertex deformation).
struct ArmatureModifier {
    entt::entity ArmatureDataEntity;
    entt::entity ArmatureObjectEntity;
};

// Component on scene objects attached to a specific armature bone.
// Source skin/node references remain recoverable via ArmatureData::ImportedSkin and BoneId mappings.
// This is primarily for uncommon glTF joint+mesh nodes (direct per-bone attachment),
// not the common skinned-mesh deformation path handled by ArmatureModifier.
struct BoneAttachment {
    entt::entity ArmatureDataEntity;
    BoneId Bone;
};

enum class AnimationPath : uint8_t {
    Translation,
    Rotation,
    Scale,
    Weights
};
enum class AnimationInterpolation : uint8_t {
    Step,
    Linear,
    CubicSpline
};

// Resolved animation channel (bone-index-based, not node-index).
struct AnimationChannel {
    uint32_t BoneIndex;
    AnimationPath Target;
    AnimationInterpolation Interp{AnimationInterpolation::Linear};
    std::vector<float> TimesSeconds;
    std::vector<float> Values;
};

struct AnimationClip {
    std::string Name;
    float DurationSeconds{0};
    std::vector<AnimationChannel> Channels;
};

// Component on armature data entities with imported skin data.
struct ArmaturePoseState {
    std::vector<Transform> BonePoseLocal; // CPU working storage (per-bone animated local transform)
    Range GpuDeformRange; // Allocation in shared ArmatureDeformBuffer arena
    bool Dirty{true};
};

// Component on armature data entities with animation data.
struct ArmatureAnimationData {
    std::vector<AnimationClip> Clips;
    uint32_t ActiveClipIndex{0};
};

// Morph weight animation channel — dedicated type (no BoneIndex/Target overhead).
struct MorphWeightChannel {
    AnimationInterpolation Interp{AnimationInterpolation::Linear};
    std::vector<float> TimesSeconds;
    std::vector<float> Values; // Packed: target_count floats per keyframe
};

struct MorphWeightClip {
    std::string Name;
    float DurationSeconds{0};
    std::vector<MorphWeightChannel> Channels;
};

// Component on mesh instance entities with morph target animation data.
struct MorphWeightAnimationData {
    std::vector<MorphWeightClip> Clips;
    uint32_t ActiveClipIndex{0};
};

// Component on mesh instance entities with morph targets (animated or static).
// DefaultWeights and TargetCount are mesh-level data in MeshStore::Entry.
struct MorphWeightState {
    std::vector<float> Weights; // Current evaluated weights (CPU), size == target count
    Range GpuWeightRange; // Allocation in MorphWeightBuffer
};

// Interpolate animation channels at `time`, writing into pre-initialized rest-pose `local_transforms`.
void EvaluateAnimation(const AnimationClip &, float time, std::span<Transform> local_transforms);

// Interpolate morph weight animation channels at `time`, writing into `weights`.
void EvaluateMorphWeights(const MorphWeightClip &clip, float time, std::span<float> weights);

// Compute final deform matrices from posed local transforms + inverse bind matrices.
// Writes directly into `out` (mapped GPU memory).
void ComputeDeformMatrices(const ArmatureData &, std::span<const Transform> local_transforms, std::span<const mat4> inverse_bind, std::span<mat4> out);
