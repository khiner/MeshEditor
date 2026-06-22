#pragma once

#include "Range.h"
#include "armature/BoneId.h"
#include "entt_fwd.h"
#include "gpu/Transform.h"
#include "numeric/mat4.h"

#include <vector>

struct AnimationClip;

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

// Derived from the bone entity Transforms plus the Armature rest pose.
// BonePoseDelta is the bone Transform relative to rest, BonePoseWorld and GpuDeformRange follow.
struct ArmaturePoseState {
    std::vector<Transform> BonePoseDelta; // Bone Transform relative to rest (identity = at rest)
    std::vector<Transform> BoneUserOffset; // Additive user offset per bone during an active drag
    std::vector<mat4> BonePoseWorld; // Per-bone pose world in armature-local space, post-constraint. Scratch, reused across frames.
    Range GpuDeformRange; // Allocation in shared ArmatureDeformBuffer arena. Count == 0 means not yet allocated.
};
struct ArmatureAnimation {
    std::vector<AnimationClip> Clips;
    uint32_t ActiveClipIndex{0};
};
