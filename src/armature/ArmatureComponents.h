#pragma once

#include "Range.h"
#include "armature/BoneId.h"
#include "gpu/Transform.h"
#include "numeric/mat4.h"
#include "state/Entity.h"

#include <vector>

struct AnimationClip;

struct ArmatureObject {
    state::Entity Entity;
    std::vector<state::Entity> BoneEntities;
    state::Entity JointEntity{state::Null};
};

// Tag on the shared joint sphere entity (excluded from silhouette and normal mesh iteration).
struct BoneJoint {};

struct BoneSubPartOf {
    state::Entity BoneEntity;
    bool IsTip;
};

struct BoneJointEntities {
    state::Entity Head{state::Null}, Tail{state::Null};
};

// Retain the armature's Selected/Active state throughout Edit/Pose mode.
struct BoneActive {};

struct ArmatureModifier {
    state::Entity ArmatureEntity, ArmatureObjectEntity;
    uint32_t SkinSlot{0};
};

struct BoneIndex {
    uint32_t Index;
};

// Separates display length from unit ECS scale so parent scaling cannot displace FK child positions.
struct BoneDisplayScale {
    float Value;
};

// Supports direct glTF joint and mesh nodes independently of skin deformation.
struct BoneAttachment {
    state::Entity ArmatureEntity;
    BoneId Bone;
};

// Canonical bone pose: each bone's local transform relative to rest (identity = at rest), by bone index.
// The persisted pose; bone entity Transforms are derived from this plus the rest pose.
struct ArmaturePose {
    std::vector<Transform> BoneDeltas;
};

// Derived from ArmaturePose plus the Armature rest pose.
struct ArmaturePoseState {
    std::vector<Transform> BoneUserOffset;
    std::vector<mat4> BonePoseWorld;
    std::vector<Range> GpuDeformRanges;
};
struct ArmatureAnimation {
    std::vector<AnimationClip> Clips;
    uint32_t ActiveClipIndex{0};
};
