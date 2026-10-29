#pragma once

#include "Range.h"
#include "armature/ArmatureSchema.h"
#include "armature/BoneId.h"
#include "gpu/Transform.h"
#include "state/Entity.h"

struct ArmatureObject {
    state::Entity Entity;
    std::vector<state::Entity> BoneEntities;
    state::Entity JointEntity{state::Null};
};

// On the shared joint sphere entity (excluded from silhouette and normal mesh iteration), naming the armature object whose joints it draws.
struct BoneJoint {
    state::Entity ArmatureObjectEntity;
};

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

// A bone entity's persisted pose relative to its rest pose.
// Identity is at rest.
struct BoneDelta {
    Transform Value;
};

// Derived on each armature object from its bone deltas and the shared rest pose.
struct ArmaturePoseState {
    std::vector<Transform> BoneUserOffset;
    std::vector<mat4> BonePoseWorld;
    std::vector<Range> GpuDeformRanges;
};
