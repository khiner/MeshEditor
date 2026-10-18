#pragma once

#include "state/Slot.h"

struct Armature;
struct ArmatureChanges;
struct ArmatureObject;
struct BoneJoint;
struct BoneSubPartOf;
struct BoneJointEntities;
struct BoneActive;
struct ArmatureModifier;
struct BoneIndex;
struct BoneDisplayScale;
struct BoneAttachment;
struct BoneDelta;
struct ArmaturePoseState;
struct BoneConstraints;
struct ConstraintTargets;

using ArmatureSchema = state::Schema<state::ArmatureBase, state::ArmatureCount, Armature, ArmatureChanges, ArmatureObject, BoneJoint, BoneSubPartOf, BoneJointEntities, BoneActive, ArmatureModifier, BoneIndex, BoneDisplayScale, BoneAttachment, BoneDelta, ArmaturePoseState, BoneConstraints, ConstraintTargets>;
template<typename T>
    requires ArmatureSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = ArmatureSchema::Of<T>();
