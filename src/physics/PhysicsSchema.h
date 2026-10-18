#pragma once

#include "state/Slot.h"

struct MeshColliderList;
struct ColliderListedUnder;
struct PhysicsChanges;
struct ReportContacts;
struct PhysicsContactImpacts;
struct PhysicsSustainedContacts;
struct PhysicsState;
struct PhysicsDefinitionUses;
struct PhysicsSimulationSettings;
struct PhysicsMaterial;
struct CollisionSystem;
struct CollisionFilter;
struct PhysicsJointDef;
struct PhysicsMotion;
struct PhysicsVelocity;
struct ColliderShape;
struct ColliderMaterial;
struct TriggerTag;
struct ColliderPolicy;
struct TriggerNodes;
struct PhysicsJoint;
struct PhysicsBodyHandle;
struct PhysicsConstraintHandle;
struct BodyPoseCache;

using PhysicsSchema = state::Schema<state::PhysicsBase, state::PhysicsCount, MeshColliderList, ColliderListedUnder, PhysicsChanges, ReportContacts, PhysicsContactImpacts, PhysicsSustainedContacts, PhysicsState, PhysicsDefinitionUses, PhysicsSimulationSettings, PhysicsMaterial, CollisionSystem, CollisionFilter, PhysicsJointDef, PhysicsMotion, PhysicsVelocity, ColliderShape, ColliderMaterial, TriggerTag, ColliderPolicy, TriggerNodes, PhysicsJoint, PhysicsBodyHandle, PhysicsConstraintHandle, BodyPoseCache>;
template<typename T>
    requires PhysicsSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = PhysicsSchema::Of<T>();
