#include "animation/AnimationTimeline.h"
#include "physics/PhysicsChanges.h"
#include "physics/PhysicsContact.h"
#include "physics/PhysicsTypes.h"
#include "snapshot/SnapshotRegistration.h"

namespace snapshot::detail {
template<> inline constexpr bool ForceFieldwise<ColliderShape> = true;
template<> inline constexpr bool ForceFieldwise<PhysicsJoint> = true;
template<> inline constexpr bool ForceFieldwise<PhysicsMotion> = true;

void RegisterPhysics(Tables &tables) {
    Persistent<
        PhysicsSimulationSettings, PhysicsMaterial, CollisionSystem, CollisionFilter, PhysicsJointDef, PhysicsMotion,
        ColliderShape, ColliderMaterial, ColliderPolicy, PhysicsVelocity, TriggerTag, TriggerNodes, PhysicsJoint>(tables);
    Derived<PhysicsBodyHandle, PhysicsConstraintHandle, BodyPoseCache, PhysicsCacheInvalid>(tables);
}
} // namespace snapshot::detail
