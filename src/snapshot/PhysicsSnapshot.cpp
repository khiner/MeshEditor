#include "animation/AnimationTimeline.h"
#include "physics/PhysicsContact.h"
#include "physics/PhysicsTypes.h"
#include "snapshot/SnapshotRegistration.h"

namespace snapshot::detail {

void RegisterPhysics(Tables &tables) {
    Persistent<
        PhysicsSimulationSettings, PhysicsMaterial, CollisionSystem, CollisionFilter, PhysicsJointDef, PhysicsMotion,
        ColliderShape, ColliderMaterial, ColliderPolicy, PhysicsVelocity, TriggerTag, TriggerNodes, PhysicsJoint>(tables);
    Derived<PhysicsBodyHandle, PhysicsConstraintHandle, BodyPoseCache>(tables);
}
} // namespace snapshot::detail
