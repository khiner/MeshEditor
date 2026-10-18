#include "physics/ColliderUpdate.h"
#include "physics/PhysicsTypes.h"
#include "snapshot/SnapshotRegistration.h"

namespace snapshot::detail {

void RegisterPhysics(Tables &tables) {
    Classify(
        tables,
        TypeList<
            PhysicsSimulationSettings, PhysicsMaterial, CollisionSystem, CollisionFilter, PhysicsJointDef, PhysicsMotion,
            ColliderShape, ColliderMaterial, ColliderPolicy, PhysicsVelocity, TriggerTag, TriggerNodes, PhysicsJoint>{},
        TypeList<PhysicsBodyHandle, PhysicsConstraintHandle, BodyPoseCache, MeshColliderList, ColliderListedUnder>{}
    );
}
} // namespace snapshot::detail
