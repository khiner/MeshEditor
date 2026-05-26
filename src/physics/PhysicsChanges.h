#pragma once

// Reactive-tracker tag types for physics, used with reactive<>() helpers in Reactive.h.
// clang-format off
namespace changes {
struct PhysicsMotion {}; struct PhysicsShape {}; struct PhysicsMaterial {}; struct PhysicsTrigger {}; struct PhysicsJoint {};
struct PhysicsMaterialDef {}; struct CollisionSystemDef {}; struct CollisionFilterDef {}; struct PhysicsJointDef {};
struct ColliderPolicy {}; struct PhysicsSimulationSettings {};
} // namespace changes
// clang-format on
