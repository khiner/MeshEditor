#pragma once

// Reactive-tracker tag types used with reactive<>() helpers in Reactive.h.
// clang-format off
namespace changes {
struct Selected {}; struct ActiveInstance {}; struct BoneSelection {}; struct Rerecord {};
struct MeshActiveElement {}; struct MeshGeometry {}; struct MeshMaterial {};
struct SoundVertices {}; struct SoundVerticesUpdated {}; struct VertexForce {}; struct ModelsBuffer {};
struct NewBufferEntity {}; struct RenderInstanceCreated {};
struct ViewportDisplay {}; struct InteractionMode {}; struct Submit {}; struct Rotation {};
struct ViewportTheme {}; struct Materials {}; struct PbrSpecialization {}; struct ActiveMaterialVariant {};
struct SceneView {}; struct CameraLens {}; struct TransformPending {};
struct TransformEnd {}; struct WorldTransform {}; struct TransformDirty {};
struct PhysicsMotion {}; struct PhysicsShape {}; struct PhysicsMaterial {}; struct PhysicsTrigger {}; struct PhysicsJoint {};
struct PhysicsMaterialDef {}; struct CollisionSystemDef {}; struct CollisionFilterDef {}; struct PhysicsJointDef {};
struct ColliderPolicy {}; struct PhysicsSimulationSettings {};
struct TimelineRange {};
} // namespace changes
// clang-format on
