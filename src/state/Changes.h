#pragma once

#include <cstdint>

namespace state {
// Reactive change sets, each bound to component events by its domain's registration.
enum class Change : uint8_t {
    // Selection and interaction
    Selected,
    ActiveInstance,
    BoneSelection,
    Rerecord,
    MeshActiveElement,
    InteractionMode,
    TransformPending,
    TransformEnd,
    TransformDirty,
    BonePose, // A bone's pose delta changed.
    WorldTransform,
    // Meshes
    MeshGeometry,
    MeshMaterial,
    MeshShading,
    TetMesh,
    NewBufferEntity,
    MeshPreview, // An entity's drawn record moved between its mesh and a preview.
    RenderInstanceCreated,
    RenderInstanceDestroyed, // A live entity lost its render instance.
    // Sound vertices
    SoundVertices,
    SoundVerticesUpdated,
    VertexForce,
    // Viewport and rendering
    ViewportDisplay,
    ViewportTheme,
    WorkspaceLights,
    Materials,
    PbrSpecialization,
    ActiveMaterialVariant,
    MaterializedTextures,
    StudioEnvironment,
    SceneWorld,
    PunctualLight,
    SceneView,
    CameraLens,
    AnimationEdited,
    MorphWeights, // Morph weights changed in the UMA weight buffer.
    // Physics
    PhysicsInput, // Any body, collider, joint, or hierarchy input.
    PhysicsTransform, // A local transform update, relevant when the entity poses a body, collider, or joint.
    PhysicsMaterialDef,
    CollisionSystemDef,
    CollisionFilterDef,
    PhysicsGeometry,
    PhysicsBodyMesh,
    ColliderPolicy,
    // Audio
    AudioVertexForce,
    ModalGain,
    ModalTuning,
    ModalSoundControls,
    RecordingStart,
    SoundVerticesDerivation,
    ContactReportingDerivation,
    ContactDynamicsDerivation,
    ModelRescaleEdit,
    AudioConfig,
    AudioMix,
    // Surface contact audio
    SurfaceEdit,
    SurfaceMaterial,
    SurfaceGeometry,
    SurfaceSoundControls,
    Count,
};
} // namespace state
