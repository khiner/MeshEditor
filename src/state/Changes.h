#pragma once

#include <cstdint>

namespace state {
// Reactive change sets, each bound to component events by its domain's registration.
enum class Change : uint8_t {
    // Selection and interaction
    Selected,
    ActiveInstance,
    BoneSelection,
    MeshActiveElement,
    InteractionMode,
    TransformPending,
    TransformStart,
    TransformEnd,
    TransformDirty,
    BonePose, // A bone's pose delta changed.
    BoneConstraints, // A bone's constraint stack changed.
    WorldTransform, // A world transform or a bone's display scale changed.
    SceneParent, // A node's parent changed.
    SceneHierarchy, // A scene graph link changed.
    Names, // A name was created or destroyed.
    ScaleLocked,
    EditMode,
    KeyframeSources, // The timeline rate or a mesh's displayed material slot changed.
    // Meshes
    MeshGeometry,
    MeshMaterial,
    PrimitiveShape, // A primitive mesh's shape fields changed.
    TetMesh,
    NewBufferEntity,
    InstanceVisibility, // An Instance or Hidden change that the settle pass derives RenderInstance from.
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
    PbrSpecialization, // The lighting the PBR pipelines specialize on changed.
    PbrMeshFeatures, // A mesh's PBR features changed.
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
    Colliders, // A collider shape was created, changed or destroyed.
    PhysicsDefinitionUses, // A reference to a physics material, collision system, collision filter, or joint definition changed.
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
