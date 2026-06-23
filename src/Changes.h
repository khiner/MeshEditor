#pragma once

// clang-format off
namespace changes {
struct Selected {}; struct ActiveInstance {}; struct BoneSelection {}; struct Rerecord {};
struct MeshActiveElement {}; struct MeshGeometry {}; struct MeshMaterial {};
struct SoundVertices {}; struct SoundVerticesUpdated {}; struct VertexForce {};
struct NewBufferEntity {}; struct RenderInstanceCreated {}; struct ObjectCreated {};
struct ViewportDisplay {}; struct InteractionMode {}; struct WorkspaceLights {}; struct Rotation {};
struct ViewportTheme {}; struct Materials {}; struct PbrSpecialization {}; struct ActiveMaterialVariant {};
struct MaterializedTextures {}; struct StudioEnvironment {}; struct SceneWorld {}; struct PunctualLight {};
struct SceneView {}; struct CameraLens {}; struct TransformPending {};
struct TransformEnd {}; struct WorldTransform {}; struct TransformDirty {};
struct TimelineRange {};
} // namespace changes
// clang-format on
