#pragma once

#include "BoneSelection.h"
#include "SceneModeComponents.h"
#include "SceneOps.h"
#include "TransformGizmo.h"
#include "gpu/DebugChannel.h"
#include "numeric/vec2.h"

#include <entt/entity/fwd.hpp>
#include <filesystem>
#include <optional>
#include <set>
#include <span>
#include <vulkan/vulkan.hpp>

struct SubmitDirty {}; // Generic tag for events that only require command buffer submission (not re-record)
struct MeshGeometryDirty {}; // Request overlay + element-state buffer refresh after mesh geometry changes
struct LightWireframeDirty {};
struct MaterialDirty {
    uint32_t Index{0};
};
struct MeshMaterialAssignment {
    uint32_t PrimitiveIndex, MaterialIndex;
};
struct MeshMaterialSlotSelection {
    uint32_t PrimitiveIndex{0};
};

// Presence indicates active transform
struct PendingTransform {
    vec3 Pivot{};
    quat PivotR{1, 0, 0, 0};
    Transform Delta{};
};

enum class ViewportShadingMode : uint8_t {
    Wireframe,
    Solid,
    MaterialPreview,
    Rendered,
};

// Component on the scene singleton entity. Changes require command buffer re-recording.
struct SceneSettings {
    ViewportShadingMode ViewportShading{ViewportShadingMode::Solid};
    ViewportShadingMode FillMode{ViewportShadingMode::Solid}; // last non-wireframe mode (for Shift+Z toggle)
    vk::ClearColorValue ClearColor{0.25f, 0.25f, 0.25f, 1.f};
    bool ShowGrid{true}, ShowBoundingBoxes{false}, ShowTetWireframe{false};
    bool ShowExtras{true}, ShowBones{true}, ShowOrigins{true}, ShowOutlineSelected{true};
    bool ShowOverlays{true}; // Master toggle for all overlays
    uint8_t NormalOverlays{0}; // Bitmask of Element
    DebugChannel DebugChannel{DebugChannel::None};
};

// Scene lights/world toggles + studio env controls
struct PBRViewportLighting {
    bool UseSceneLights, UseSceneWorld;
    float EnvIntensity, EnvRotationDegrees;
    float BackgroundBlur{0.5f}, WorldOpacity{0.f};
    // Render the scene into a transmission framebuffer (with mips) and sample it at the
    // refracted ray exit point, instead of approximating refraction by sampling the IBL.
    bool RealTransmission{true};
};

// Two distinct ECS component types sharing the same layout, with different defaults
struct MaterialPreviewLighting : PBRViewportLighting {}; // defaults: both OFF (studio HDRI)
struct RenderedLighting : PBRViewportLighting {}; // defaults: both ON (scene world/lights)

struct ViewportExtent {
    vk::Extent2D Value{};
};

// Present iff a loaded glTF declared variants.
// Empty Active means no variant active - each primitive shows its source-default material
// (per spec, also applied per-primitive when the active variant has no mapping).
struct MaterialVariants {
    std::vector<std::string> Names;
    std::optional<uint32_t> Active;
};

// Per-mesh-entity: bitmask of PbrFeature bits that are explicitly enabled for that mesh.
// Scene-wide mask = OR of all PbrMeshFeatures + Punctual bit from "Use Scene Lights".
struct PbrMeshFeatures {
    uint32_t Mask{0};
};

// Snapshot of selection state at the start of a shift+box-drag.
// Presence on SceneEntity means an additive box-drag is active.
struct AdditiveBoxSelectBaseline {
    std::vector<entt::entity> SelectedEntities;
    std::vector<std::pair<entt::entity, BoneSelection>> BoneSelections;
    std::vector<uint32_t> ElementBitset;
};

// Singleton flags on SceneEntity, consumed and cleared by ProcessComponentEvents.
struct SelectionBitsDirty {}; // Bitset written by Interact; dispatches the compute update.
struct ElementStatesDirty {}; // Element state buffers updated by GPU compute; triggers a submit.
struct ProfileNextProcessComponentEvents {}; // Profile the next ProcessComponentEvents pass.
struct SelectionStale {}; // Selection fragment data no longer matches current scene. Cleared after RenderSelectionPass.

// Smooth float frame position for playback, advanced by Render. Singleton on SceneEntity.
struct PlaybackFrame {
    float Value{1.0f};
};
// Last frame where armature poses were evaluated. Singleton on SceneEntity.
struct LastEvaluatedFrame {
    int Value{-1};
};
// Requested transform type for the next gizmo drag, latched by keyboard shortcuts.
// Cleared by InteractOverlay after consumption. Singleton on SceneEntity.
struct StartScreenTransform {
    std::optional<TransformGizmo::TransformType> Value;
};

// Interaction modes available for cycling/selection. Singleton on SceneEntity.
// Excite is added/removed reactively based on whether the scene has any SoundVertices.
struct EnabledInteractionModes {
    std::set<InteractionMode> Value{InteractionMode::Object, InteractionMode::Edit, InteractionMode::Pose};
};

// Pending edit-mode element click. Apply emits this; ProcessComponentEvents runs the GPU pick and applies bit/active updates, then removes it.
struct PendingEditElementClick {
    uvec2 MousePx;
    bool Toggle;
};

// Pending HDR prefilter / activation. Apply emits this; ProcessComponentEvents prefilters and activates, then removes it.
struct PendingSetStudioEnvironment {
    uint32_t Index;
};

// Pending edit-element-mode change. Apply emits this; ProcessComponentEvents performs the bitset conversion + GPU compute dispatch, then removes it.
struct PendingSetEditMode {
    Element Mode;
};

// Pending mesh import (file load + texture uploads). Apply emits this; ProcessComponentEvents performs the GPU work, then removes it.
struct PendingImportMesh {
    std::filesystem::path Path;
    MeshInstanceCreateInfo Info;
};

// Forces a fresh Rebuild on the next tick even if the start frame is cached.
// Emitted by JumpToStart and LoadGltf; cleared after the cache is cleared.
struct PhysicsCacheInvalid {};

// Non-owning span over the GPU-mapped SelectionBitset words. Lets Apply read/write the bitset
// without depending on SceneStores. Initialized once by Scene; the underlying storage is stable.
struct SelectionBitsetRef {
    std::span<uint32_t> Value;
};

// Descriptor-slot IDs for the selection compute/render pipeline.
// RAII for the slots lives in Scene; this component publishes the stable IDs.
struct SelectionSlots {
    uint32_t HeadImage;
    uint32_t SelectionCounter;
    uint32_t ElementPickCandidates;
    uint32_t SelectionBitset;
};

// Shared one-shot GPU sync primitives for synchronous passes (selection compute,
// element pick, glTF load, etc.). RAII for the underlying Vulkan resources lives
// in Scene; raw handles published here so registry-only helpers can read them.
struct SceneOneShotGpu {
    vk::CommandPool Pool;
    vk::CommandBuffer Cb;
    vk::Fence Fence;
    vk::Semaphore SelectionReady;
};

// X-ray (occlusion-ignoring) selection toggle.
// Kept distinct from SceneSettings so toggling does not trip the SceneSettings
// reactive handler that forces a full draw-list re-record.
struct SelectionXRay {
    bool Value{false};
};
