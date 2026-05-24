#pragma once

#include "ObjectCreateInfo.h"
#include "TransformGizmoTypes.h"
#include "ViewCamera.h"
#include "entt_fwd.h"
#include "gpu/DebugChannel.h"
#include "gpu/Element.h"
#include "gpu/InteractionMode.h"

#include <entt/entity/registry.hpp>
#include <filesystem>
#include <set>
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

// Component on the viewport singleton entity. Changes require command buffer re-recording.
struct ViewportDisplay {
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

inline const PBRViewportLighting &GetActivePbrLighting(const entt::registry &r, entt::entity viewport, ViewportShadingMode mode) {
    return mode == ViewportShadingMode::Rendered ? static_cast<const PBRViewportLighting &>(r.get<const RenderedLighting>(viewport)) : static_cast<const PBRViewportLighting &>(r.get<const MaterialPreviewLighting>(viewport));
}

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

// Singleton flags on viewport, consumed and cleared by ProcessComponentEvents.
struct ProfileNextProcessComponentEvents {}; // Profile the next ProcessComponentEvents pass.

// Smooth float frame position for playback, advanced by Render. Singleton on viewport.
struct PlaybackFrame {
    float Value{1.0f};
};
// Last frame where armature poses were evaluated. Singleton on viewport.
struct LastEvaluatedFrame {
    int Value{-1};
};
// Requested transform type for the next gizmo drag, latched by keyboard shortcuts.
// Presence == active latch; removed by InteractOverlay after consumption. Singleton on viewport.
struct StartScreenTransform {
    TransformGizmo::TransformType Value;
};

// Interaction modes available for cycling/selection. Singleton on viewport.
// Excite is added/removed reactively based on whether the scene has any SoundVertices.
struct EnabledInteractionModes {
    std::set<InteractionMode> Value{InteractionMode::Object, InteractionMode::Edit, InteractionMode::Pose};
};

struct PendingSetStudioEnvironment {
    uint32_t Index;
};
struct PendingSetEditMode {
    Element Mode;
};

struct PendingShaderRecompile {};

enum class ColliderShapeBuffer : uint8_t { Box,
                                           Sphere,
                                           CapsuleCap,
                                           Circle,
                                           Line,
                                           Count };
struct ColliderShapeBuffers {
    std::array<entt::entity, std::size_t(ColliderShapeBuffer::Count)> Entities{
        null_entity, null_entity, null_entity, null_entity, null_entity
    };
};

// Pending mesh import (file load + texture uploads).
// Apply emits this; ProcessComponentEvents performs the GPU work, then removes it.
struct PendingImportMesh {
    std::filesystem::path Path;
    MeshInstanceCreateInfo Info;
};

// Forces a fresh Rebuild on the next tick even if the start frame is cached.
// Emitted by JumpToStart and LoadGltf; cleared after the cache is cleared.
struct PhysicsCacheInvalid {};

// One-shot GPU sync primitives for synchronous passes (selection compute, element pick,
// glTF load, texture uploads, etc.). Owns its Vk resources directly; the render-pipeline
// command buffers are allocated from `Pool` and freed before this component is destroyed.
struct OneShotGpu {
    vk::UniqueCommandPool Pool;
    vk::UniqueCommandBuffer Cb;
    vk::UniqueFence Fence;
    vk::UniqueSemaphore SelectionReady;
};

inline OneShotGpu MakeOneShotGpu(vk::Device device, uint32_t queue_family) {
    auto pool = device.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queue_family});
    auto cb = std::move(device.allocateCommandBuffersUnique({*pool, vk::CommandBufferLevel::ePrimary, 1}).front());
    return {
        .Pool = std::move(pool),
        .Cb = std::move(cb),
        .Fence = device.createFenceUnique({}),
        .SelectionReady = device.createSemaphoreUnique({}),
    };
}

// In Edit/Excite mode, orbit camera to active element on selection change.
struct OrbitToActive {
    bool Value{false};
};

struct TransformGizmoState {
    TransformGizmo::Config Config;
    TransformGizmo::Mode Mode;
};

struct BBoxWireframe {
    entt::entity Instance{null_entity};
};
struct TetWireframe {
    entt::entity Instance{null_entity};
};

// At most one camera carries this component at a time.
struct LookingThrough {
    ViewCamera SavedViewCamera; // The pre-look-through ViewCamera, restored on exit.
};
