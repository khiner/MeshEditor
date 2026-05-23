#pragma once

#include "Bindless.h"
#include "BoneSelection.h"
#include "SceneModeComponents.h"
#include "SceneOps.h"
#include "TransformGizmo.h"
#include "ViewCamera.h"
#include "entt_fwd.h"
#include "gpu/DebugChannel.h"
#include "numeric/vec2.h"

#include <array>
#include <entt/entity/fwd.hpp>
#include <entt/entity/registry.hpp>
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

// A contiguous span of a mesh's elements (vertices/edges/faces) in the SelectionBitset.
struct ElementRange {
    entt::entity MeshEntity;
    uint32_t Offset, Count;
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

inline const PBRViewportLighting &GetActivePbrLighting(const entt::registry &r, entt::entity scene_entity, ViewportShadingMode mode) {
    return mode == ViewportShadingMode::Rendered ? static_cast<const PBRViewportLighting &>(r.get<const RenderedLighting>(scene_entity)) : static_cast<const PBRViewportLighting &>(r.get<const MaterialPreviewLighting>(scene_entity));
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
// Presence == active latch; removed by InteractOverlay after consumption. Singleton on SceneEntity.
struct StartScreenTransform {
    TransformGizmo::TransformType Value;
};

// Interaction modes available for cycling/selection. Singleton on SceneEntity.
// Excite is added/removed reactively based on whether the scene has any SoundVertices.
struct EnabledInteractionModes {
    std::set<InteractionMode> Value{InteractionMode::Object, InteractionMode::Edit, InteractionMode::Pose};
};

struct PendingEditElementClick {
    uvec2 MousePx;
    bool Toggle;
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

// Non-owning span over the GPU-mapped SelectionBitset words.
// Lets Apply read/write the bitset without depending on SceneStores.
// Initialized once by Scene; the underlying storage is stable.
struct SelectionBitsetRef {
    std::span<uint32_t> Value;
};

// RAII for the descriptor-slot leases used by the selection compute/render pipeline.
struct SelectionSlots {
    uint32_t HeadImage{}, SelectionCounter{}, ObjectPickKey{}, ElementPickCandidates{}, ObjectPickSeenBits{}, SelectionBitset{};
    uint32_t ObjectIdSampler{}, DepthSampler{}, SilhouetteSampler{}, ColorSampler{}, LineDataSampler{}, TransmissionSampler{};

    using Entry = std::pair<SlotType, uint32_t SelectionSlots::*>;
    static constexpr std::array<Entry, 12> Entries{{
        {SlotType::Image, &SelectionSlots::HeadImage},
        {SlotType::Buffer, &SelectionSlots::SelectionCounter},
        {SlotType::Buffer, &SelectionSlots::ObjectPickKey},
        {SlotType::Buffer, &SelectionSlots::ElementPickCandidates},
        {SlotType::Buffer, &SelectionSlots::ObjectPickSeenBits},
        {SlotType::Buffer, &SelectionSlots::SelectionBitset},
        {SlotType::Sampler, &SelectionSlots::ObjectIdSampler},
        {SlotType::Sampler, &SelectionSlots::DepthSampler},
        {SlotType::Sampler, &SelectionSlots::SilhouetteSampler},
        {SlotType::Sampler, &SelectionSlots::ColorSampler},
        {SlotType::Sampler, &SelectionSlots::LineDataSampler},
        {SlotType::Sampler, &SelectionSlots::TransmissionSampler},
    }};

    explicit SelectionSlots(DescriptorSlots &slots) : Slots(&slots) {
        for (const auto &[type, field] : Entries) this->*field = slots.Allocate(type);
    }
    SelectionSlots(const SelectionSlots &) = delete;
    SelectionSlots &operator=(const SelectionSlots &) = delete;
    SelectionSlots(SelectionSlots &&o) noexcept : Slots(o.Slots) {
        for (const auto &[_, field] : Entries) this->*field = o.*field;
        o.Slots = nullptr;
    }
    SelectionSlots &operator=(SelectionSlots &&o) noexcept {
        if (this != &o) {
            Release();
            Slots = o.Slots;
            for (const auto &[_, field] : Entries) this->*field = o.*field;
            o.Slots = nullptr;
        }
        return *this;
    }
    ~SelectionSlots() { Release(); }

private:
    DescriptorSlots *Slots{nullptr};
    void Release() {
        if (!Slots) return;
        for (const auto &[type, field] : Entries) Slots->Release({type, this->*field});
        Slots = nullptr;
    }
};

// One-shot GPU sync primitives for synchronous passes (selection compute, element pick,
// glTF load, texture uploads, etc.). Owns its Vk resources directly; Scene's render-pipeline
// command buffers are allocated from `Pool` and freed before this component is destroyed.
struct SceneOneShotGpu {
    vk::UniqueCommandPool Pool;
    vk::UniqueCommandBuffer Cb;
    vk::UniqueFence Fence;
    vk::UniqueSemaphore SelectionReady;
};

inline SceneOneShotGpu MakeSceneOneShotGpu(vk::Device device, uint32_t queue_family) {
    auto pool = device.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queue_family});
    auto cb = std::move(device.allocateCommandBuffersUnique({*pool, vk::CommandBufferLevel::ePrimary, 1}).front());
    return {
        .Pool = std::move(pool),
        .Cb = std::move(cb),
        .Fence = device.createFenceUnique({}),
        .SelectionReady = device.createSemaphoreUnique({}),
    };
}

// Selection ignores occlusion when true.
struct SelectionXRay {
    bool Value{false};
};

// In Edit/Excite mode, orbit camera to active element on selection change.
struct OrbitToActive {
    bool Value{false};
};

enum class SelectionGesture : uint8_t {
    Click,
    Box,
};

struct BoxSelectState {
    SelectionGesture Gesture{SelectionGesture::Box};
};

struct TransformGizmoState {
    TransformGizmo::Config Config;
    TransformGizmo::Mode Mode;
};

struct SelectedInstanceCount {
    uint32_t Value{0};
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
