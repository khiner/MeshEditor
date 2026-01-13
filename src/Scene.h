#pragma once

#include "Camera.h"
#include "TransformGizmo.h"
#include "mesh/Handle.h"
#include "mesh/MeshStore.h"
#include "numeric/vec2.h"
#include "numeric/vec4.h"
#include "vulkan/Image.h"

#include "entt_fwd.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <set>
#include <span>
#include <unordered_set>
#include <vector>

using uint = uint32_t;
namespace fs = std::filesystem;

struct Path {
    fs::path Value;
};

struct PipelineRenderer;

struct World {
    const vec3 Origin{0, 0, 0}, Up{0, 1, 0};
};

struct Lights {
    // RGB: Color of the light emitting from the view position.
    // A: Ambient intensity. (Using a single vec4 for 16-byte alignment.)
    vec4 ViewColorAndAmbient;
    // RGB: Color of the directional light.
    // A: Intensity of the directional light.
    vec4 DirectionalColorAndIntensity;
    vec3 Direction;
};

enum class InteractionMode {
    Object, // Select objects
    Edit, // Select individual mesh elements (vertices, edges, faces)
    // Available when any `Excitable` objects are present.
    // Mouse-down on an excitable object adds an `ExcitedVertex` component to the object entity, and mouse-up removes it.
    Excite,
};

enum class ViewportShadingMode {
    Wireframe,
    Solid,
};

enum class ColorMode {
    Mesh,
    Normals,
};

inline std::string to_string(InteractionMode mode) {
    switch (mode) {
        case InteractionMode::Object: return "Object";
        case InteractionMode::Edit: return "Edit";
        case InteractionMode::Excite: return "Excite";
    }
}

struct MeshCreateInfo {
    std::string Name{};
    Transform Transform{};

    enum class SelectBehavior {
        Exclusive,
        Additive,
        None,
    };
    SelectBehavior Select{SelectBehavior::Exclusive};
    bool Visible{true};
};

static constexpr Camera CreateDefaultCamera() { return {{0, 0, 2}, {0, 0, 0}, glm::radians(60.f), 0.01, 100}; }

inline static vk::SampleCountFlagBits GetMaxUsableSampleCount(vk::PhysicalDevice pd) {
    const auto props = pd.getProperties();
    const auto counts = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
    if (counts & vk::SampleCountFlagBits::e64) return vk::SampleCountFlagBits::e64;
    if (counts & vk::SampleCountFlagBits::e32) return vk::SampleCountFlagBits::e32;
    if (counts & vk::SampleCountFlagBits::e16) return vk::SampleCountFlagBits::e16;
    if (counts & vk::SampleCountFlagBits::e8) return vk::SampleCountFlagBits::e8;
    if (counts & vk::SampleCountFlagBits::e4) return vk::SampleCountFlagBits::e4;
    if (counts & vk::SampleCountFlagBits::e2) return vk::SampleCountFlagBits::e2;

    return vk::SampleCountFlagBits::e1;
}

struct SvgResource;
struct ScenePipelines;
struct SceneBuffers;
struct DescriptorSlots;

struct SceneVulkanResources {
    vk::Instance Instance;
    vk::PhysicalDevice PhysicalDevice;
    vk::Device Device;
    uint32_t QueueFamily;
    vk::Queue Queue;
};

struct Excitable;

struct Scene {
    Scene(SceneVulkanResources, entt::registry &);
    ~Scene();

    void LoadIcons(vk::Device);

    World World{};

    // Returns (mesh_entity, instance_entity)
    std::pair<entt::entity, entt::entity> AddMesh(Mesh &&, MeshCreateInfo = {});
    std::pair<entt::entity, entt::entity> AddMesh(MeshData &&, MeshCreateInfo = {});
    std::pair<entt::entity, entt::entity> AddMesh(const fs::path &, MeshCreateInfo = {});

    entt::entity Duplicate(entt::entity, std::optional<MeshCreateInfo> = {});
    entt::entity DuplicateLinked(entt::entity, std::optional<MeshCreateInfo> = {});

    void SetMeshPositions(entt::entity, std::span<const vec3>);
    void ClearMeshes();

    void Destroy(entt::entity);

    void SetVisible(entt::entity, bool visible);

    entt::entity GetMeshEntity(entt::entity) const;
    entt::entity GetActiveMeshEntity() const;
    void Select(entt::entity);
    void ToggleSelected(entt::entity);

    // Actions on selected entities
    void Duplicate();
    void DuplicateLinked();
    void Delete();

    vk::Extent2D GetExtent() const { return Extent; }
    vk::ImageView GetViewportImageView() const;

    // Handle mouse/keyboard interactions.
    void Interact();

    // Renders to an image view that can be accessed with `GetViewportImageView()`.
    // The extent of the resolve image can be found with `GetExtent()` after the call,
    // and it will be equal to the dimensions of `GetContentRegionAvail()` at the beginning of the call.
    // Returns true if the extent changed, which requires recreating the ImGui texture wrapper.
    // Note: This submits GPU work but does NOT wait for completion. Call WaitForRender() before sampling the viewport image.
    // If provided, waits on `viewportConsumerFence` before destroying old resources on extent change.
    bool SubmitViewport(vk::Fence viewportConsumerFence = {});
    // Wait for pending viewport render to complete. No-op if no render pending.
    void WaitForRender();
    // The "overlay" is everything drawn ontop of the viewport with ImGui, independent of the main scene vulkan pipeline:
    // - Orientation/Transform gizmos
    // - Active object center-dot
    void RenderOverlay();
    void RenderControls();

    mvk::ImageResource RenderBitmapToImage(std::span<const std::byte> data, uint width, uint height) const;

    std::string DebugBufferHeapUsage() const;

private:
    SceneVulkanResources Vk;
    entt::registry &R;
    vk::UniqueCommandPool CommandPool;
    vk::UniqueCommandBuffer RenderCommandBuffer;
#ifdef MVK_FORCE_STAGED_TRANSFERS
    vk::UniqueCommandBuffer TransferCommandBuffer;
#endif
    vk::UniqueFence RenderFence;
    vk::UniqueFence OneShotFence; // For short-lived transfers (e.g. selection passes and image uploads)
    vk::UniqueSemaphore SelectionReadySemaphore; // Signals selection buffers ready for compute.
    vk::UniqueCommandBuffer ClickCommandBuffer;
    std::unique_ptr<DescriptorSlots> Slots;

    struct SelectionSlotHandles;
    std::unique_ptr<SelectionSlotHandles> SelectionHandles;

    struct EntityDestroyTracker;
    std::unique_ptr<EntityDestroyTracker> DestroyTracker;

    entt::entity SettingsEntity{null_entity}; // Singleton for SceneSettings component

    Camera Camera{CreateDefaultCamera()};
    Lights Lights{{1, 1, 1, 0.1}, {1, 1, 1, 0.15}, {-1, -1, -1}};

    vec4 EdgeColor{1, 1, 1, 1}; // Used for line mode.
    vec4 MeshEdgeColor{0, 0, 0, 1}; // Used for faces mode.

    std::set<InteractionMode> InteractionModes{InteractionMode::Object, InteractionMode::Edit};
    he::Element EditMode{he::Element::Face}; // Which element type to edit (vertex/edge/face)
    vec2 AccumulatedWrapMouseDelta{0, 0};
    std::vector<uint32_t> BoxSelectZeroBits;
    uint32_t NextObjectId{1}; // Monotonically increasing, assigned to RenderInstance on show

    vk::Extent2D Extent;
    struct Colors {
        vec4 Active{1, 0.627, 0.157, 1}; // Blender's default `Preferences->Themes->3D Viewport->Active Object`.
        vec4 Selected{0.929, 0.341, 0, 1}; // Blender's default `Preferences->Themes->3D Viewport->Object Selected`.
    };
    Colors Colors;

    std::unique_ptr<ScenePipelines> Pipelines;
    std::unique_ptr<SceneBuffers> Buffers;
    MeshStore Meshes;

    enum class SelectionMode { Click,
                               Box };
    SelectionMode SelectionMode{SelectionMode::Click};
    std::optional<vec2> BoxSelectStart, BoxSelectEnd;
    bool SelectionXRay{false}; // Edit mode only. Whether to ignore occlusion when selecting elements.

    struct TransformGizmoState {
        TransformGizmo::Config Config;
        TransformGizmo::Mode Mode;
    };
    TransformGizmoState MGizmo;
    std::optional<TransformGizmo::TransformType> StartScreenTransform;
    bool TransformModePillsHovered{false};

    struct TransformIcons {
        std::unique_ptr<SvgResource> Select, SelectBox, Move, Rotate, Scale, Universal;
    };
    TransformIcons Icons;

    static inline const std::vector<he::Element> NormalElements{he::Element::Vertex, he::Element::Face};

    bool CommandBufferDirty{false}; // Render command buffer needs re-recording.
    bool NeedsRender{false}; // Scene needs to be re-rendered (submit command buffers).
    bool RenderPending{false}; // GPU render submitted but not yet waited on.
    bool SelectionStale{true}; // Selection fragment data no longer matches current scene.

    struct ElementRange {
        entt::entity MeshEntity;
        uint32_t Offset;
        uint32_t Count;
    };

    void UpdateMeshElementStateBuffers(entt::entity);
#ifdef MVK_FORCE_STAGED_TRANSFERS
    void RecordTransferCommandBuffer();
#endif
    void RecordRenderCommandBuffer();

    // Process deferred component events. Called once per frame.
    void ProcessComponentEvents();

    void SetInteractionMode(InteractionMode);
    void SetEditMode(he::Element mode);
    void SelectElement(entt::entity mesh_entity, he::AnyHandle element, bool toggle = false);

    std::vector<entt::entity> RunClickSelect(glm::uvec2 pixel);
    std::vector<entt::entity> RunBoxSelect(std::pair<glm::uvec2, glm::uvec2>);
    void RenderSelectionPass(vk::Semaphore signal_semaphore = {}); // On-demand selection fragment rendering.
    void RenderSilhouetteDepth(vk::CommandBuffer cb);
    void RenderSelectionPassWith(bool render_depth, const std::function<void(vk::CommandBuffer, const PipelineRenderer &)> &draw_fn, vk::Semaphore signal_semaphore = {});
    void RenderEditSelectionPass(std::span<const ElementRange>, he::Element, vk::Semaphore signal_semaphore = {});
    std::vector<std::vector<uint32_t>> RunBoxSelectElements(std::span<const ElementRange>, he::Element, std::pair<glm::uvec2, glm::uvec2>);
    std::optional<he::AnyHandle> RunClickSelectElement(entt::entity mesh_entity, he::Element element, glm::uvec2 mouse_px);
    std::optional<uint32_t> RunClickSelectExcitableVertex(entt::entity instance_entity, glm::uvec2 mouse_px);

    void RenderEntityControls(entt::entity);
    void RenderEntitiesTable(std::string name, entt::entity parent);

    // VK buffer update methods
    void UpdateSceneUBO();
    void UpdateEdgeColors();
    void UpdateEntitySelectionOverlays(entt::entity mesh_entity);
};
