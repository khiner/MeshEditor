#pragma once

#include "AnimationTimeline.h"
#include "TransformGizmo.h"
#include "World.h"
#include "mesh/Handle.h"
#include "numeric/vec2.h"
#include "vulkan/Image.h"

#include "entt_fwd.h"
#include <vulkan/vulkan.hpp>

#include <filesystem>
#include <functional>
#include <memory>
#include <set>
#include <span>
#include <vector>

using uint = uint32_t;

struct Path {
    std::filesystem::path Value;
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

enum class FaceColorMode {
    Mesh,
    Normals,
};

enum class ShaderPipelineType {
    Fill,
    Line,
    LineOverlayFaceNormals,
    LineOverlayVertexNormals,
    LineOverlayBBox,
    Point,
    Grid,
    SilhouetteDepthObject,
    SilhouetteEdgeDepthObject,
    SilhouetteEdgeDepth,
    SilhouetteEdgeColor,
    SelectionElementFace,
    SelectionElementEdge,
    SelectionElementVertex,
    SelectionElementFaceXRay,
    SelectionElementEdgeXRay,
    SelectionElementVertexXRay,
    SelectionFragmentXRay,
    SelectionFragment,
    SelectionFragmentLineXRay,
    SelectionFragmentPointXRay,
    DebugNormals,
};

struct DrawListBuilder;
struct SelectionDrawInfo;

struct MeshInstanceCreateInfo {
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

struct ObjectCreateInfo {
    std::string Name{};
    Transform Transform{};
    MeshInstanceCreateInfo::SelectBehavior Select{MeshInstanceCreateInfo::SelectBehavior::Exclusive};
};

struct Mesh;
struct MeshData;
struct MeshStore;
struct ScenePipelines;
struct SceneBuffers;
struct DescriptorSlots;
struct SvgResource;

namespace mvk {
struct ImGuiTexture;
} // namespace mvk

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

    void LoadIcons();

    World GetWorld() const;

    // Create mesh backing data, optionally with an instance.
    // Returns (mesh_entity, instance_entity) - instance_entity is entt::null if no instance info provided.
    std::pair<entt::entity, entt::entity> AddMesh(Mesh &&, std::optional<MeshInstanceCreateInfo> = {});
    std::pair<entt::entity, entt::entity> AddMesh(MeshData &&, std::optional<MeshInstanceCreateInfo> = {});
    // Loads a single mesh from non-scene formats (e.g. OBJ/PLY).
    std::pair<entt::entity, entt::entity> AddMesh(const std::filesystem::path &, std::optional<MeshInstanceCreateInfo> = {});
    // Imports glTF as a scene (may create multiple mesh + instance entities).
    std::pair<entt::entity, entt::entity> AddGltfScene(const std::filesystem::path &);
    entt::entity AddMeshInstance(entt::entity mesh_entity, MeshInstanceCreateInfo);
    entt::entity AddEmpty(ObjectCreateInfo = {});
    entt::entity AddArmature(ObjectCreateInfo = {});

    entt::entity Duplicate(entt::entity, std::optional<MeshInstanceCreateInfo> = {});
    entt::entity DuplicateLinked(entt::entity, std::optional<MeshInstanceCreateInfo> = {});

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

    // Handle mouse/keyboard interactions.
    void Interact();

    // Submit GPU render (nonblocking), draw the resolve image into the current ImGui window, and draw overlays.
    // Call WaitForRender() before the ImGui frame samples the resolve image.
    // If provided, waits on `viewportConsumerFence` before destroying old resources on extent change.
    void Render(vk::Fence viewportConsumerFence = {});
    // Wait for pending viewport render to complete. No-op if no render pending.
    void WaitForRender();
    void RenderControls();

    const AnimationTimeline &GetTimeline() const;
    AnimationTimelineView &GetTimelineView() { return TimelineView; }
    const AnimationIcons &GetAnimationIcons() const { return AnimIcons; }
    void ApplyTimelineAction(const AnimationTimelineAction &);

    mvk::ImageResource RenderBitmapToImage(std::span<const std::byte> data, uint width, uint height) const;
    void CreateSvgResource(std::unique_ptr<SvgResource> &svg, std::filesystem::path path);

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

    entt::entity SceneEntity{null_entity}; // Singleton for scene-level components

    std::set<InteractionMode> InteractionModes{InteractionMode::Object, InteractionMode::Edit};
    vec2 AccumulatedWrapMouseDelta{0, 0};
    std::vector<uint32_t> BoxSelectZeroBits;
    uint32_t NextObjectId{1}; // Monotonically increasing, assigned to RenderInstance on show

    std::unique_ptr<ScenePipelines> Pipelines;
    std::unique_ptr<SceneBuffers> Buffers;
    std::unique_ptr<MeshStore> Meshes;

    enum class SelectionMode { Click,
                               Box };
    SelectionMode SelectionMode{SelectionMode::Click};
    std::optional<vec2> BoxSelectStart, BoxSelectEnd;
    bool SelectionXRay{false}; // Edit mode: Whether to ignore occlusion when selecting elements.
    bool OrbitToActive{false}; // Edit/Excite mode: When true, orbit camera to active element.

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
    AnimationIcons AnimIcons;
    AnimationTimelineView TimelineView;
    float PlaybackFrame{1.0f}; // Smooth float frame position for playback
    int LastEvaluatedFrame{-1}; // Last frame where armature poses were evaluated

    static inline const std::vector<he::Element> NormalElements{he::Element::Vertex, he::Element::Face};

    enum class RenderRequest : uint8_t {
        None,
        Submit,
        ReRecord,
    };

    std::unique_ptr<mvk::ImGuiTexture> ViewportTexture;
    bool RenderPending{false}; // GPU render submitted but not yet waited on.
    bool SelectionStale{true}; // Selection fragment data no longer matches current scene.
    bool ShaderRecompileRequested{false};

    struct ElementRange {
        entt::entity MeshEntity;
        uint32_t Offset;
        uint32_t Count;
    };

#ifdef MVK_FORCE_STAGED_TRANSFERS
    void RecordTransferCommandBuffer();
#endif
    bool SubmitViewport(vk::Fence viewportConsumerFence);
    // The overlay is everything drawn ontop of the viewport with ImGui, independent of the main scene vulkan pipeline:
    // Orientation/Transform gizmos, active object center-dot
    void RenderOverlay();

    void RecordRenderCommandBuffer();

    // Process deferred component events. Called once per frame.
    RenderRequest ProcessComponentEvents();

    void SetInteractionMode(InteractionMode);
    void SetEditMode(he::Element mode);

    std::vector<entt::entity> RunClickSelect(glm::uvec2 pixel);
    std::vector<entt::entity> RunBoxSelect(std::pair<glm::uvec2, glm::uvec2>);
    void DispatchBoxSelect(glm::uvec2 box_min, glm::uvec2 box_max, uint32_t max_id, bool xray, vk::Semaphore wait_semaphore);
    void RenderSelectionPass(vk::Semaphore signal_semaphore = {}); // On-demand selection fragment rendering.
    using SelectionBuildFn = std::function<std::vector<SelectionDrawInfo>(DrawListBuilder &)>;
    void RenderSelectionPassWith(bool render_depth, const SelectionBuildFn &build_fn, vk::Semaphore signal_semaphore = {}, bool render_silhouette = true);
    void RenderEditSelectionPass(std::span<const ElementRange>, he::Element, vk::Semaphore signal_semaphore = {});
    std::vector<std::vector<uint32_t>> RunBoxSelectElements(std::span<const ElementRange>, he::Element, std::pair<glm::uvec2, glm::uvec2>);
    std::optional<uint32_t> RunClickSelectElement(entt::entity mesh_entity, he::Element element, glm::uvec2 mouse_px);
    std::optional<uint32_t> RunClickSelectExcitableVertex(entt::entity instance_entity, glm::uvec2 mouse_px);

    void RenderEntityControls(entt::entity);
    void RenderEntitiesTable(std::string name, entt::entity parent);
};
