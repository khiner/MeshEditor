#pragma once

#include "AnimationTimeline.h"
#include "Entity.h"
#include "Path.h"
#include "SceneVulkanResources.h"
#include "TransformGizmo.h"
#include "ViewCamera.h"
#include "gpu/InteractionMode.h"
#include "gpu/PunctualLight.h"
#include "mesh/Handle.h"

#include <expected>
#include <set>
#include <unordered_set>

enum class ViewportShadingMode : uint8_t {
    Wireframe,
    Solid,
    MaterialPreview,
    Rendered,
};

enum class FaceColorMode {
    Mesh,
    Normals,
};

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

struct DescriptorSlots;
struct DrawBufferPair;
struct DrawListBuilder;
struct EnvironmentStore;
struct Mesh;
struct MeshData;
struct MeshStore;
struct PhysicsWorld;
struct ScenePipelines;
struct SceneBuffers;
struct SelectionDrawInfo;
struct TextureStore;

namespace mvk {
struct ImGuiTexture;
} // namespace mvk

struct Scene {
    Scene(SceneVulkanResources, entt::registry &);
    ~Scene();

    void LoadIcons();

    // Create mesh backing data, optionally with an instance.
    // Returns (mesh_entity, instance_entity) - instance_entity is entt::null if no instance info provided.
    std::pair<entt::entity, entt::entity> AddMesh(Mesh &&, std::optional<MeshInstanceCreateInfo> = {});
    std::pair<entt::entity, entt::entity> AddMesh(MeshData &&, std::optional<MeshInstanceCreateInfo> = {});
    // Loads a single mesh from non-scene formats (e.g. OBJ/PLY).
    std::pair<entt::entity, entt::entity> AddMesh(const std::filesystem::path &, std::optional<MeshInstanceCreateInfo> = {});
    // Imports glTF as a scene (may create multiple mesh + instance entities).
    std::expected<std::pair<entt::entity, entt::entity>, std::string> AddGltfScene(const std::filesystem::path &);
    void LoadRealImpact(const std::filesystem::path &directory);

    entt::entity AddMeshInstance(entt::entity mesh_entity, MeshInstanceCreateInfo);
    entt::entity AddEmpty(ObjectCreateInfo = {});
    entt::entity AddArmature(ObjectCreateInfo = {});
    entt::entity AddCamera(ObjectCreateInfo = {});
    entt::entity AddLight(ObjectCreateInfo = {}, std::optional<PunctualLight> = {});

    entt::entity Duplicate(entt::entity, std::optional<MeshInstanceCreateInfo> = {});
    entt::entity DuplicateLinked(entt::entity, std::optional<MeshInstanceCreateInfo> = {});

    void ReplaceMesh(entt::entity, MeshData &&);
    void ClearMeshes();

    void Destroy(entt::entity);

    void SetVisible(entt::entity, bool visible);

    entt::entity GetMeshEntity(entt::entity) const;
    entt::entity GetActiveMeshEntity() const;
    void Select(entt::entity);
    void ToggleSelected(entt::entity);

    // Actions on selected entities
    bool CanDuplicate() const;
    bool CanDuplicateLinked() const;
    void Duplicate();
    void DuplicateLinked();
    bool CanDelete() const;
    void Delete();

    void AddBone();
    void ExtrudeBone();
    void DuplicateSelectedBones();
    void DeleteSelectedBones();

    // Handle mouse/keyboard interactions.
    void Interact();

    // Submit GPU render (nonblocking), draw the final image into the current ImGui window, and draw overlays.
    // Call WaitForRender() before the ImGui frame samples the final image.
    // If provided, waits on `viewportConsumerFence` before destroying old resources on extent change.
    void Render(vk::Fence viewportConsumerFence = {});
    // Wait for pending viewport render to complete. No-op if no render pending.
    void WaitForRender();
    void RenderControls();

    const AnimationTimeline &GetTimeline() const;
    AnimationTimelineView &GetTimelineView() { return TimelineView; }
    const AnimationIcons &GetAnimationIcons() const { return AnimIcons; }
    void ApplyTimelineAction(const AnimationTimelineAction &);

    void CreateSvgResource(std::unique_ptr<SvgResource> &svg, std::filesystem::path path);

    std::string DebugBufferHeapUsage() const;

    entt::entity GetSceneEntity() const { return SceneEntity; }

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

    struct DrawState;
    std::unique_ptr<DrawState> Draw;

    entt::entity SceneEntity{null_entity}; // Singleton for scene-level components

    std::unordered_set<std::string> NameSet; // O(1) uniqueness check for CreateName
    std::set<InteractionMode> InteractionModes{InteractionMode::Object, InteractionMode::Edit, InteractionMode::Pose};
    vec2 AccumulatedWrapMouseDelta{0, 0};
    double LastWheelZoomTime{-1.0};
    float WheelZoomBurst{0.f}; // Signed by zoom direction; stores the next burst level to apply.
    uint32_t ObjectPickEpochTag{255}; // 8-bit epoch encoded in object click keys; wraps with periodic key reset.
    uint32_t NextObjectId{1}; // Monotonically increasing, assigned to RenderInstance on show

    std::unique_ptr<ScenePipelines> Pipelines;
    std::unique_ptr<SceneBuffers> Buffers;
    std::unique_ptr<MeshStore> Meshes;
    std::unique_ptr<TextureStore> Textures;
    std::unique_ptr<EnvironmentStore> Environments;
    std::unique_ptr<PhysicsWorld> Physics;

    // Shared buffer entities for collider wireframe overlays, indexed by ColliderShapeBuffer enum.
    enum ColliderShapeBuffer : uint8_t { CSB_Box,
                                         CSB_Sphere,
                                         CSB_CapsuleBody,
                                         CSB_CapsuleCap,
                                         CSB_Cylinder,
                                         CSB_Count };
    entt::entity ColliderShapeBufferEntities[CSB_Count]{null_entity, null_entity, null_entity, null_entity, null_entity};
    void SyncColliderWireframes();

    enum class SelectionMode { Click,
                               Box };
    SelectionMode SelectionMode{SelectionMode::Box};
    std::optional<vec2> BoxSelectStart, BoxSelectEnd;
    bool SelectionXRay{false}; // Edit mode: Whether to ignore occlusion when selecting elements.
    bool OrbitToActive{false}; // Edit/Excite mode: When true, orbit camera to active element.
    std::optional<ViewCamera> SavedViewCamera; // Saved viewport state when looking through a scene camera.

    struct TransformGizmoState {
        TransformGizmo::Config Config;
        TransformGizmo::Mode Mode;
    };
    TransformGizmoState MGizmo;
    std::optional<GizmoTransform> GizmoRenderTransform; // Set by InteractOverlay, consumed by DrawOverlay.
    std::optional<TransformGizmo::TransformType> StartScreenTransform;
    bool OverlayControlsHovered{false};

    struct TransformIcons {
        std::unique_ptr<SvgResource> Select, SelectBox, Move, Rotate, Scale, Universal;
    };
    struct ViewportShadingIcons {
        std::unique_ptr<SvgResource> Wireframe, Solid, MaterialPreview, Rendered;
    };
    TransformIcons Icons;
    ViewportShadingIcons ShadingIcons;
    std::unique_ptr<SvgResource> OverlayIcon;
    AnimationIcons AnimIcons;
    AnimationTimelineView TimelineView;
    float PlaybackFrame{1.0f}; // Smooth float frame position for playback
    int LastEvaluatedFrame{-1}; // Last frame where armature poses were evaluated

    static inline const std::vector<Element> NormalElements{Element::Vertex, Element::Face};

    enum class RenderRequest : uint8_t {
        None,
        Submit,
        ReRecordSilhouette, // Only silhouette batch + command buffer
        ReRecord, // Full draw list rebuild
    };

    std::unique_ptr<mvk::ImGuiTexture> ViewportTexture;
    bool RenderPending{false}; // GPU render submitted but not yet waited on.
    bool SelectionStale{true}; // Selection fragment data no longer matches current scene.
    bool ElementStatesDirty{false}; // Element state buffers updated by GPU compute; triggers a submit.
    bool SelectionBitsDirty{false}; // Bitset written by Interact; ProcessComponentEvents dispatches the compute update.
    bool ShaderRecompileRequested{false};
    bool ProfileNextProcessComponentEvents{false};

    struct ElementRange {
        entt::entity MeshEntity;
        uint32_t Offset;
        uint32_t Count;
    };

#ifdef MVK_FORCE_STAGED_TRANSFERS
    void RecordTransferCommandBuffer();
#endif
    bool SubmitViewport(vk::Fence viewportConsumerFence);
    // The overlay is everything drawn on top of the viewport with ImGui, independent of the main scene vulkan pipeline.
    // Split into interact (state changes) and draw (visuals) so that DrawOverlay runs after ProcessComponentEvents
    // and reads up-to-date WorldTransforms.
    void InteractOverlay();
    void DrawOverlay();
    void ExitLookThroughCamera();
    void SnapToCamera(entt::entity camera_entity);
    void AnimateToCamera(entt::entity camera_entity);

    void RecordRenderCommandBuffer(bool silhouette_only = false);
    void FlushDrawList(const DrawListBuilder &, DrawBufferPair &);

    // Process deferred component events. Called once per frame.
    RenderRequest ProcessComponentEvents();

    // Batch-process all deferred ModelsBuffer GPU operations (construction, insert, erase).
    // Called from ProcessComponentEvents before transform/state sync.
    struct SyncResult {
        std::vector<entt::entity> NewlyInserted; // Entities inserted into GPU buffers — callers must write their WorldTransform before submit.
        std::vector<entt::entity> NewMeshEntities; // Mesh entities needing deferred index buffer creation.
        std::vector<entt::entity> NewExtrasEntities; // Non-mesh buffer entities (extras/bone/joint) needing deferred index creation.
    };
    SyncResult SyncModelsBuffers();

    void SetInteractionMode(InteractionMode);
    void SetEditMode(Element mode);

    std::vector<entt::entity> RunObjectPick(uvec2 pixel, uint32_t radius_px = 0);
    std::vector<entt::entity> RunBoxSelect(std::pair<uvec2, uvec2>);
    void DispatchBoxSelect(uvec2 box_min, uvec2 box_max, uint32_t max_id, vk::Semaphore wait_semaphore);
    void RenderSelectionPass(vk::Semaphore signal_semaphore = {}); // On-demand selection fragment rendering.
    using SelectionBuildFn = std::function<std::vector<SelectionDrawInfo>(DrawListBuilder &)>;
    void RenderSelectionPassWith(bool render_depth, const SelectionBuildFn &build_fn, vk::Semaphore signal_semaphore = {}, bool render_silhouette = true);
    void RenderEditSelectionPass(std::span<const ElementRange>, Element, vk::Semaphore signal_semaphore = {});
    void RenderElementSelectionPass(std::span<const ElementRange>, Element, bool write_bitset, uvec2 box_min = {}, uvec2 box_max = {}, vk::Semaphore signal_semaphore = {});
    void DispatchUpdateSelectionStates(std::span<const ElementRange>, Element); // Submits UpdateSelectionState.comp for each range and waits.
    void ApplySelectionStateUpdate(std::span<const ElementRange>, Element); // Dispatch + GPU/CPU state update + dirty in one call.
    std::vector<ElementRange> GetBitsetRangesForSelected() const; // Build ElementRange list from selected mesh bitset ranges.
    void RunBoxSelectElements(std::span<const ElementRange>, Element, std::pair<uvec2, uvec2>, bool is_additive);
    std::optional<std::pair<entt::entity, uint32_t>> RunElementPickFromRanges(std::span<const ElementRange>, Element, uvec2 mouse_px);
    std::optional<uint32_t> RunSoundVerticesVertexPick(entt::entity instance_entity, uvec2 mouse_px);

    void ApplySelectBehavior(entt::entity, MeshInstanceCreateInfo::SelectBehavior);
    entt::entity CreateExtrasBufferEntity(std::span<const vec3> positions, std::span<const uint8_t> vertex_classes = {}, std::span<const uint32_t> edge_indices = {});
    entt::entity CreateExtrasObject(std::span<const vec3> positions, std::span<const uint8_t> vertex_classes, std::span<const uint32_t> edge_indices, ObjectType, ObjectCreateInfo, const std::string &default_name);

    std::string CreateName(std::string_view prefix);
    void OnDestroyName(entt::registry &, entt::entity);

    void CreateBoneInstances(entt::entity arm_obj_entity, entt::entity arm_data_entity);
    entt::entity CreateSingleBoneInstance(entt::entity arm_obj_entity, uint32_t bone_id); // Create ECS entity + joints for one bone.
    void DestroyArmatureData(entt::entity arm_obj_entity);
    void RebuildBoneStructure(entt::entity arm_data_entity);

    // Prefilter HDR at index (if not already cached) and activate it as the studio environment.
    void SetStudioEnvironment(uint32_t index);

    void RenderObjectTree();
    void RenderEntityControls(entt::entity);
};
