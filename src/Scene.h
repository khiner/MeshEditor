#pragma once

#include "AnimationTimeline.h"
#include "Entity.h"
#include "ShaderPipelineType.h"
#include "TransformGizmo.h"
#include "ViewCamera.h"
#include "World.h"
#include "gpu/InteractionMode.h"
#include "gpu/PunctualLight.h"
#include "mesh/Handle.h"
#include "numeric/vec2.h"

#include <expected>
#include <filesystem>
#include <set>

using uint = uint32_t;

struct Path {
    std::filesystem::path Value;
};

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

struct ExtrasWireframe;
struct Mesh;
struct MeshData;
struct MeshStore;
struct ScenePipelines;
struct SceneBuffers;
struct DescriptorSlots;
struct SvgResource;
struct TextureStore;
struct EnvironmentStore;

namespace mvk {
struct ImGuiTexture;
} // namespace mvk

#include "SceneVulkanResources.h"

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
    std::expected<std::pair<entt::entity, entt::entity>, std::string> AddGltfScene(const std::filesystem::path &);
    entt::entity AddMeshInstance(entt::entity mesh_entity, MeshInstanceCreateInfo);
    entt::entity AddEmpty(ObjectCreateInfo = {});
    entt::entity AddArmature(ObjectCreateInfo = {});
    entt::entity AddCamera(ObjectCreateInfo = {});
    entt::entity AddLight(ObjectCreateInfo = {}, std::optional<PunctualLight> = {});

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
    void SelectBone(entt::entity);

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
        ReRecord,
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
    // The overlay is everything drawn ontop of the viewport with ImGui, independent of the main scene vulkan pipeline:
    // Orientation/Transform gizmos, active object origin dots
    void RenderOverlay();
    void ExitLookThroughCamera();
    void SnapToCamera(entt::entity camera_entity);
    void AnimateToCamera(entt::entity camera_entity);

    void RecordRenderCommandBuffer();

    // Process deferred component events. Called once per frame.
    RenderRequest ProcessComponentEvents();

    // Batch-process all deferred ModelsBuffer GPU operations (construction, insert, erase).
    // Called from ProcessComponentEvents before transform/state sync.
    // Returns entities that were newly inserted into GPU buffers this frame.
    // WorldTransform is NOT written for these — callers must ensure it gets written before submit.
    std::vector<entt::entity> SyncModelsBuffers();

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
    std::optional<uint32_t> RunExcitableVertexPick(entt::entity instance_entity, uvec2 mouse_px);

    void ApplySelectBehavior(entt::entity, MeshInstanceCreateInfo::SelectBehavior);
    entt::entity CreateExtrasBufferEntity(ExtrasWireframe &&);
    entt::entity CreateExtrasObject(ExtrasWireframe &&, ObjectType, ObjectCreateInfo, const std::string &default_name);

    void CreateBoneInstances(entt::entity arm_obj_entity, entt::entity arm_data_entity);
    entt::entity CreateSingleBoneInstance(entt::entity arm_obj_entity, uint32_t bone_id); // Create ECS entity + joints for one bone.
    void DestroyArmatureData(entt::entity arm_obj_entity);
    void RebuildBoneStructure(entt::entity arm_data_entity); // Call after Armature::AddBone/RemoveBone

    // Prefilter HDR at index (if not already cached) and activate it as the studio environment.
    void SetStudioEnvironment(uint32_t index);

    void ClearSelectedBoneTransforms(bool position, bool rotation, bool scale);
    void RenderEntityControls(entt::entity);
    void RenderObjectTree();
};
