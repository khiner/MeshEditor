#pragma once

#include "Action.h"
#include "AnimationTimeline.h"
#include "SceneVulkanResources.h"
#include "TransformGizmo.h"
#include "ViewCamera.h"

#include <expected>

struct Armature;
struct DescriptorSlots;
struct DrawBatchInfo;
struct DrawBufferPair;
struct DrawListBuilder;
struct EnvironmentStore;
struct Mesh;
struct MeshData;
struct MeshStore;
struct PhysicsWorld;
struct ScenePipelines;
struct SceneBuffers;
struct VideoRecorder;
struct SelectionDrawInfo;
struct TextureStore;

namespace mvk {
struct ImGuiTexture;
} // namespace mvk

// Marks the camera entity currently being "looked through".
// At most one camera carries this component at a time.
struct LookingThrough {
    ViewCamera SavedViewCamera; // The pre-look-through ViewCamera, restored on exit.
};

// A contiguous span of a mesh's elements (vertices/edges/faces) in the SelectionBitset.
struct ElementRange {
    entt::entity MeshEntity;
    uint32_t Offset;
    uint32_t Count;
};

struct Scene {
    Scene(SceneVulkanResources, entt::registry &);
    ~Scene();

    void LoadIcons();

    void Apply(const action::Action &) const;
    std::expected<void, std::string> Apply(const action::FallibleAction &) const;
    // Dispatch any sub-variant whose alts are convertible to action::Action.
    template<typename... Ts>
    void Apply(std::variant<Ts...> v) const {
        std::visit([this](auto &&x) { this->Apply(std::forward<decltype(x)>(x)); }, std::move(v));
    }

    // Branch on bone-edit mode, so callers can't pick the action variant inline.
    void Duplicate(std::optional<action::Action> &out);
    void Delete(std::optional<action::Action> &out);

    // Handle mouse/keyboard interactions.
    void Interact(std::optional<action::Action> &out);

    // Submit GPU render (nonblocking), draw the final image into the current ImGui window, and draw overlays.
    // Call WaitForRender() before the ImGui frame samples the final image.
    // If provided, waits on `viewportConsumerFence` before destroying old resources on extent change.
    void Render(std::optional<action::Action> &out, vk::Fence viewportConsumerFence = {});
    // Wait for pending viewport render to complete. No-op if no render pending.
    void WaitForRender();
    void RenderControls(std::optional<action::Action> &out);

    const TimelineRange &GetTimelineRange() const;
    const TimelinePlayback &GetTimelinePlayback() const;
    const AnimationTimelineView &GetTimelineView() const;
    const AnimationIcons &GetAnimationIcons() const { return AnimIcons; }
    // Render per-source animation-clip pickers above the timeline.
    void RenderClipPickers(std::optional<action::Action> &out);

    // Record the viewport to an H.264 mp4 by piping frames to an `ffmpeg` subprocess.
    // When a look-through camera is active, captures only the framed sub-region matching
    // what the user sees inside the dimmed overlay. Locks to the initial capture extent;
    // any resize or look-through change stops recording.
    void StartRecording(std::filesystem::path, int fps);
    void StopRecording();
    // Copy the current FinalColorImage to the recorder. No-op if not recording.
    // Call after WaitForRender() so the source image is coherent.
    void CaptureRecordFrame();
    bool IsRecording() const;
    uint64_t CapturedFrameCount() const;

    void CreateSvgResource(std::unique_ptr<SvgResource> &svg, std::filesystem::path path);

    std::string DebugBufferHeapUsage() const;

    entt::entity GetSceneEntity() const { return SceneEntity; }

private:
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

    struct SelectionSlotHandles;
    std::unique_ptr<SelectionSlotHandles> SelectionHandles;

    struct EntityDestroyTracker;
    std::unique_ptr<EntityDestroyTracker> DestroyTracker;

    struct DrawState;
    std::unique_ptr<DrawState> Draw;

    entt::entity SceneEntity{null_entity}; // Singleton for scene-level components

    vec2 AccumulatedWrapMouseDelta{0, 0};
    uint32_t ObjectPickEpochTag{255}; // 8-bit epoch encoded in object click keys; wraps with periodic key reset.

public:
    vec2 PreciseWheelDelta{0, 0};

private:
    std::unique_ptr<VideoRecorder> Recorder;
    std::pair<vk::Offset3D, vk::Extent2D> RecordRegion; // Locked at StartRecording; CaptureRecordFrame stops if the live region diverges.

    // Region of FinalColorImage to record.
    // Full image, or the camera-frame sub-rect if a look-through camera is active.
    // Extent is clamped to even dimensions (libx264/yuv420p requires even width/height).
    std::pair<vk::Offset3D, vk::Extent2D> GetCaptureRegion() const;

    // Shared buffer entities for collider wireframe overlays, indexed by ColliderShapeBuffer enum.
    enum class ColliderShapeBuffer : uint8_t {
        Box,
        Sphere,
        CapsuleCap,
        Circle,
        Line,
        Count
    };
    entt::entity ColliderShapeBufferEntities[uint8_t(ColliderShapeBuffer::Count)]{null_entity, null_entity, null_entity, null_entity, null_entity};

    void EnsureWireframes();
    void UpdateWireframeTransforms();

    std::optional<vec2> BoxSelectStart, BoxSelectEnd; // Per-frame drag plumbing — producer/consumer is the same InteractOverlay call.
    std::optional<GizmoTransform> GizmoRenderTransform; // Set by InteractOverlay, consumed by DrawOverlay.
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

    static inline const std::vector<Element> NormalElements{Element::Vertex, Element::Face};

    enum class RenderRequest : uint8_t {
        None,
        Submit,
        ReRecordSilhouette, // Only silhouette batch + command buffer
        ReRecord, // Full draw list rebuild
    };

    std::unique_ptr<mvk::ImGuiTexture> ViewportTexture;
    bool RenderPending{false}; // GPU render submitted but not yet waited on.
    bool ShaderRecompileRequested{false};

#ifdef MVK_FORCE_STAGED_TRANSFERS
    void RecordTransferCommandBuffer();
#endif
    bool SubmitViewport(vk::Fence viewportConsumerFence);
    // The overlay is everything drawn on top of the viewport with ImGui, independent of the main scene vulkan pipeline.
    // Split into interact (state changes) and draw (visuals) so that DrawOverlay runs after ProcessComponentEvents
    // and reads up-to-date WorldTransforms.
    void InteractOverlay(std::optional<action::Action> &out);
    void DrawOverlay();

    void RenderObjectTree(std::optional<action::Action> &out);
    void RenderEntityControls(entt::entity, std::optional<action::Action> &out);

    void RecordRenderCommandBuffer(bool silhouette_only = false);

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

    std::vector<entt::entity> RunObjectPick(uvec2 pixel, uint32_t radius_px = 0);
    std::vector<entt::entity> RunBoxSelect(std::pair<uvec2, uvec2>);
    void DispatchBoxSelect(uvec2 box_min, uvec2 box_max, uint32_t max_id, vk::Semaphore wait_semaphore);
    void RenderSelectionPass(vk::Semaphore signal_semaphore = {}) const; // On-demand selection fragment rendering.
    using SelectionBuildFn = std::function<std::vector<SelectionDrawInfo>(DrawListBuilder &)>;
    void RenderSelectionPassWith(bool render_depth, const SelectionBuildFn &build_fn, vk::Semaphore signal_semaphore = {}, bool render_silhouette = true) const;
    void RunBoxSelectElements(std::span<const ElementRange>, Element, std::pair<uvec2, uvec2>, bool is_additive);
    std::optional<uint32_t> RunSoundVerticesVertexPick(entt::entity instance_entity, uvec2 mouse_px);
};
