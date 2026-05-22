#pragma once

#include "Action.h"
#include "AnimationTimeline.h"
#include "SceneVulkanResources.h"
#include "TransformGizmo.h"

struct DrawListBuilder;
struct VideoRecorder;
struct SelectionDrawInfo;

namespace mvk {
struct ImGuiTexture;
} // namespace mvk

struct Scene {
    Scene(SceneVulkanResources, entt::registry &);
    ~Scene();

    void LoadIcons();

    // Handle mouse/keyboard interactions.
    void Interact(action::Emit emit);

    // Draw the overlay controls and emit their state changes. Must run before Render().
    void InteractOverlay(action::Emit emit);

    // Submit GPU render (nonblocking), draw the final image into the current ImGui window, and draw overlays.
    // Call WaitForRender() before the ImGui frame samples the final image.
    // If provided, waits on `viewportConsumerFence` before destroying old resources on extent change.
    void Render(vk::Fence viewportConsumerFence = {});
    // Wait for pending viewport render to complete. No-op if no render pending.
    void WaitForRender();
    void RenderControls(action::Emit);

    const AnimationIcons &GetAnimationIcons() const { return AnimIcons; }
    // Render per-source animation-clip pickers above the timeline.
    void RenderClipPickers(action::Emit emit);

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
    vk::UniqueCommandBuffer RenderCommandBuffer;
#ifdef MVK_FORCE_STAGED_TRANSFERS
    vk::UniqueCommandBuffer TransferCommandBuffer;
#endif
    vk::UniqueFence RenderFence;

    struct SelectionSlotHandles;
    std::unique_ptr<SelectionSlotHandles> SelectionHandles;

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

    std::unique_ptr<mvk::ImGuiTexture> ViewportTexture;
    bool RenderPending{false}; // GPU render submitted but not yet waited on.

#ifdef MVK_FORCE_STAGED_TRANSFERS
    void RecordTransferCommandBuffer();
#endif
    bool SubmitViewport(vk::Fence viewportConsumerFence);
    // The overlay is everything drawn on top of the viewport with ImGui, independent of the main scene vulkan pipeline.
    // Split into interact (state changes) and draw (visuals) so that DrawOverlay runs after ProcessComponentEvents
    // and reads up-to-date WorldTransforms.
    void DrawOverlay();

    void RenderObjectTree(action::Emit emit);
    void RenderEntityControls(entt::entity, action::Emit emit);

    void RecordRenderCommandBuffer(bool silhouette_only = false);

    std::vector<entt::entity> RunObjectPick(uvec2 pixel, uint32_t radius_px = 0);
    std::vector<entt::entity> RunBoxSelect(std::pair<uvec2, uvec2>);
    void DispatchBoxSelect(uvec2 box_min, uvec2 box_max, uint32_t max_id, vk::Semaphore wait_semaphore);
    void RenderSelectionPass(vk::Semaphore signal_semaphore = {}) const; // On-demand selection fragment rendering.
    using SelectionBuildFn = std::function<std::vector<SelectionDrawInfo>(DrawListBuilder &)>;
    void RenderSelectionPassWith(bool render_depth, const SelectionBuildFn &build_fn, vk::Semaphore signal_semaphore = {}, bool render_silhouette = true) const;
    void RunBoxSelectElements(std::span<const ElementRange>, Element, std::pair<uvec2, uvec2>, bool is_additive);
    std::optional<uint32_t> RunSoundVerticesVertexPick(entt::entity instance_entity, uvec2 mouse_px);
};
