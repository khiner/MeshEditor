#pragma once

#include "SceneFrameState.h"
#include "SceneVulkanResources.h"
#include "entt_fwd.h"

#include <entt/entity/fwd.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

struct VideoRecorder;
struct SvgResource;

namespace mvk {
struct ImGuiTexture;
} // namespace mvk

struct Scene {
    Scene(SceneVulkanResources, entt::registry &);
    ~Scene();

    // Submit GPU render (nonblocking), draw the final image into the current ImGui window, and draw overlays.
    // Call WaitForRender() before the ImGui frame samples the final image.
    // If provided, waits on `viewportConsumerFence` before destroying old resources on extent change.
    void Render(vk::Fence viewportConsumerFence = {});
    // Wait for pending viewport render to complete. No-op if no render pending.
    void WaitForRender();

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

    SceneFrameState Frame;

private:
    entt::registry &R;
    vk::UniqueCommandBuffer RenderCommandBuffer;
#ifdef MVK_FORCE_STAGED_TRANSFERS
    vk::UniqueCommandBuffer TransferCommandBuffer;
#endif
    vk::UniqueFence RenderFence;

    entt::entity SceneEntity{null_entity}; // Singleton for scene-level components

    std::unique_ptr<VideoRecorder> Recorder;
    std::pair<vk::Offset3D, vk::Extent2D> RecordRegion; // Locked at StartRecording; CaptureRecordFrame stops if the live region diverges.

    std::unique_ptr<mvk::ImGuiTexture> ViewportTexture;

    bool SubmitViewport(vk::Fence viewportConsumerFence);
};
