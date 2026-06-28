#pragma once

#include "render/CreateSvgResource.h"
#include "vulkan/VulkanResources.h"

#include <entt/entity/fwd.hpp>

#include <expected>
#include <filesystem>

// Build the process-lifetime engine and return the viewport entity.
entt::entity InitEngine(entt::registry &, VulkanResources);
void DeinitViewport(entt::registry &, entt::entity viewport);

// App-only presentation/media layered on InitEngine.
void InitViewportMedia(entt::registry &, CreateSvgResource &&);
void DeinitViewportMedia(entt::registry &);

// Run ProcessComponentEvents, then record and submit the GPU render (nonblocking).
// `viewport_consumer_fence`, if set, is waited on before old resources are destroyed on an extent change.
// Call WaitForRender() before the ImGui frame samples the final image.
void SubmitViewport(entt::registry &, entt::entity viewport, vk::Fence viewport_consumer_fence = {});

// Reset all per-document viewport state to defaults, leaving the scene empty.
void SetupScene(entt::registry &, entt::entity viewport);

void AddDefaultSceneContent(entt::registry &);
void ClearScene(entt::registry &, entt::entity viewport);

// Draw the recorded viewport image into the current ImGui window, then draw overlays.
// Call after SubmitViewport, inside the viewport's Begin block.
void DisplayViewport(entt::registry &, entt::entity viewport);
// Wait for pending viewport render to complete. No-op if no render pending.
void WaitForRender(entt::registry &);

// Process deferred component events and rebuild draw lists, without presenting.
void AdvanceViewport(entt::registry &, entt::entity viewport);

// Resume on-screen display after a headless replay: render the current scene at the current ViewportExtent and present synchronously.
void PresentViewport(entt::registry &, entt::entity viewport);

// True once the viewport's color image has been built and rendered at a valid extent.
bool ViewportImageReady(const entt::registry &);

// Record the viewport to an H.264 mp4 by piping frames to an `ffmpeg` subprocess.
// When a look-through camera is active, captures only the framed sub-region matching
// what the user sees inside the dimmed overlay. Locks to the initial capture extent;
// any resize or look-through change stops recording.
void StartRecording(entt::registry &, entt::entity viewport, const std::filesystem::path &, int fps);
// Copy the current FinalColorImage to the recorder. No-op if not recording.
// Call after WaitForRender() so the source image is coherent.
void CaptureRecordFrame(entt::registry &, entt::entity viewport);
bool IsRecording(const entt::registry &, entt::entity viewport);
uint64_t CapturedFrameCount(const entt::registry &, entt::entity viewport);

// Upright, tightly-packed RGBA8 pixels read back from the viewport's final color image.
struct ViewportImageRgba8 {
    std::vector<std::byte> Pixels;
    uint32_t Width, Height;
};
// Read back the current FinalColorImage as upright RGBA8 (the framed sub-region matching recording).
// Call after WaitForRender() so the source image is coherent. Returns an error message on failure.
std::expected<ViewportImageRgba8, std::string> ReadbackViewportImage(entt::registry &);

std::string DebugBufferHeapUsage(const entt::registry &);
