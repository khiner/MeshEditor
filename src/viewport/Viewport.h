#pragma once

#include "metal/MetalCpp.h"

namespace mtl {
struct Context;
} // namespace mtl

#include <entt/entity/fwd.hpp>

#include <expected>
#include <filesystem>
#include <vector>

// Build the process-lifetime engine and return the viewport entity.
entt::entity InitEngine(entt::registry &);
void DeinitViewport(entt::registry &, entt::entity viewport);

void InitViewportMedia(entt::registry &);
void DeinitViewportMedia(entt::registry &);

// Processes component events and submits a nonblocking render.
// `viewport_consumer_fence`, if set, is waited on before old resources are destroyed on an extent change.
// Call WaitForRender() before the ImGui frame samples the final image.
void SubmitViewport(entt::registry &, entt::entity viewport, MTL::CommandBuffer *viewport_consumer = nullptr);

// Reset all per-document viewport state to defaults and clear the scene.
void SetupScene(entt::registry &, entt::entity viewport);

void AddDefaultSceneContent(entt::registry &);
void ClearScene(entt::registry &, entt::entity viewport);

// Call after SubmitViewport, inside the viewport's Begin block.
void DisplayViewport(entt::registry &, entt::entity viewport);
// Waits for a pending viewport render.
void WaitForRender(entt::registry &);

// Resume on-screen display after a headless replay: render the current scene at the current ViewportExtent and present synchronously.
void PresentViewport(entt::registry &, entt::entity viewport);

bool ViewportImageReady(const entt::registry &);

// Starts H.264 recording through an `ffmpeg` subprocess.
// A look-through camera records only the framed region inside the dimmed overlay.
// Resizing or changing look-through state after capture begins stops recording.
// `with_audio` also captures the master output and muxes it in when the recording stops.
void StartRecording(entt::registry &, entt::entity viewport, const std::filesystem::path &, int fps, bool with_audio = false);
// Call after WaitForRender() so the source image is coherent.
void CaptureRecordFrame(entt::registry &, entt::entity viewport);
bool IsRecording(const entt::registry &, entt::entity viewport);
uint64_t CapturedFrameCount(const entt::registry &, entt::entity viewport);

struct ViewportImageRgba8 {
    std::vector<std::byte> Pixels;
    uint32_t Width, Height;
};
// Requires WaitForRender() to complete before reading the source image.
// Returns an error message on failure.
std::expected<ViewportImageRgba8, std::string> ReadbackViewportImage(entt::registry &);

std::string DebugBufferHeapUsage(const entt::registry &);
