#pragma once

#include "render/CreateSvgResource.h"
#include "render/VulkanResources.h"

#include <entt/entity/fwd.hpp>

#include <filesystem>

// Build the process-lifetime engine and return the viewport entity.
entt::entity InitEngine(entt::registry &, VulkanResources, CreateSvgResource);
// Inverse of InitEngine. Must be called before clearing the registry so command buffers
// are freed before their owning pool, and ctx stores are torn down in dependency order.
void DeinitViewport(entt::registry &, entt::entity viewport);

// Reset all per-document viewport state to defaults, leaving the scene empty.
void SetupScene(entt::registry &, entt::entity viewport);

void AddDefaultSceneContent(entt::registry &);
void ClearScene(entt::registry &, entt::entity viewport);

// Submit GPU render (nonblocking), draw the final image into the current ImGui window, and draw overlays.
// Call WaitForRender() before the ImGui frame samples the final image.
// If provided, waits on `viewport_consumer_fence` before destroying old resources on extent change.
void RenderViewport(entt::registry &, entt::entity viewport, vk::Fence viewport_consumer_fence = {});
// Wait for pending viewport render to complete. No-op if no render pending.
void WaitForRender(entt::registry &, entt::entity viewport);

// Process deferred component events and rebuild draw lists, without presenting.
void AdvanceViewport(entt::registry &, entt::entity viewport);

// Synchronously submit and present the already-recorded frame.
void PresentViewport(entt::registry &, entt::entity viewport);

// Record the viewport to an H.264 mp4 by piping frames to an `ffmpeg` subprocess.
// When a look-through camera is active, captures only the framed sub-region matching
// what the user sees inside the dimmed overlay. Locks to the initial capture extent;
// any resize or look-through change stops recording.
void StartRecording(entt::registry &, entt::entity viewport, std::filesystem::path, int fps);
void StopRecording(entt::registry &, entt::entity viewport);
// Copy the current FinalColorImage to the recorder. No-op if not recording.
// Call after WaitForRender() so the source image is coherent.
void CaptureRecordFrame(entt::registry &, entt::entity viewport);
bool IsRecording(const entt::registry &, entt::entity viewport);
uint64_t CapturedFrameCount(const entt::registry &, entt::entity viewport);

std::string DebugBufferHeapUsage(const entt::registry &);
