#pragma once

namespace MTL {
class CommandBuffer;
} // namespace MTL

namespace mtl {
struct Context;
} // namespace mtl

#include "state/Entity.h"
#include <expected>
#include <filesystem>
#include <vector>

// Build the process-lifetime engine and return the viewport entity.
state::Entity InitEngine(state::Registry &);
void DeinitViewport(state::Registry &, state::Entity viewport);

void InitViewportMedia(state::Registry &);
void DeinitViewportMedia(state::Registry &);

// Submits a nonblocking render of the processed scene.
// Call WaitForRender() before the ImGui frame samples the final image.
void SubmitViewport(state::Registry &, state::Entity viewport);

// Reset all per-document viewport state to defaults and clear the scene.
void SetupScene(state::Registry &, state::Entity viewport);

void AddDefaultSceneContent(state::Registry &);

// Call after SubmitViewport, inside the viewport's Begin block.
void DisplayViewport(state::Registry &, state::Entity viewport);
// Waits for a pending viewport render.
void WaitForRender(state::Registry &);

// Resume on-screen display after a headless replay: render the current scene at the current ViewportExtent and present synchronously.
void PresentViewport(state::Registry &, state::Entity viewport);

bool ViewportImageReady(const state::Registry &);

// Starts H.264 recording through an `ffmpeg` subprocess.
// A look-through camera records only the framed region inside the dimmed overlay.
// Resizing or changing look-through state after capture begins stops recording.
// `with_audio` also captures the master output and muxes it in when the recording stops.
void StartRecording(state::Registry &, state::Entity viewport, const std::filesystem::path &, int fps, bool with_audio = false);
// Call after WaitForRender() so the source image is coherent.
void CaptureRecordFrame(state::Registry &, state::Entity viewport);
bool IsRecording(const state::Registry &, state::Entity viewport);
uint64_t CapturedFrameCount(const state::Registry &, state::Entity viewport);

struct ViewportImageRgba8 {
    std::vector<std::byte> Pixels;
    uint32_t Width, Height;
};
// Requires WaitForRender() to complete before reading the source image.
// Returns an error message on failure.
std::expected<ViewportImageRgba8, std::string> ReadbackViewportImage(state::Registry &);

std::string DebugBufferHeapUsage(const state::Registry &);
