#pragma once

#include "metal/Buffer.h"
#include "render/RenderTargets.h"
#include "viewport/RenderView.h"
#include "viewport/VideoRecording.h"
#include "viewport/ViewportRenderGpu.h"
#include "viewport/ViewportSchema.h"

#include <expected>
#include <filesystem>
#include <vector>

// The live render state of a viewport entity, with the GPU resources its frames render into.
struct Viewport {
    Viewport(mtl::BufferContext &, mtl::BindlessSet &);

    // The panel drew the viewport this frame, which the UI writes each frame, and always without a UI.
    bool Shown{true};
    state::Entity RenderedScene{state::Null}; // The scene the last submitted frame rendered.
    std::optional<VideoRecording> Recording{};

    RenderTargets Targets;
    RenderSamplerSlots Samplers; // The bindless slots the render passes read Targets through.
    // The live view at instance zero and one aligned instance per motion-blur step.
    mtl::Buffer SceneViewUBO;
    mtl::Buffer WorkspaceLightsUBO;
    // Coarse clusters the last cull's cut selected, which the classification accumulates.
    mtl::Buffer MeshletCoarseCount;
    RenderView FrameView{}; // The view instance zero of SceneViewUBO holds.
    RenderPhase RecordedPhase{RenderPhase::Full}; // The phase of the last submitted frame, which a pick's Prepare recording leaves alone.
    // The document or the view changed since the last submitted frame.
    bool RenderRequested{true};
    // A change since the last submitted frame can reveal geometry the depth pyramid occludes.
    bool OcclusionStale{true};
};

// Requests the viewport's frame and marks its depth pyramid stale.
void RequestFrame(state::Registry &, state::Entity viewport);
RenderView CurrentRenderView(const state::Registry &, state::Entity viewport);
void PrepareSelectionViewport(state::Registry &, state::Entity viewport);
void ReconcileWorkspace(state::Registry &);

// A displayed viewport has its panel open and drawn, renders, and its display settings join the union the mesh records derive from.
bool Displayed(const state::Registry &, state::Entity viewport);
// The lowest-index viewport, or null without one.
state::Entity FirstViewport(const state::Registry &);
// The document's active viewport when it is displayed, otherwise the lowest-index displayed viewport, otherwise the lowest-index viewport, otherwise null.
state::Entity FindActiveViewport(const state::Registry &);
// The viewport's scene resolved to a live scene.
state::Entity SceneOf(const state::Registry &, state::Entity viewport);
// The active viewport's scene, or the lowest-index scene without a viewport.
state::Entity ActingScene(const state::Registry &);
// The distinct live scenes of the displayed viewports in entity index order, or the acting scene when no viewport is displayed.
std::vector<state::Entity> ShownScenes(const state::Registry &);

struct DisplayUnion;
// The display settings the displayed viewports share, which the mesh display records derive from.
DisplayUnion VisibleDisplays(const state::Registry &);

// The viewport's Rendered lighting when the display shades in Rendered mode, and its Material Preview lighting otherwise.
const PBRViewportLighting &ActivePbrLighting(const state::Registry &, state::Entity viewport, const ViewportDisplay &);
// Whether the viewport shades with materials and refracts through real transmission.
bool RealTransmission(const state::Registry &, state::Entity viewport, const ViewportDisplay &, uint32_t mesh_features);

// Build the process-lifetime engine and create the document entity.
void InitEngine(state::Registry &);
void DeinitViewport(state::Registry &);

void InitViewportMedia(state::Registry &);
void DeinitViewportMedia(state::Registry &);

// Runs each displayed viewport's frame step, then records and submits a nonblocking frame for each one that requested it.
// Call WaitForRender() before the ImGui frame samples the final images.
void SubmitViewports(state::Registry &);
// Applies the document's pending view setup to the viewport and removes it.
// A camera the setup looks through is new to the scene, so the look-through follow later in the same settle pass moves the view onto it.
void TakePendingViewSetup(state::Registry &, state::Entity viewport);

// Emplace the document components on the document entity.
void SetupDocument(state::Registry &);

// Adds the startup cube, light, and camera to the frame's scene.
void AddDefaultSceneContent(state::Registry &);

// Call after SubmitViewports, inside the viewport's Begin block.
// `active` marks the document's active viewport, whose overlay draws the gesture guides.
void DisplayViewport(state::Registry &, state::Entity viewport, bool active);
// Waits for a pending viewport render.
void WaitForRender(state::Registry &);

// Resume on-screen display after a headless replay: render the current scene at the viewport's extent and present synchronously.
void PresentViewport(state::Registry &, state::Entity viewport);

bool ViewportImageReady(const Viewport &);

// Starts H.264 recording through an `ffmpeg` subprocess.
// A look-through camera records only the framed region inside the dimmed overlay.
// Resizing or changing look-through state after capture begins stops recording.
// `with_audio` also captures the master output and muxes it in when the recording stops.
void StartRecording(state::Registry &, state::Entity viewport, const std::filesystem::path &, int fps, bool with_audio = false);
// Call after WaitForRender() so the source image is coherent.
void CaptureRecordFrame(state::Registry &, state::Entity viewport);
bool IsRecording(const Viewport &);
uint64_t CapturedFrameCount(const Viewport &);

struct ViewportImageRgba8 {
    std::vector<std::byte> Pixels;
    uint32_t Width, Height;
};
// Requires WaitForRender() to complete before reading the source image.
// Returns an error message on failure.
std::expected<ViewportImageRgba8, std::string> ReadbackViewportImage(state::Registry &, state::Entity viewport);

std::string DebugBufferHeapUsage(const state::Registry &);
