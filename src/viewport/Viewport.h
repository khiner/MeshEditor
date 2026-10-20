#pragma once

namespace MTL {
class CommandBuffer;
} // namespace MTL

namespace mtl {
struct Context;
} // namespace mtl

#include "TypeList.h"
#include "metal/Buffer.h"
#include "render/PbrFeature.h"
#include "render/RenderTargets.h"
#include "scene/Defaults.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderView.h"
#include "viewport/VideoRecording.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportRenderGpu.h"

#include <expected>
#include <filesystem>
#include <vector>

// The view values of one viewport.
struct ViewportState {
    ViewCameraState Camera{Defaults::ViewCamera};
    state::Entity LookThroughEntity{state::Null}; // The scene camera the view looks through, or null.
    OrbitToActive Orbit;
    ViewportDisplay Display;
    MaterialPreviewLighting PreviewLighting{{false, false, 1.f, 0.f}};
    RenderedLighting RenderLighting{{true, true, 1.f, 0.f}};
    WorkspaceLights Lights{Defaults::WorkspaceLights};
    StudioEnvironment Environment{"forest"};
    uvec2 Extent{}; // The view's logical size.

    using Members = TypeList<ViewCameraState, state::Entity, OrbitToActive, ViewportDisplay, MaterialPreviewLighting, RenderedLighting, WorkspaceLights, StudioEnvironment, uvec2>;
    template<typename C> static constexpr bool Holds = InTypeList<C, Members>;

    template<typename C>
        requires Holds<C>
    C &Edit() { return std::get<C &>(Tie(*this)); }
    template<typename C>
        requires Holds<C>
    const C &Get() const { return std::get<const C &>(Tie(*this)); }

    bool operator==(const ViewportState &) const = default;

private:
    static auto Tie(auto &s) { return std::tie(s.Camera, s.LookThroughEntity, s.Orbit, s.Display, s.PreviewLighting, s.RenderLighting, s.Lights, s.Environment, s.Extent); }
};

// The lighting of the Rendered shading mode, and of Material Preview otherwise.
inline const PBRViewportLighting &ActivePbrLighting(const ViewportState &state) {
    return state.Display.ViewportShading == ViewportShadingMode::Rendered ? state.RenderLighting.Value : state.PreviewLighting.Value;
}
// The PBR features the view shades with: the scene's material features, and punctual lights when the view uses the scene lights.
inline PbrFeatureMask PbrFeaturesOf(const ViewportState &state, uint32_t mesh_features) {
    return mesh_features | (ActivePbrLighting(state).UseSceneLights ? uint32_t(PbrFeature::Punctual) : 0u);
}
// Whether the view shades with materials and refracts through real transmission.
inline bool RealTransmission(const ViewportState &state, uint32_t mesh_features) {
    return !WorkbenchShading(state.Display.ViewportShading) && ActivePbrLighting(state).RealTransmission && HasFeature(PbrFeaturesOf(state, mesh_features), PbrFeature::Transmission);
}

// A view onto the document, with the GPU resources its frames render into.
struct Viewport {
    Viewport(uint32_t id, mtl::BufferContext &, mtl::BindlessSet &);

    uint32_t Id; // Stable within a workspace.
    bool Visible{true}; // The panel is open.
    // The panel drew the viewport this frame, which the UI writes each frame, and always without a UI.
    bool Shown{true};
    // A displayed viewport renders and its display settings join the union the mesh records derive from.
    bool Displayed() const { return Visible && Shown; }
    ViewportState State{};
    std::optional<ViewportState> Rendered{}; // The state the last submitted frame rendered, absent before the first.
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

// The engine-lifetime viewports, holding at least one.
struct Viewports {
    Viewports(mtl::BufferContext &buffers, mtl::BindlessSet &slots) { All.emplace_back(1u, buffers, slots); }

    std::vector<Viewport> All;
    // The viewport that receives input.
    // SubmitViewports moves it to the first displayed viewport when it is not displayed.
    size_t ActiveIndex{0};

    Viewport &Active() { return All[ActiveIndex]; }
    const Viewport &Active() const { return All[ActiveIndex]; }
    Viewport *Find(uint32_t id) {
        const auto it = std::ranges::find(All, id, &Viewport::Id);
        return it != All.end() ? &*it : nullptr;
    }
};

// Appends a viewport with the default view values and an Id above every existing one.
void AddViewport(state::Registry &);
// Removes the viewport with the Id unless it is the only one, once the live UI frame stops sampling its images.
// The active viewport stays active when it remains, and the first displayed viewport becomes active otherwise.
void RemoveViewport(state::Registry &, uint32_t id);

struct DisplayUnion;
// The display settings the displayed viewports share, which the mesh display records derive from.
DisplayUnion VisibleDisplays(const Viewports &);

// Build the process-lifetime engine and create the document entity.
void InitEngine(state::Registry &);
void DeinitViewport(state::Registry &);

void InitViewportMedia(state::Registry &);
void DeinitViewportMedia(state::Registry &);

// Runs each displayed viewport's frame step, then records and submits a nonblocking frame for each one that requested it.
// Call WaitForRender() before the ImGui frame samples the final images.
void SubmitViewports(state::Registry &);
// Applies the document's pending view setup to the viewport and removes it.
void TakePendingViewSetup(state::Registry &, Viewport &);

// Emplace the document components on the document entity.
void SetupDocument(state::Registry &);

void AddDefaultSceneContent(state::Registry &);

// Call after SubmitViewports, inside the viewport's Begin block.
void DisplayViewport(state::Registry &, const Viewport &);
// Waits for a pending viewport render.
void WaitForRender(state::Registry &);

// Resume on-screen display after a headless replay: render the current scene at the viewport's extent and present synchronously.
void PresentViewport(state::Registry &, Viewport &);

bool ViewportImageReady(const Viewport &);

// Starts H.264 recording through an `ffmpeg` subprocess.
// A look-through camera records only the framed region inside the dimmed overlay.
// Resizing or changing look-through state after capture begins stops recording.
// `with_audio` also captures the master output and muxes it in when the recording stops.
void StartRecording(state::Registry &, Viewport &, const std::filesystem::path &, int fps, bool with_audio = false);
// Call after WaitForRender() so the source image is coherent.
void CaptureRecordFrame(state::Registry &, Viewport &);
bool IsRecording(const Viewport &);
uint64_t CapturedFrameCount(const Viewport &);

struct ViewportImageRgba8 {
    std::vector<std::byte> Pixels;
    uint32_t Width, Height;
};
// Requires WaitForRender() to complete before reading the source image.
// Returns an error message on failure.
std::expected<ViewportImageRgba8, std::string> ReadbackViewportImage(state::Registry &, const Viewport &);

std::string DebugBufferHeapUsage(const state::Registry &);
