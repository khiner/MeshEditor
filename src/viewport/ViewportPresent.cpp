#include "numeric/uvec2.h"

#include "state/Scene.h"
#include "viewport/Viewport.h"

#include "Camera.h"
#include "audio/AudioSystem.h"
#include "metal/ImGuiTexture.h"
#include "render/GpuBuffers.h"
#include "render/RenderTargets.h"
#include "render/Textures.h"
#include "scene/CameraLens.h"
#include "viewport/FrameState.h"
#include "viewport/VideoRecording.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportIcons.h"
#include "viewport/ViewportUi.h"

#include "imgui.h"

#include <print>

namespace {

std::pair<uvec2, mtl::Extent2D> GetCaptureRegion(const state::Scene &r) {
    const auto full = r.Context.get<const RenderTargets>().Resources->FinalColorImage.Extent;
    const auto camera = LookThroughCameraEntity(r);
    const auto cd = camera != state::Null ? LensOf(r, camera) : std::nullopt;
    if (!cd) return {{0, 0}, full};

    const auto cam_aspect = AspectRatio(*cd);
    const auto ratio = LookThroughFrameRatio(cam_aspect, float(full.Width) / float(full.Height));
    // yuv420p requires even width and height.
    const auto w = uint32_t(float(full.Height) * cam_aspect * ratio) & ~1u;
    const auto h = uint32_t(float(full.Height) * ratio) & ~1u;
    return {{(full.Width - w) / 2, (full.Height - h) / 2}, {w, h}};
}
} // namespace

void InitViewportMedia(state::Scene &r) {
    LoadViewportIcons(r);
}

void DeinitViewportMedia(state::Scene &r) {
    r.Context.erase<ViewportIcons>();
}

void DisplayViewport(state::Scene &r, state::Entity viewport) {
    auto &dl = *ImGui::GetWindowDrawList();
    dl.ChannelsSetCurrent(0);
    if (const auto &targets = r.Context.get<const RenderTargets>(); targets.Resources) {
        const auto p = ImGui::GetCursorScreenPos();
        const auto extent = r.Context.get<ViewportExtent>().Value;
        dl.AddImage(mtl::ImGuiTextureId(*targets.Resources->FinalColorImage), p, p + ImVec2{float(extent.x), float(extent.y)});
    }

    dl.ChannelsSetCurrent(1);
    DrawOverlay(r, viewport, r.Context.get<FrameState>());
}

// Intentionally mutates VideoRecording outside Apply (not replayed).
void StartRecording(state::Scene &r, state::Entity viewport, const std::filesystem::path &path, int fps, bool with_audio) {
    r.remove<VideoRecording>(viewport);
    EndAudioCapture(r);
    if (!r.Context.get<const RenderTargets>().Resources) {
        std::println(stderr, "StartRecording: render resources not ready");
        return;
    }
    const auto region = GetCaptureRegion(r);
    const auto &ctx = r.Context.get<const mtl::Context>();
    // Zero selects video-only encoding, identical to a recording made without audio.
    // Render one audio frame per captured video frame when live device capture is unavailable.
    const auto device_rate = with_audio ? BeginAudioCapture(r) : 0u;
    // The offline render follows the same rate the modal bank was built at, so a headless capture and the bank agree at any AUDIO_SAMPLE_RATE.
    const auto offline_rate = with_audio && device_rate == 0 ? DeviceSampleRate(r) : 0u;
    const auto audio_rate = device_rate ? device_rate : offline_rate;
    // Video playback uses device units.
    // WAV measurement output remains in pascals.
    const bool monitor = with_audio && path.extension() != ".wav";
    r.emplace<VideoRecording>(viewport, VideoRecording{.Recorder = std::make_unique<VideoRecorder>(ctx, path, region.first.x, region.first.y, region.second, fps, audio_rate), .Region = region, .Monitor = monitor, .OfflineRate = offline_rate, .Fps = fps});
}

bool IsRecording(const state::Scene &r, state::Entity viewport) {
    const auto *rec = r.try_get<VideoRecording>(viewport);
    return rec && rec->Recorder && rec->Recorder->IsActive();
}

uint64_t CapturedFrameCount(const state::Scene &r, state::Entity viewport) {
    const auto *rec = r.try_get<VideoRecording>(viewport);
    return rec && rec->Recorder ? rec->Recorder->CapturedFrameCount() : 0;
}

void CaptureRecordFrame(state::Scene &r, state::Entity viewport) {
    const auto &targets = r.Context.get<const RenderTargets>();
    auto *rec = r.try_edit<VideoRecording>(viewport);
    if (!rec || !rec->Recorder || !rec->Recorder->IsActive() || !targets.Resources) return;
    if (GetCaptureRegion(r) != rec->Region) {
        std::println(stderr, "Viewport: capture region changed; stopping recording.");
        r.remove<VideoRecording>(viewport);
        return;
    }
    // Drain all device audio produced since the last frame to preserve wall-clock duration.
    // A no-op when recording without audio.
    rec->Drained.clear();
    if (rec->OfflineRate > 0) {
        const double owed = double(rec->OfflineRate) / double(rec->Fps) + rec->OfflineCarry;
        const auto whole = uint32_t(owed);
        rec->OfflineCarry = owed - double(whole);
        RenderAudioOffline(r, viewport, rec->Drained, whole);
    } else {
        DrainAudioCapture(r, rec->Drained);
    }
    if (rec->Monitor) MonitorFrames(r, rec->Drained, rec->Limiter);
    rec->Recorder->CaptureAudio(rec->Drained);
    rec->Recorder->CaptureFrame(targets.Resources->FinalColorImage);
}

std::expected<ViewportImageRgba8, std::string> ReadbackViewportImage(state::Scene &r) {
    const auto &targets = r.Context.get<const RenderTargets>();
    if (!targets.Resources) return std::unexpected{"render resources not ready"};

    const auto [offset, extent] = GetCaptureRegion(r);
    if (extent.Width == 0 || extent.Height == 0) return std::unexpected{"viewport extent is zero"};

    const auto &ctx = r.Context.get<const mtl::Context>();
    auto pixels = ReadbackImageRgba8(ctx, targets.Resources->FinalColorImage, offset.x, offset.y, extent);
    // Format::Color is BGRA, so red and blue trade places.
    for (size_t i = 0; i < pixels.size(); i += 4) std::swap(pixels[i], pixels[i + 2]);

    return ViewportImageRgba8{std::move(pixels), extent.Width, extent.Height};
}

std::string DebugBufferHeapUsage(const state::Scene &r) {
    return r.Context.get<const GpuBuffers>().Ctx.DebugHeapUsage();
}
