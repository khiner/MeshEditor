#include "viewport/Viewport.h"

#include "Camera.h"
#include "VideoRecorder.h"
#include "audio/AudioSystem.h"
#include "metal/ImGuiTexture.h"
#include "render/GpuBuffers.h"
#include "render/Pipelines.h"
#include "render/Textures.h"
#include "viewport/FrameState.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportIcons.h"
#include "viewport/ViewportUi.h"

#include "imgui.h"
#include <entt/entity/registry.hpp>

#include <print>

namespace {
// Present on the viewport entity iff recording is active.
struct VideoRecording {
    std::unique_ptr<VideoRecorder> Recorder;
    std::pair<uvec2, mtl::Extent2D> Region; // Locked at StartRecording.
    std::vector<float> Drained{}; // Scratch the captured audio is drained through each frame.
    // Set for a recording to be played back, which takes the master mix at the monitor's level rather than in pascals.
    bool Monitor{false};
    MonitorLimiter Limiter{}; // This recording's own envelope, so monitoring never touches the device's.
    // Set when no device produces audio, so each captured frame renders its own share on this thread.
    uint32_t OfflineRate{0};
    double OfflineCarry{0}; // Fractional frames owed, so a non-integral rate over fps stays in step.
    int Fps{0};
};

std::pair<uvec2, mtl::Extent2D> GetCaptureRegion(const entt::registry &r) {
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto full = pipelines.Main.Resources->FinalColorImage.Extent;
    const auto camera = LookThroughCameraEntity(r);
    const auto *cd = camera != entt::null ? r.try_get<Camera>(camera) : nullptr;
    if (!cd) return {{0, 0}, full};

    const auto cam_aspect = AspectRatio(*cd);
    const auto ratio = LookThroughFrameRatio(cam_aspect, float(full.Width) / float(full.Height));
    // yuv420p requires even width and height.
    const auto w = uint32_t(float(full.Height) * cam_aspect * ratio) & ~1u;
    const auto h = uint32_t(float(full.Height) * ratio) & ~1u;
    return {{(full.Width - w) / 2, (full.Height - h) / 2}, {w, h}};
}
} // namespace

void InitViewportMedia(entt::registry &r) {
    LoadViewportIcons(r);
}

void DeinitViewportMedia(entt::registry &r) {
    r.ctx().erase<ViewportIcons>();
}

void DisplayViewport(entt::registry &r, entt::entity viewport) {
    auto &dl = *ImGui::GetWindowDrawList();
    dl.ChannelsSetCurrent(0);
    if (const auto &pipelines = r.ctx().get<const Pipelines>(); pipelines.Main.Resources) {
        const auto p = ImGui::GetCursorScreenPos();
        const auto extent = r.ctx().get<ViewportExtent>().Value;
        dl.AddImage(mtl::ImGuiTextureId(*pipelines.Main.Resources->FinalColorImage), p, p + ImVec2{float(extent.x), float(extent.y)});
    }

    dl.ChannelsSetCurrent(1);
    DrawOverlay(r, viewport, r.ctx().get<FrameState>());
}

// Intentionally mutates VideoRecording outside Apply (not replayed).
void StartRecording(entt::registry &r, entt::entity viewport, const std::filesystem::path &path, int fps, bool with_audio) {
    r.remove<VideoRecording>(viewport);
    EndAudioCapture(r);
    const auto &pipelines = r.ctx().get<const Pipelines>();
    if (!pipelines.Main.Resources) {
        std::println(stderr, "StartRecording: render resources not ready");
        return;
    }
    const auto region = GetCaptureRegion(r);
    const auto &ctx = r.ctx().get<const mtl::Context>();
    // Zero means video only, which leaves the encode byte-identical to a recording made without audio.
    // With no device to capture, the audio is rendered here instead, one frame's worth per captured frame.
    const auto device_rate = with_audio ? BeginAudioCapture(r) : 0u;
    // The offline render follows the same rate the modal bank was built at, so a headless capture and the bank agree at any AUDIO_SAMPLE_RATE.
    const auto offline_rate = with_audio && device_rate == 0 ? DeviceSampleRate(r) : 0u;
    const auto audio_rate = device_rate ? device_rate : offline_rate;
    // A video is played back, so its track is in device units. A wav is a measurement and stays in pascals.
    const bool monitor = with_audio && path.extension() != ".wav";
    r.emplace<VideoRecording>(viewport, VideoRecording{.Recorder = std::make_unique<VideoRecorder>(ctx, path, region.first.x, region.first.y, region.second, fps, audio_rate), .Region = region, .Monitor = monitor, .OfflineRate = offline_rate, .Fps = fps});
}

bool IsRecording(const entt::registry &r, entt::entity viewport) {
    const auto *rec = r.try_get<VideoRecording>(viewport);
    return rec && rec->Recorder && rec->Recorder->IsActive();
}

uint64_t CapturedFrameCount(const entt::registry &r, entt::entity viewport) {
    const auto *rec = r.try_get<VideoRecording>(viewport);
    return rec && rec->Recorder ? rec->Recorder->CapturedFrameCount() : 0;
}

void CaptureRecordFrame(entt::registry &r, entt::entity viewport) {
    const auto &pipelines = r.ctx().get<const Pipelines>();
    auto *rec = r.try_get<VideoRecording>(viewport);
    if (!rec || !rec->Recorder || !rec->Recorder->IsActive() || !pipelines.Main.Resources) return;
    if (GetCaptureRegion(r) != rec->Region) {
        std::println(stderr, "Viewport: capture region changed; stopping recording.");
        r.remove<VideoRecording>(viewport); // Intentional direct registry mutation outside Apply
        return;
    }
    // Hand over whatever the device produced since the last captured frame, so the muxed track runs at wall-clock length rather than at one buffer per video frame.
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
    rec->Recorder->CaptureFrame(pipelines.Main.Resources->FinalColorImage);
}

std::expected<ViewportImageRgba8, std::string> ReadbackViewportImage(entt::registry &r) {
    const auto &pipelines = r.ctx().get<const Pipelines>();
    if (!pipelines.Main.Resources) return std::unexpected{"render resources not ready"};

    const auto [offset, extent] = GetCaptureRegion(r);
    if (extent.Width == 0 || extent.Height == 0) return std::unexpected{"viewport extent is zero"};

    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto pixels = ReadbackImageRgba8(ctx, pipelines.Main.Resources->FinalColorImage, offset.x, offset.y, extent);
    // Format::Color is BGRA, so red and blue trade places.
    for (size_t i = 0; i < pixels.size(); i += 4) std::swap(pixels[i], pixels[i + 2]);

    return ViewportImageRgba8{std::move(pixels), extent.Width, extent.Height};
}

std::string DebugBufferHeapUsage(const entt::registry &r) {
    return r.ctx().get<const GpuBuffers>().Ctx.DebugHeapUsage();
}
