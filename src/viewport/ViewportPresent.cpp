#include "viewport/Viewport.h"

#include "Camera.h"
#include "VideoRecorder.h"
#include "audio/AudioSystem.h"
#include "audio/ModalAudio.h"
#include "render/GpuBuffers.h"
#include "render/OneShotGpu.h"
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
constexpr vk::Extent2D ToExtent2D(vk::Extent3D extent) { return {extent.width, extent.height}; }

// The ImGui texture handle for the final color image, recreated when a resize swaps the image.
struct ViewportTextureState {
    std::unique_ptr<mvk::ImGuiTexture> Texture;
    vk::ImageView View{}; // The FinalColorImage view Texture was built from; recreate when it changes.
};

// Present on the viewport entity iff recording is active.
struct VideoRecording {
    std::unique_ptr<VideoRecorder> Recorder;
    std::pair<vk::Offset3D, vk::Extent2D> Region; // Locked at StartRecording.
    std::vector<float> Drained{}; // Scratch the captured audio is drained through each frame.
    // Set when no device produces audio, so each captured frame renders its own share on this thread.
    uint32_t OfflineRate{0};
    double OfflineCarry{0}; // Fractional frames owed, so a non-integral rate over fps stays in step.
    int Fps{0};
};

std::pair<vk::Offset3D, vk::Extent2D> GetCaptureRegion(const entt::registry &r) {
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto full = ToExtent2D(pipelines.Main.Resources->FinalColorImage.Extent);
    const auto camera = LookThroughCameraEntity(r);
    const auto *cd = camera != entt::null ? r.try_get<Camera>(camera) : nullptr;
    if (!cd) return {{0, 0, 0}, full};

    const auto cam_aspect = AspectRatio(*cd);
    const auto ratio = LookThroughFrameRatio(cam_aspect, float(full.width) / float(full.height));
    // yuv420p requires even width/height.
    const auto w = uint32_t(float(full.height) * cam_aspect * ratio) & ~1u;
    const auto h = uint32_t(float(full.height) * ratio) & ~1u;
    return {{int32_t(full.width - w) / 2, int32_t(full.height - h) / 2, 0}, {w, h}};
}
} // namespace

void InitViewportMedia(entt::registry &r) {
    LoadViewportIcons(r);
    InitAudioSystem(r);
    r.ctx().emplace<ViewportTextureState>();
}

void DeinitViewportMedia(entt::registry &r) {
    r.ctx().erase<ViewportTextureState>();
    r.ctx().erase<ViewportIcons>();
    r.ctx().erase<ModalAudio>();
}

void DisplayViewport(entt::registry &r, entt::entity viewport) {
    auto &dl = *ImGui::GetWindowDrawList();
    dl.ChannelsSetCurrent(0);
    // Recreate the ImGui texture when the final color image view changed (a resize swaps the image).
    auto &tex = r.ctx().get<ViewportTextureState>();
    if (const auto &pipelines = r.ctx().get<const Pipelines>(); pipelines.Main.Resources) {
        if (const vk::ImageView view = *pipelines.Main.Resources->FinalColorImage.View; view != tex.View) {
            tex.Texture = std::make_unique<mvk::ImGuiTexture>(r.ctx().get<const VulkanResources>().Device, view, vec2{0, 1}, vec2{1, 0});
            tex.View = view;
        }
    }
    if (const auto &t_ptr = tex.Texture) {
        const auto p = ImGui::GetCursorScreenPos();
        const auto extent = r.ctx().get<ViewportExtent>().Value;
        const auto &t = *t_ptr;
        dl.AddImage(ImTextureID(VkDescriptorSet(t.DescriptorSet)), p, p + ImVec2{float(extent.x), float(extent.y)}, std::bit_cast<ImVec2>(t.Uv0), std::bit_cast<ImVec2>(t.Uv1));
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
    const auto &vk = r.ctx().get<const VulkanResources>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    // Zero means video only, which leaves the encode byte-identical to a recording made without audio.
    // With no device to capture, the audio is rendered here instead, one frame's worth per captured frame.
    const auto device_rate = with_audio ? BeginAudioCapture(r) : 0u;
    // The offline render follows the same rate the modal bank was built at, so a headless capture and the bank agree at any AUDIO_SAMPLE_RATE.
    const auto offline_rate = with_audio && device_rate == 0 ? DeviceSampleRate(r) : 0u;
    const auto audio_rate = device_rate ? device_rate : offline_rate;
    r.emplace<VideoRecording>(viewport, VideoRecording{.Recorder = std::make_unique<VideoRecorder>(vk, buffers.Ctx, path, region.first, region.second, fps, audio_rate), .Region = region, .OfflineRate = offline_rate, .Fps = fps});
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
    rec->Recorder->CaptureAudio(rec->Drained);
    rec->Recorder->CaptureFrame(*pipelines.Main.Resources->FinalColorImage.Image);
}

std::expected<ViewportImageRgba8, std::string> ReadbackViewportImage(entt::registry &r) {
    const auto &pipelines = r.ctx().get<const Pipelines>();
    if (!pipelines.Main.Resources) return std::unexpected{"render resources not ready"};

    const auto [offset, extent] = GetCaptureRegion(r);
    if (extent.width == 0 || extent.height == 0) return std::unexpected{"viewport extent is zero"};

    const auto &vk = r.ctx().get<const VulkanResources>();
    const auto &one_shot = r.ctx().get<const OneShotGpu>();
    const auto bgra = ReadbackImageRgba8(vk, r.ctx().get<GpuBuffers>().Ctx, *one_shot.Pool, *one_shot.Fence, *pipelines.Main.Resources->FinalColorImage.Image, offset, extent);

    // Convert BGRA (Format::Color) to RGBA and flip vertically: the viewport renders with a
    // negative-height Vulkan viewport, so row 0 in image memory is the bottom of the screen.
    std::vector<std::byte> rgba8(bgra.size());
    for (uint32_t y = 0; y < extent.height; ++y) {
        const std::byte *src = bgra.data() + size_t(y) * extent.width * 4;
        std::byte *dst = rgba8.data() + size_t(extent.height - 1 - y) * extent.width * 4;
        for (uint32_t x = 0; x < extent.width; ++x) {
            dst[x * 4 + 0] = src[x * 4 + 2];
            dst[x * 4 + 1] = src[x * 4 + 1];
            dst[x * 4 + 2] = src[x * 4 + 0];
            dst[x * 4 + 3] = src[x * 4 + 3];
        }
    }

    return ViewportImageRgba8{std::move(rgba8), extent.width, extent.height};
}

std::string DebugBufferHeapUsage(const entt::registry &r) {
    return r.ctx().get<const GpuBuffers>().Ctx.DebugHeapUsage();
}
