#include "Compress.h"
#include "File.h"
#include "FileDialog.h"
#include "LogEnabled.h"
#include "Paths.h"
#include "ProcessEvents.h"
#include "TransformMath.h"
#include "Window.h"
#include "action/ActionApply.h"
#include "action/ActionIndex.h"
#include "action/Build.h"
#include "action/Emit.h"
#include "action/Errors.h"
#include "action/Io.h"
#include "action/Log.h"
#include "action/View.h"
#include "animation/AnimationData.h"
#include "animation/TimelineUi.h"
#include "armature/ArmatureComponents.h"
#include "audio/AudioDevice.h"
#include "audio/AudioSystem.h"
#include "audio/AudioTypes.h"
#include "audio/ModalModelFile.h"
#include "gizmo/TransformGizmoTypes.h"
#include "image/ImageEncode.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "object/ExtrasComponents.h"
#include "physics/PhysicsTypes.h"
#include "render/MaterialComponents.h"
#include "render/Profile.h"
#include "scene/Entity.h"
#include "scene/SceneControlsUi.h"
#include "scene/WorldTransform.h"
#include "snapshot/ReplayTestFixture.h"
#include "snapshot/SaveState.h"
#include "snapshot/SceneSnapshot.h"
#include "viewport/FrameState.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportIcons.h"
#include "viewport/ViewportUi.h"
#include "vulkan/VulkanContext.h"

#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "imgui_internal.h"
#include "implot.h"
#include "imspinner_demo.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <entt/entity/registry.hpp>

#include <csignal>
#include <exception>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <print>
#include <set>

#include <fcntl.h>
#include <unistd.h>

static_assert(null_entity == entt::null, "null_entity does not match entt::null");

using std::ranges::any_of, std::ranges::all_of;

namespace fs = std::filesystem;

// #define IMGUI_UNLIMITED_FRAME_RATE

namespace {
void CheckVk(vk::Result err) {
    if (err != vk::Result::eSuccess) throw std::runtime_error(std::format("Vulkan error: {}", vk::to_string(err)));
}

bool RebuildSwapchain = false;
void RenderFrame(vk::Device device, vk::Queue queue, ImGui_ImplVulkanH_Window &wd, ImDrawData *draw_data) {
    auto *image_acquired_semaphore = wd.FrameSemaphores[wd.SemaphoreIndex].ImageAcquiredSemaphore;
    const auto err = device.acquireNextImageKHR(wd.Swapchain, UINT64_MAX, image_acquired_semaphore, nullptr, &wd.FrameIndex);
    if (err == vk::Result::eErrorOutOfDateKHR || err == vk::Result::eSuboptimalKHR) {
        RebuildSwapchain = true;
        return;
    }
    CheckVk(err);

    const auto &fd = wd.Frames[wd.FrameIndex];
    const vk::Fence fd_fence{fd.Fence};
    CheckVk(device.waitForFences(fd_fence, true, UINT64_MAX));
    device.resetFences(fd_fence);
    device.resetCommandPool(fd.CommandPool);
    const vk::CommandBuffer command_buffer{fd.CommandBuffer};
    command_buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    constexpr static vk::ClearValue clear_color{{0.45f, 0.55f, 0.60f, 1.f}};
    command_buffer.beginRenderPass({wd.RenderPass, fd.Framebuffer, {{0, 0}, {uint32_t(wd.Width), uint32_t(wd.Height)}}, 1, &clear_color}, vk::SubpassContents::eInline);
    ImGui_ImplVulkan_RenderDrawData(draw_data, fd.CommandBuffer);
    command_buffer.endRenderPass();
    command_buffer.end();

    const vk::Semaphore wait_semaphores[]{image_acquired_semaphore};
    const vk::PipelineStageFlags wait_stage{vk::PipelineStageFlagBits::eColorAttachmentOutput};
    const vk::CommandBuffer command_buffers[]{command_buffer};
    const vk::Semaphore signal_semaphores[]{wd.FrameSemaphores[wd.SemaphoreIndex].RenderCompleteSemaphore};
    queue.submit(vk::SubmitInfo{wait_semaphores, wait_stage, command_buffers, signal_semaphores}, fd_fence);
}
void PresentFrame(vk::Queue queue, ImGui_ImplVulkanH_Window &wd) {
    if (RebuildSwapchain) return;

    const vk::Semaphore wait_semaphores[]{wd.FrameSemaphores[wd.SemaphoreIndex].RenderCompleteSemaphore};
    const vk::SwapchainKHR swapchains[]{wd.Swapchain};
    const uint32_t image_indices[]{wd.FrameIndex};
    auto result = queue.presentKHR({wait_semaphores, swapchains, image_indices});
    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
        RebuildSwapchain = true;
        return;
    }
    CheckVk(result);

    wd.SemaphoreIndex = (wd.SemaphoreIndex + 1) % wd.SemaphoreCount; // Now we can use the next set of semaphores.
}

using namespace ImGui;

enum class FontFamily {
    Main,
    Monospace,
    Count
};

constexpr float FontAtlasScale = 2; // Rasterize to a scaled-up texture and scale down the font size globally, for sharper text.
ImFont *MainFont{nullptr}, *MonospaceFont{nullptr};
ImFont *AddFont(FontFamily family, const std::string_view font_file) {
    static const auto FontsPath = Paths::Res() / "fonts";
    static constexpr auto PixelsForFamily = [] {
        // These are eyeballed.
        std::array<uint, size_t(FontFamily::Count)> v{};
        v[size_t(FontFamily::Main)] = 15;
        v[size_t(FontFamily::Monospace)] = 17;
        return v;
    }();
    return GetIO().Fonts->AddFontFromFileTTF((FontsPath / font_file).c_str(), PixelsForFamily[size_t(family)] * FontAtlasScale);
}
void InitFonts(float scale = 1.f) {
    MainFont = AddFont(FontFamily::Main, "Inter-Regular.ttf");
    MonospaceFont = AddFont(FontFamily::Monospace, "JetBrainsMono-Regular.ttf");
    ImGui::GetIO().FontGlobalScale = scale / FontAtlasScale;
}

} // namespace

/*
namespace MeshEditor {
// Returns true if the font was changed.
// **Only call `ImGui::PopFont` if `PushFont` returns true.**
bool PushFont(FontFamily family) {
    auto *new_font = family == FontFamily::Main ? MainFont : MonospaceFont;
    if (ImGui::GetFont() == new_font) return false;

    ImGui::PushFont(new_font);
    return true;
}
} // namespace MeshEditor
*/

namespace {
struct GltfSample {
    std::string Label;
    fs::path Path;
    std::set<std::string> Extensions; // top-level "extensionsUsed"
};

struct GltfSampleTree {
    std::map<std::string, GltfSampleTree> Children;
    std::vector<GltfSample> Files;
};

// Read a glTF/glb file's top-level "extensionsUsed" array without constructing a full Asset.
// Scans the JSON portion of the file (whole .gltf, or the JSON chunk of a .glb) for
// `"extensionsUsed":[ ... ]` and pulls each quoted name. Cheap enough to run on every sample at scan time.
std::set<std::string> ReadExtensionsUsed(const fs::path &path) {
    std::ifstream f{path, std::ios::binary};
    if (!f) return {};
    const std::string json = [&] {
        if (path.extension() != ".glb") return std::string{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
        // GLB: 12-byte container header, then 8-byte chunk header (length, type), then JSON bytes.
        char header[20];
        if (!f.read(header, sizeof(header)) || std::memcmp(header, "glTF", 4) != 0) return std::string{};
        uint32_t chunk_len{}, chunk_type{};
        std::memcpy(&chunk_len, header + 12, 4);
        std::memcpy(&chunk_type, header + 16, 4);
        if (chunk_type != 0x4E4F534A /* 'JSON' */) return std::string{};
        std::string s(chunk_len, '\0');
        f.read(s.data(), chunk_len);
        return s;
    }();
    const auto key_pos = json.find("\"extensionsUsed\"");
    if (key_pos == std::string::npos) return {};
    const auto open = json.find('[', key_pos);
    if (open == std::string::npos) return {};
    const auto close = json.find(']', open);
    if (close == std::string::npos) return {};
    std::set<std::string> result;
    for (auto i = open + 1; i < close;) {
        const auto str_open = json.find('"', i);
        if (str_open == std::string::npos || str_open >= close) break;
        const auto str_close = json.find('"', str_open + 1);
        if (str_close == std::string::npos || str_close >= close) break;
        result.insert(json.substr(str_open + 1, str_close - str_open - 1));
        i = str_close + 1;
    }
    return result;
}

// Recursively collect every .glb/.gltf under `root`. No stem-dedupe so variant subdirs
// (e.g. glTF-Sample-Assets `Models/<Name>/glTF-IBL/<Name>.gltf`) round-trip into the tree
// and contribute their own `extensionsUsed` to the filter set.
std::vector<GltfSample> CollectGltfSamples(const fs::path &root) {
    if (!fs::is_directory(root)) return {};
    std::vector<GltfSample> samples;
    for (const auto &entry : fs::recursive_directory_iterator(root)) {
        const auto ext = entry.path().extension();
        if (entry.is_regular_file() && (ext == ".glb" || ext == ".gltf")) {
            samples.emplace_back(entry.path().filename().string(), entry.path(), ReadExtensionsUsed(entry.path()));
        }
    }
    std::ranges::sort(samples, [](const auto &a, const auto &b) { return a.Path < b.Path; });
    return samples;
}

// Tree mirroring the directory structure under `root`. Leaves always show the real filename, after collapsing
// redundant levels (in order):
//   - Merge a dir with no files and a single child into that child (AnimatedCube/glTF/ -> AnimatedCube/).
//   - Flatten a dir holding one file whose stem repeats the dir name (AnimatedCube/AnimatedCube.gltf -> AnimatedCube.gltf).
// So a single-variant model flattens fully, while a multi-variant model (Box/{glTF,glTF-Binary,...}) keeps its variants.
GltfSampleTree BuildGltfSampleTree(const fs::path &root) {
    GltfSampleTree tree;
    for (auto &s : CollectGltfSamples(root)) {
        auto *node = &tree;
        for (const auto &c : s.Path.lexically_relative(root).parent_path()) {
            node = &node->Children[c.string()];
        }
        node->Files.emplace_back(std::move(s));
    }
    const auto flatten_named = [](this auto &&self, GltfSampleTree &n) -> void {
        for (auto it = n.Children.begin(); it != n.Children.end();) {
            self(it->second);
            auto &child = it->second;
            if (child.Children.empty() && child.Files.size() == 1 && it->first == child.Files.front().Path.stem().string()) {
                n.Files.emplace_back(std::move(child.Files.front()));
                it = n.Children.erase(it);
            } else {
                ++it;
            }
        }
    };
    const auto merge_sole_child = [](this auto &&self, GltfSampleTree &n) -> void {
        for (auto &[_, child] : n.Children) self(child);
        while (n.Files.empty() && n.Children.size() == 1) {
            auto child = std::move(n.Children.begin()->second);
            n.Children = std::move(child.Children);
            n.Files = std::move(child.Files);
        }
    };
    merge_sole_child(tree);
    flatten_named(tree);
    return tree;
}

// The File menu's sample-asset trees, scanned once off-thread at startup.
struct GltfSampleTrees {
    GltfSampleTree Examples, SampleAssets, Physics;
    std::set<std::string> SampleAssetsExtensions;
};

GltfSampleTrees BuildSampleTrees() {
    GltfSampleTrees t;
    t.Examples = BuildGltfSampleTree(Paths::Res() / "examples");
#ifdef GLTF_SAMPLE_ASSETS_DIR
    t.SampleAssets = BuildGltfSampleTree(fs::path{GLTF_SAMPLE_ASSETS_DIR} / "Models");
    [&](this auto &&self, const GltfSampleTree &n) -> void {
        for (const auto &f : n.Files) t.SampleAssetsExtensions.insert_range(f.Extensions);
        for (const auto &[_, c] : n.Children) self(c);
    }(t.SampleAssets);
#endif
#ifdef GLTF_PHYSICS_DIR
    t.Physics = BuildGltfSampleTree(GLTF_PHYSICS_DIR);
#endif
    return t;
}

std::future<GltfSampleTrees> SampleTreesFuture;

// Apply `action` now and settle the scene's derived state, for actions that must take effect outside the main loop.
template<typename ActionType> void Perform(entt::registry &r, entt::entity viewport, ActionType action) {
    action::ApplyNow(r, viewport, std::move(action));
    ProcessComponentEvents(r, viewport);
}

// Finish in-flight GPU work and stop playback, so scene structure can be safely torn down.
void QuiesceScene(entt::registry &r, entt::entity viewport) {
    r.ctx().get<const VulkanResources>().Device.waitIdle();
    const auto &playback = r.get<const TimelinePlayback>(viewport);
    if (playback.Playing) action::ApplyNow(r, viewport, action::timeline::TogglePlay{playback.CurrentFrame});
}

constexpr std::string_view SessionLogName{"session.actions"}, ProjectStateName{"project.state"}, ProjectExt{".project"}, ActionsExt{".actions"};
fs::path CurrentProjectPath;

// Replay action log, restore the current viewport extent and camera, and present.
void ReplayPreservingView(entt::registry &r, entt::entity viewport, const fs::path &log_path, uint64_t skip = 0) {
    const auto live_extent = r.ctx().get<ViewportExtent>().Value;
    auto live_view_cameras = GetViewCameraState(r, viewport);
    if (action::ReplayLog(r, viewport, log_path, [](entt::registry &r, entt::entity viewport) { ProcessComponentEvents(r, viewport); }, skip)) {
        r.ctx().get<ViewportExtent>().Value = live_extent;
        SetViewCameraState(r, viewport, std::move(live_view_cameras));
        PresentViewport(r, viewport);
    }
}

void StartScratchSession(entt::registry &r, entt::entity viewport) {
    QuiesceScene(r, viewport);
    action::StopLog();
    Paths::SetProject(action::ReserveRestoreSession());
    ClearScene(r, viewport);
    action::StartLog(Paths::Project() / SessionLogName);
    CurrentProjectPath.clear();
}

// Reset to a fresh scratch session with the default scene, or an empty one.
void NewScene(entt::registry &r, entt::entity viewport, bool empty) {
    StartScratchSession(r, viewport);
    if (!empty) action::Emit(action::io::LoadDefaultScene{});
}

// Replay a standalone `.actions` log into a fresh scratch session.
void ReplayLogIntoNewSession(entt::registry &r, entt::entity viewport, const fs::path &log_path) {
    StartScratchSession(r, viewport);
    ReplayPreservingView(r, viewport, log_path);
}

// Load a snapshot file and return its action-log position.
uint64_t LoadStateBase(entt::registry &r, entt::entity viewport, const fs::path &path) {
    const auto bytes = File::Read(path).value_or(std::vector<std::byte>{});
    r.ctx().get<const VulkanResources>().Device.waitIdle();
    ClearScene(r, viewport);
    snapshot::LoadState(r, bytes);
    ProcessComponentEvents(r, viewport);
    return r.all_of<ActionIndex>(viewport) ? r.get<ActionIndex>(viewport).Index : 0;
}

// Load the base snapshot (if any), replay the session log past the base's action index, and re-open the log for appending.
void OpenProjectDir(entt::registry &r, entt::entity viewport, const fs::path &working_dir) {
    QuiesceScene(r, viewport);
    action::StopLog();
    CurrentProjectPath.clear();
    Paths::SetProject(working_dir);
    const auto state_path = working_dir / ProjectStateName, log_path = working_dir / SessionLogName;
    uint64_t skip = 0;
    if (std::error_code ec; fs::exists(state_path, ec)) skip = LoadStateBase(r, viewport, state_path);
    else ClearScene(r, viewport);
    ReplayPreservingView(r, viewport, log_path, skip);
    action::StartLog(log_path, /*append=*/true);
}

// Decompress a `.project` archive into a fresh working directory and open it.
void OpenProjectFile(entt::registry &r, entt::entity viewport, const fs::path &archive_path) {
    const auto working_dir = action::ReserveRestoreSession();
    if (!Decompress(archive_path, working_dir)) {
        std::println(stderr, "Failed to open project '{}'", archive_path.string());
        return;
    }
    OpenProjectDir(r, viewport, working_dir);
    CurrentProjectPath = archive_path;
}

void OpenFile(entt::registry &r, entt::entity viewport, const fs::path &path) {
    if (const auto ext = path.extension(); ext == ProjectExt) OpenProjectFile(r, viewport, path);
    else if (ext == ActionsExt) ReplayLogIntoNewSession(r, viewport, path);
    else action::Emit(action::io::Load{.Path = path});
}

// Snapshot the scene and compress the working directory into `archive_path`.
void SaveProjectFile(entt::registry &r, entt::entity viewport, const fs::path &archive_path) {
    Perform(r, viewport, action::io::SaveState{.Path = Paths::Project() / ProjectStateName});
    const auto log_path = Paths::Project() / SessionLogName;
    action::StopLog(); // flush the log before archiving
    const bool ok = Compress(Paths::Project(), archive_path);
    action::StartLog(log_path, /*append=*/true);
    if (!ok) {
        std::println(stderr, "Failed to save project '{}'", archive_path.string());
        return;
    }
    CurrentProjectPath = archive_path;
}

// Drop the session log and modal media, keeping the current scene as a fresh snapshot.
void ClearHistory(entt::registry &r, entt::entity viewport) {
    r.emplace_or_replace<ActionIndex>(viewport); // reset log position (write outside Apply: session bookkeeping)
    Perform(r, viewport, action::io::SaveState{.Path = Paths::Project() / ProjectStateName});
    action::StopLog();
    std::error_code ec;
    fs::remove(Paths::Project() / SessionLogName, ec);
    fs::remove_all(ModalModelsDir(), ec);
    action::StartLog(Paths::Project() / SessionLogName);
}

#ifdef DEBUG_BUILD
// Validate replay then snapshot correctness, aborting on the first divergence:
// - Replay: replaying the current log onto a fresh scene must reproduce the saved image (writes a replay-test fixture on failure).
// - Round-trip: save, clear, restore must reproduce the saved image.
void ValidateRoundTrip(entt::registry &r, entt::entity viewport) {
    QuiesceScene(r, viewport);

    const auto current_log = Paths::Project() / SessionLogName;
    if (std::error_code ec; !fs::exists(current_log, ec)) {
        std::println(stderr, "[snapshot] replay SKIPPED (no log)");
    } else {
        const auto expected = snapshot::SnapshotSceneState(r);
        ReplayLogIntoNewSession(r, viewport, current_log);
        const auto actual = snapshot::SnapshotSceneState(r);
        if (const auto diff = snapshot::Compare(expected, actual); !diff.Equal) {
            std::println(stderr, "[snapshot] replay DIVERGED at byte {} (expected {} / actual {})", diff.FirstDifferingByte, expected.size(), actual.size());
            if (const auto fixture_dir = snapshot::WriteReplayTestFixture(current_log, expected, actual); !fixture_dir.empty()) {
                std::println(stderr, "[snapshot] wrote replay-test fixture to {}", fixture_dir.string());
            }
            std::abort();
        }
    }

    const auto before = snapshot::SaveState(r);
    ClearScene(r, viewport);
    snapshot::LoadState(r, before);
    ProcessComponentEvents(r, viewport);
    const auto after = snapshot::SaveState(r);
    if (const auto diff = snapshot::Compare(before, after); !diff.Equal) {
        std::println(stderr, "[snapshot] round-trip DIVERGED at byte {} (before {} / after {})", diff.FirstDifferingByte, before.size(), after.size());
        std::abort();
    }
    PresentViewport(r, viewport);
}
#endif

// Read back the viewport and write it to `path`, choosing the encoder by extension (defaulting to .webp).
// Returns the resolved output path on success.
std::expected<fs::path, std::string> SaveScreenshot(entt::registry &r, const fs::path &path) {
    auto image = ReadbackViewportImage(r);
    if (!image) return std::unexpected{std::move(image.error())};

    auto ext = path.extension().string();
    std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return std::tolower(c); });
    auto out_path = ext.empty() ? fs::path{path}.replace_extension(".webp") : path;
    const auto name = out_path.filename().string();
    const auto encoded = ext == ".jpg" || ext == ".jpeg" ? EncodeImageJpegRgba8(image->Pixels, image->Width, image->Height, 95, name) :
        ext == ".png"                                    ? EncodeImagePngRgba8(image->Pixels, image->Width, image->Height, name) :
                                                           EncodeImageWebpRgba8(image->Pixels, image->Width, image->Height, name);
    if (!encoded) return std::unexpected{std::move(encoded.error())};

    std::ofstream out{out_path, std::ios::binary};
    out.write(reinterpret_cast<const char *>(encoded->data()), std::streamsize(encoded->size()));
    if (!out) return std::unexpected{std::format("failed to write '{}'", out_path.string())};
    return out_path;
}

// Slide the view camera along its current view axis until the scene's longest visual side spans the middle half of the view. No-op if empty.
void FrameScene(entt::registry &r, entt::entity viewport, float aspect_ratio) {
    const auto &cam = r.get<const ViewCamera>(viewport);
    const auto *persp = std::get_if<Perspective>(&cam.Data); // The launch view camera is always perspective.
    if (!persp) return;

    // Keep the current orientation; measure each vertex against this camera's basis.
    const vec3 right = cam.Orientation * vec3{1, 0, 0}, up = cam.Orientation * vec3{0, 1, 0}, away = cam.Forward();
    // Half the real frustum tangents, so a vertex reaching the (narrower) edge only fills the middle half of the true view.
    const float ty = 0.5f * std::tan(persp->FieldOfViewRad * 0.5f), tx = aspect_ratio * ty;
    constexpr float lowest = std::numeric_limits<float>::lowest();
    float top = lowest, bottom = lowest, rgt = lowest, lft = lowest;
    BBox scene;
    for (const auto [e, ri, wt] : r.view<const RenderInstance, const WorldTransform>().each()) {
        BBox local;
        if (const auto *db = r.try_get<const DeformedBounds>(e); db) {
            local = db->Box;
        } else {
            const auto mesh = TryGetMesh(r, FindMeshEntity(r, e));
            if (!mesh) continue;
            local = mesh->GetBBox();
        }
        if (glm::any(glm::greaterThan(local.Min, local.Max))) continue;

        const auto m = ToMatrix(wt);
        for (int c = 0; c < 8; ++c) {
            const vec3 v{m * vec4{(c & 1) ? local.Max.x : local.Min.x, (c & 2) ? local.Max.y : local.Min.y, (c & 4) ? local.Max.z : local.Min.z, 1.f}};
            scene.Min = glm::min(scene.Min, v);
            scene.Max = glm::max(scene.Max, v);
            const float a = glm::dot(v, right), b = glm::dot(v, up), f = glm::dot(v, away);
            top = std::max(top, b / ty + f);
            bottom = std::max(bottom, -b / ty + f);
            rgt = std::max(rgt, a / tx + f);
            lft = std::max(lft, -a / tx + f);
        }
    }
    if (glm::any(glm::greaterThan(scene.Min, scene.Max))) return;

    const auto center = (scene.Min + scene.Max) * 0.5f;
    const float ca = glm::dot(center, right), cb = glm::dot(center, up), cf = glm::dot(center, away);
    const float distance = std::max({top - cb / ty, bottom + cb / ty, rgt - ca / tx, lft + ca / tx}) - cf;
    if (distance <= 0.f) return;

    // Clip planes bracket the scene depth so nothing is z-clipped.
    const float plane_reach = 6 * glm::length(scene.Max - scene.Min);
    auto fit = *persp;
    fit.FarClip = distance + plane_reach;
    fit.NearClip = std::max(distance - plane_reach, *fit.FarClip / 10000.f);

    r.replace<ViewCamera>(viewport, ViewCamera{center + distance * away, center, Camera{fit}});
}

// The default window size, doubling as the headless viewport extent.
constexpr uvec2 DefaultWindowSize{1280, 800};

// Capture options from the CLI. `--render` is a preset for the full scene corpus; `--screenshot`/`--record` target one output.
struct CaptureRequest {
    bool Play{false};
    float PlayDuration{0}; // 0 = run until playback completes one loop.
    int Fps{60};
    fs::path RecordPath{}, ScreenshotPath{};
    fs::path RenderBasename{}; // Output basename, no extension.
    std::optional<uint8_t> MotionBlurSteps{}; // Disengaged = leave the viewport's own setting alone.
    int BenchFrames{0}; // Headless: re-render every tick and exit after this many frames.
};

// Surface and clear any failures action handlers reported this frame. Returns true if there were any.
bool ReportActionErrors(entt::registry &r) {
    auto &errors = r.ctx().get<action::Errors>().Messages;
    if (errors.empty()) return false;
    for (const auto &message : errors) std::cerr << message << std::endl;
    errors.clear();
    return true;
}

// Seed this run's initial scene and session log. Returns false if the initial file failed to load.
bool SeedScene(entt::registry &r, entt::entity viewport, const CaptureRequest &capture, const char *initial_file, bool empty) {
    if (initial_file) {
        if (const fs::path path = initial_file; path.extension() == ProjectExt) {
            OpenProjectFile(r, viewport, path);
            return true;
        } else if (path.extension() == ActionsExt) {
            ReplayLogIntoNewSession(r, viewport, path);
            return true;
        }
    }
    if (capture.RenderBasename.empty()) {
        Paths::SetProject(action::ReserveRestoreSession());
        action::StartLog(Paths::Project() / SessionLogName);
    } else {
        Paths::SetProject(capture.RenderBasename.parent_path());
        action::StartLog(fs::path{capture.RenderBasename.string() + ".actions"});
    }
    if (initial_file) {
        Perform(r, viewport, action::io::Load{.Path = initial_file});
        return !ReportActionErrors(r);
    }
    if (!empty) Perform(r, viewport, action::io::LoadDefaultScene{});
    return true;
}

// Per-frame capture orchestration shared by the windowed and headless run loops: scene framing,
// playback start, the screenshot + material-variant sequence, video recording (with render mode's
// per-clip loops), and run completion.
struct CaptureDriver {
    // `fixed_step` runs the sim on a fixed-step, GPU-paced clock: one timeline frame per tick, every
    // tick captured, video fps = timeline fps. Render mode is always fixed-step and headless runs pass
    // true. Otherwise the sim runs at wall-clock rate and recording samples it every `1/Fps` seconds.
    CaptureDriver(entt::registry &r, entt::entity viewport, const CaptureRequest &capture, bool play, bool fixed_step)
        : Play(play), PlayDuration(capture.PlayDuration),
          FixedStep(fixed_step || !capture.RenderBasename.empty()),
          RecordPath(capture.RecordPath), ScreenshotPath(capture.ScreenshotPath), RenderBasename(capture.RenderBasename) {
        if (RenderMode()) {
            const auto with = [&](const char *ext) { return fs::path{capture.RenderBasename.string() + ext}; };
            const bool dynamic = !r.view<const PhysicsMotion>().empty() ||
                !r.view<const ArmatureAnimation>().empty() ||
                !r.view<const NodeTransformAnimation>().empty() ||
                !r.view<const MorphWeightAnimation>().empty();
            if (dynamic) RecordPath = with(".mp4");
            else ScreenshotPath = with(".webp");
        }
        const float timeline_fps = r.get<const TimelineRange>(viewport).Fps;
        RenderDt = 1.f / timeline_fps;
        RecordFps = FixedStep ? int(std::lround(timeline_fps)) : capture.Fps;
    }

    bool RenderMode() const { return !RenderBasename.empty(); }
    bool RecordingMode() const { return !RecordPath.empty(); }
    bool ScreenshotMode() const { return !ScreenshotPath.empty(); }
    bool Presenting() const { return Play || ScreenshotMode() || RecordingMode(); }

    bool DurationElapsed(const entt::registry &r, entt::entity viewport) const {
        if (PlayDuration <= 0) return false;
        const float elapsed = RecordingMode() ? float(CapturedFrameCount(r, viewport)) / float(RecordFps) : ElapsedPlayTime;
        return elapsed >= PlayDuration;
    }

    // Emit this frame's capture-driven actions. Call before ApplyEmitted, with `settled` true once
    // the viewport is at its final extent with the image built.
    void EmitFrameActions(entt::registry &r, entt::entity viewport, bool settled, uvec2 extent) {
        if (!ViewFramed && Presenting() && r.view<const Camera>().empty() && extent != uvec2{}) {
            FrameScene(r, viewport, float(extent.x) / float(extent.y));
            ViewFramed = settled;
        }
        // Start playback once settled, for play or video (the screenshot stays on the held frame).
        // Fixed-step recording waits one more tick, until recording has begun, so the start frame is captured.
        const bool ready = RenderMode() || (FixedStep && RecordingMode()) ? IsRecording(r, viewport) : (Play || RecordingMode());
        if (!PlaybackStarted && settled && ready) {
            action::Emit(action::timeline::StartPresentation{});
            PlaybackStarted = true;
        }
    }

    // Save/record the frame. Call after WaitForRender so the source image is coherent.
    // Returns true when capture is complete and the run should end.
    bool CaptureFrame(entt::registry &r, entt::entity viewport, bool settled) {
        bool done = false;
        // Save the image, then finish unless recording or a play duration is still running.
        if (ScreenshotMode() && !ScreenshotSaved && settled) {
            if (auto saved = SaveScreenshot(r, ScreenshotPath); saved) std::println("Saved screenshot: {}", saved->string());
            else std::println(stderr, "Screenshot: {}", saved.error());
            // After the default, save one image per material variant.
            const auto *mv = RenderMode() ? r.try_get<const MaterialVariants>(viewport) : nullptr;
            if (mv && NextRenderVariant < mv->Names.size()) {
                auto name = mv->Names[NextRenderVariant].empty() ? std::format("Variant {}", NextRenderVariant) : mv->Names[NextRenderVariant];
                std::ranges::replace(name, '/', '-');
                ScreenshotPath = fs::path{RenderBasename.string() + "." + name + ".webp"};
                action::Emit(action::UpdateOf<&MaterialVariants::Active>(viewport, std::optional{NextRenderVariant}));
                ++NextRenderVariant;
            } else {
                ScreenshotSaved = true;
                if (!RecordingMode() && PlayDuration <= 0) done = true;
            }
        }
        if (RecordingMode() && !IsRecording(r, viewport) && settled) {
            StartRecording(r, viewport, RecordPath, RecordFps);
            if (IsRecording(r, viewport)) NextCaptureNs = SDL_GetTicksNS();
            else done = true;
        }
        const bool loop_end = r.get<const TimelinePlayback>(viewport).CurrentFrame == r.get<const TimelineRange>(viewport).EndFrame;
        if (IsRecording(r, viewport)) {
            if (FixedStep) {
                // Fixed step: every tick is one timeline frame and each is captured.
                // Clip switches are Emitted, not Performed: a mid-loop Perform would advance playback an extra tick.
                CaptureRecordFrame(r, viewport);
                if (loop_end) {
                    if (RenderMode()) {
                        bool switched = false;
                        const auto switch_clips = [&]<typename Anim>() {
                            for (const auto [entity, anim] : r.view<const Anim>().each()) {
                                if (NextRenderClip < anim.Clips.size()) {
                                    action::Emit(action::UpdateOf<&Anim::ActiveClipIndex>(entity, NextRenderClip));
                                    switched = true;
                                }
                            }
                        };
                        switch_clips.template operator()<ArmatureAnimation>();
                        switch_clips.template operator()<MorphWeightAnimation>();
                        switch_clips.template operator()<NodeTransformAnimation>();
                        if (switched) ++NextRenderClip;
                        else done = true;
                    } else if (PlayDuration <= 0) {
                        done = true;
                    }
                }
            } else if (SDL_GetTicksNS() >= NextCaptureNs) {
                CaptureRecordFrame(r, viewport);
                NextCaptureNs += 1'000'000'000ULL / uint64_t(RecordFps);
            }
        } else if (FixedStep && !RecordingMode() && PlaybackStarted && PlayDuration <= 0 && loop_end) {
            // A duration-less fixed-step play run (headless --play) ends after one timeline loop.
            done = true;
        }
        return done;
    }

    bool Play;
    float PlayDuration;
    bool FixedStep;
    fs::path RecordPath, ScreenshotPath, RenderBasename;
    float RenderDt; // Fixed-step seconds per tick (one timeline frame).
    int RecordFps;
    uint64_t NextCaptureNs{0}; // Wall-clock recording: next capture time, initialized when recording starts.
    float ElapsedPlayTime{0}; // Caller-accumulated sim seconds, for the play-duration cap.
    uint32_t NextRenderClip{1}; // Next clip to capture once the current loop finishes.
    uint32_t NextRenderVariant{0}; // Next material variant to capture once the default image saves.
    bool PlaybackStarted{false}, ScreenshotSaved{false}, ViewFramed{false};
};

// Seed the scene and its session log, then enter presentation mode so the first rendered frame matches the capture.
CaptureDriver BeginCaptureSession(entt::registry &r, entt::entity viewport, const CaptureRequest &capture, const char *initial_file, bool empty, bool fixed_step) {
    const bool play = SeedScene(r, viewport, capture, initial_file, empty) && capture.Play;
    CaptureDriver driver{r, viewport, capture, play, fixed_step};
    // A benchmark run keeps the editor view, so frames and screenshots cover what the editor draws.
    if (driver.Presenting() && capture.BenchFrames == 0) Perform(r, viewport, action::timeline::EnterPresentation{});
    r.ctx().get<FrameState>().FixedFrameStep = driver.FixedStep;
    // Force motion blur on for the whole recording run (not still-screenshot renders).
    r.ctx().get<FrameState>().Capturing = driver.RecordingMode();
    if (capture.MotionBlurSteps) {
        Perform(r, viewport, action::UpdateOf<&ViewportDisplay::MotionBlur>(viewport, std::optional{MotionBlur{.Steps = *capture.MotionBlurSteps}}));
    }
    return driver;
}

// Resolve the executable base dir (read-only resources) and the writable per-user data dir.
void InitPaths() {
    const char *base = SDL_GetBasePath();
    char *user_data = SDL_GetPrefPath("", "MeshEditor");
    if (!base || !user_data) throw std::runtime_error(std::format("SDL path error: {}", SDL_GetError()));
    Paths::Init(base, user_data);
    SDL_free(user_data);
}

void run(const char *initial_file, bool quiet, bool empty, const CaptureRequest &capture) {
    LogEnabled = !quiet;

    const bool render_mode = !capture.RenderBasename.empty();

    SDL_SetHint(SDL_HINT_MAC_SCROLL_MOMENTUM, "1");
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) throw std::runtime_error(std::format("SDL_Init error: {}", SDL_GetError()));

    InitPaths();

    auto *window = SDL_CreateWindow(
        "MeshEditor", int(DefaultWindowSize.x), int(DefaultWindowSize.y),
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED | SDL_WINDOW_HIGH_PIXEL_DENSITY
    );

    uint32_t extensions_count = 0;
    const char *const *instance_extensions_raw = SDL_Vulkan_GetInstanceExtensions(&extensions_count);
    auto vc = std::make_unique<VulkanContext>(std::vector<const char *>{instance_extensions_raw, instance_extensions_raw + extensions_count});

    const std::array imgui_pool_sizes{
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 16},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 2}, // IMGUI_IMPL_VULKAN_MINIMUM_SAMPLER_POOL_SIZE
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 8},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 4},
    };
    auto imgui_descriptor_pool = vc->Device->createDescriptorPoolUnique({vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 64, imgui_pool_sizes});

    constexpr static uint32_t MinImageCount = 2;
    ImGui_ImplVulkanH_Window wd;
    { // Set up Vulkan window.
        VkSurfaceKHR surface;
        if (!SDL_Vulkan_CreateSurface(window, *vc->Instance, nullptr, &surface)) throw std::runtime_error("Failed to create Vulkan surface.\n");
        wd.Surface = surface;

        int w, h;
        SDL_GetWindowSize(window, &w, &h);

        if (auto res = vc->PhysicalDevice.getSurfaceSupportKHR(vc->QueueFamily, wd.Surface); res != VK_TRUE) {
            throw std::runtime_error("Error no WSI support on physical device 0\n");
        }

        const VkFormat requestSurfaceImageFormat[]{VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM};
        const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
        wd.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(vc->PhysicalDevice, wd.Surface, requestSurfaceImageFormat, (size_t)IM_COUNTOF(requestSurfaceImageFormat), requestSurfaceColorSpace);

#ifdef IMGUI_UNLIMITED_FRAME_RATE
        const bool unthrottled = true;
#else
        // Render mode is GPU-paced: content is fixed-step per tick, so present pacing only affects wall time.
        const bool unthrottled = render_mode;
#endif
        const VkPresentModeKHR unthrottled_modes[]{VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR};
        wd.PresentMode = unthrottled ?
            ImGui_ImplVulkanH_SelectPresentMode(vc->PhysicalDevice, wd.Surface, &unthrottled_modes[0], IM_COUNTOF(unthrottled_modes)) :
            VK_PRESENT_MODE_FIFO_KHR; // Always supported.
        ImGui_ImplVulkanH_CreateOrResizeWindow(*vc->Instance, vc->PhysicalDevice, *vc->Device, &wd, vc->QueueFamily, nullptr, w, h, MinImageCount, 0);
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    auto &io = GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad | ImGuiConfigFlags_DockingEnable;
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows
    io.IniFilename = nullptr; // Disable ImGui's .ini file saving
    io.ConfigDebugIgnoreFocusLoss = true; // Keep input state across Cmd+Tab so in-flight gizmo drags survive focus loss.
    io.ConfigDragClickToInputText = true; // A click-release without dragging turns a Drag field into a text input.

    StyleColorsDark();

    ImGui_ImplSDL3_InitForVulkan(window);
    ImGui_ImplVulkan_InitInfo init_info{
        .ApiVersion = VkApiVersion,
        .Instance = *vc->Instance,
        .PhysicalDevice = vc->PhysicalDevice,
        .Device = *vc->Device,
        .QueueFamily = vc->QueueFamily,
        .Queue = vc->Queue,
        .DescriptorPool = *imgui_descriptor_pool,
        .DescriptorPoolSize = 0,
        .MinImageCount = MinImageCount,
        .ImageCount = wd.ImageCount,
        .PipelineCache = VK_NULL_HANDLE,
        .PipelineInfoMain = {
            .RenderPass = wd.RenderPass,
            .Subpass = 0,
            .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
            .ExtraDynamicStates = {},
            .PipelineRenderingCreateInfo = {},
            .SwapChainImageUsage = {},
        },
        .PipelineInfoForViewports = {},
        .UseDynamicRendering = false,
        .Allocator = nullptr,
        .CheckVkResultFn = [](VkResult res) {
            if (res != 0) throw std::runtime_error(std::format("Vulkan error: {}", int(res)));
        },
        .MinAllocationSize = {},
        .CustomShaderVertCreateInfo = {},
        .CustomShaderFragCreateInfo = {},
    };
    ImGui_ImplVulkan_Init(&init_info);

    InitFonts();

    entt::registry r;
    const auto viewport = InitEngine(r, vc->Resources());
    InitViewportMedia(r);
    SetupScene(r, viewport); // Before the first frame reads viewport state.
    // Capture the DPI scale (only set during NewFrame) before priming DPI-scaled GPU state like edge-line width.
    ImGui_ImplSDL3_NewFrame();
    r.ctx().get<FrameState>().DisplayFramebufferScale = {io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y};
    ProcessComponentEvents(r, viewport); // Prime derived state before the first frame reads it.

    auto &audio_device = r.ctx().emplace<AudioDeviceResource>(r, viewport);
    ReconcileAudioDevice(audio_device, r.get<const AudioOutputConfig>(viewport), r.get<const AudioOutputMix>(viewport));

    SampleTreesFuture = std::async(std::launch::async, BuildSampleTrees);

    auto driver = BeginCaptureSession(r, viewport, capture, initial_file, empty, /*fixed_step=*/false);

    bool viewport_resizing{false}; // True while a resize drag is staged but not yet committed.
    bool done{false};
    WindowsState windows;
    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_MOUSE_WHEEL) {
                r.ctx().get<FrameState>().PreciseWheelDelta += vec2{-event.wheel.x, event.wheel.y};
                // SDL's pixel-derived deltas overscroll ImGui panels.
                constexpr float ImGuiWheelScale{0.3};
                event.wheel.x *= ImGuiWheelScale;
                event.wheel.y *= ImGuiWheelScale;
            }
            if (event.type == SDL_EVENT_DROP_FILE) OpenFile(r, viewport, event.drop.data);
            // SDL3 backend invalidates MousePos to -FLT_MAX on leave when no mouse button is held,
            // which flings a keyboard-initiated (G/R/S) transform offscreen when switching focus.
            if (event.type != SDL_EVENT_WINDOW_MOUSE_LEAVE || !TransformGizmo::IsUsing(r, viewport)) {
                ImGui_ImplSDL3_ProcessEvent(&event);
                done = event.type == SDL_EVENT_QUIT || (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window));
            }
        }
        if (driver.DurationElapsed(r, viewport)) done = true;
        FileDialog::Pump(); // Runs callbacks for file dialogs the user completed since last frame.

        if (RebuildSwapchain) {
            int w, h;
            SDL_GetWindowSize(window, &w, &h);
            if (w > 0 && h > 0) {
                ImGui_ImplVulkan_SetMinImageCount(MinImageCount);
                ImGui_ImplVulkanH_CreateOrResizeWindow(*vc->Instance, vc->PhysicalDevice, *vc->Device, &wd, vc->QueueFamily, nullptr, w, h, MinImageCount, 0);
                wd.FrameIndex = 0;
                RebuildSwapchain = false;
            }
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        driver.ElapsedPlayTime += io.DeltaTime;
        // Scene-affecting code reads FrameState::DeltaTime. `io.DeltaTime` is wall-clock, UI-only.
        r.ctx().get<FrameState>().DeltaTime = driver.FixedStep ? driver.RenderDt : io.DeltaTime;
        NewFrame();

        auto dockspace_id = DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_AutoHideTabBar);
        if (GetFrameCount() == 1) {
            auto controls_node_id = DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.3f, nullptr, &dockspace_id);
            auto extra_node_id = DockBuilderSplitNode(controls_node_id, ImGuiDir_Down, 0.4f, nullptr, &controls_node_id);
            auto animation_node_id = DockBuilderSplitNode(dockspace_id, ImGuiDir_Down, 0.1f, nullptr, &dockspace_id);
            DockBuilderDockWindow(windows.Debug.Name, extra_node_id);
            DockBuilderDockWindow(windows.ImGuiDemo.Name, extra_node_id);
            DockBuilderDockWindow(windows.ImSpinnerDemo.Name, extra_node_id);
            DockBuilderDockWindow(windows.ImPlotDemo.Name, extra_node_id);
            DockBuilderDockWindow(windows.SceneControls.Name, controls_node_id);
            DockBuilderDockWindow(windows.Animation.Name, animation_node_id);
            DockBuilderDockWindow(windows.Viewport.Name, dockspace_id);
        }

        if (BeginMainMenuBar()) {
            if (BeginMenu("File")) {
                if (BeginMenu("New")) {
                    if (MenuItem("Default")) NewScene(r, viewport, /*empty=*/false);
                    if (MenuItem("Empty")) NewScene(r, viewport, /*empty=*/true);
                    EndMenu();
                }
                if (MenuItem("Open")) {
                    static constexpr std::array filters{FileDialog::Filter{"MeshEditor project", "project"}, FileDialog::Filter{"Scene state", "state"}, FileDialog::Filter{"Action log", "actions"}};
                    FileDialog::ShowOpen(filters, [&](const fs::path &path) { OpenFile(r, viewport, path); });
                }
                const auto save_project_as = [&] {
                    static constexpr std::array filters{FileDialog::Filter{"MeshEditor project", "project"}};
                    FileDialog::ShowSave(filters, "scene.project", [&](const fs::path &picked) {
                        auto path = picked;
                        if (path.extension() != ProjectExt) path += ProjectExt; // The dialog doesn't force the filter's extension.
                        SaveProjectFile(r, viewport, path);
                    });
                };
                if (MenuItem("Save")) {
                    if (CurrentProjectPath.empty()) save_project_as();
                    else SaveProjectFile(r, viewport, CurrentProjectPath);
                }
                if (MenuItem("Save as...")) save_project_as();
                if (MenuItem("Clear history")) ClearHistory(r, viewport);
                if (BeginMenu("Restore")) {
                    const auto sessions = action::ListRestoreSessions(); // Most-recent first; the newest is the live session.
                    for (size_t i = 0; i < sessions.size(); ++i) {
                        const std::time_t t = sessions[i].UnixSeconds;
                        char date[32];
                        std::strftime(date, sizeof date, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
                        const auto label = i == 0 ? std::format("Current ({})", date) : std::string{date};
                        if (MenuItem(label.c_str())) OpenProjectDir(r, viewport, sessions[i].Path);
                    }
                    EndMenu();
                }
                const auto import_dialog = [](std::span<const FileDialog::Filter> filters) {
                    FileDialog::ShowOpen(filters, [](const fs::path &path) { action::Emit(action::io::Load{.Path = path}); });
                };
                if (BeginMenu("Import")) {
                    if (MenuItem("glTF 2.0 (.glb/.gltf)")) {
                        static constexpr std::array filters{FileDialog::Filter{"glTF scene", "gltf;glb"}};
                        import_dialog(filters);
                    }
                    if (MenuItem("Wavefront (.obj)")) {
                        static constexpr std::array filters{FileDialog::Filter{"Wavefront OBJ", "obj"}};
                        import_dialog(filters);
                    }
                    if (MenuItem("Stanford PLY (.ply)")) {
                        static constexpr std::array filters{FileDialog::Filter{"Stanford PLY", "ply"}};
                        import_dialog(filters);
                    }
                    if (MenuItem("RealImpact")) {
                        FileDialog::ShowPickFolder([](const fs::path &path) { action::Emit(action::io::LoadRealImpact{.Directory = path}); });
                    }
                    EndMenu();
                }
                const auto render_tree = [&](this auto &&self, const GltfSampleTree &n, const auto &passes) -> void {
                    const auto has_visible = [&](this auto &&rec, const GltfSampleTree &m) -> bool {
                        return any_of(m.Files, passes) || any_of(m.Children, [&](const auto &c) { return rec(c.second); });
                    };
                    struct Item {
                        const std::string *Name;
                        const GltfSampleTree *Child;
                        const GltfSample *File;
                    };
                    std::vector<Item> items;
                    items.reserve(n.Children.size() + n.Files.size());
                    for (const auto &[name, c] : n.Children) items.emplace_back(&name, &c, nullptr);
                    for (const auto &f : n.Files) items.emplace_back(&f.Label, nullptr, &f);
                    std::ranges::sort(items, {}, [](const Item &it) { return *it.Name; });
                    for (const auto &it : items) {
                        if (it.Child) {
                            if (!has_visible(*it.Child)) continue;
                            if (BeginMenu(it.Name->c_str())) {
                                self(*it.Child, passes);
                                EndMenu();
                            }
                        } else {
                            if (!passes(*it.File)) continue;
                            if (MenuItem(it.File->Label.c_str())) action::Emit(action::io::Load{.Path = it.File->Path});
                        }
                    }
                };
                const auto render_submenu = [&](const char *label, const GltfSampleTree &tree) {
                    if (tree.Children.empty() && tree.Files.empty()) return;
                    if (BeginMenu(label)) {
                        render_tree(tree, [](const GltfSample &) { return true; });
                        EndMenu();
                    }
                };
                static std::optional<GltfSampleTrees> trees;
                if (!trees && SampleTreesFuture.valid() && SampleTreesFuture.wait_for(std::chrono::seconds{0}) == std::future_status::ready) trees = SampleTreesFuture.get();
                if (trees) {
                    render_submenu("Examples", trees->Examples);
                    static std::set<std::string> sample_assets_filter;
                    if (!trees->SampleAssets.Files.empty() || !trees->SampleAssets.Children.empty()) {
                        if (BeginMenu("glTF Samples")) {
                            if (BeginMenu("Filter extensions")) {
                                PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
                                for (const auto &ext : trees->SampleAssetsExtensions) {
                                    const bool checked = sample_assets_filter.contains(ext);
                                    if (MenuItem(ext.c_str(), nullptr, checked)) {
                                        if (checked) sample_assets_filter.erase(ext);
                                        else sample_assets_filter.insert(ext);
                                    }
                                }
                                PopItemFlag();
                                EndMenu();
                            }
                            render_tree(trees->SampleAssets, [](const GltfSample &f) { return all_of(sample_assets_filter, [&](const auto &e) { return f.Extensions.contains(e); }); });
                            EndMenu();
                        }
                    }
                    render_submenu("glTF_Physics Samples", trees->Physics);
                }
                if (MenuItem("Save glTF", nullptr)) {
                    static constexpr std::array filters{FileDialog::Filter{"glTF scene", "gltf;glb"}};
                    FileDialog::ShowSave(filters, "scene.gltf", [](const fs::path &path) { action::Emit(action::io::SaveGltf{.Path = path}); });
                }
#ifdef DEBUG_BUILD
                if (MenuItem("[Debug] Roundtrip")) ValidateRoundTrip(r, viewport);
#endif
                EndMenu();
            }
            if (BeginMenu("Windows")) {
                MenuItem(windows.Debug.Name, nullptr, &windows.Debug.Visible);
                MenuItem(windows.ImGuiDemo.Name, nullptr, &windows.ImGuiDemo.Visible);
                MenuItem(windows.ImSpinnerDemo.Name, nullptr, &windows.ImSpinnerDemo.Visible);
                MenuItem(windows.ImPlotDemo.Name, nullptr, &windows.ImPlotDemo.Visible);
                MenuItem(windows.SceneControls.Name, nullptr, &windows.SceneControls.Visible);
                MenuItem(windows.Animation.Name, nullptr, &windows.Animation.Visible);
                MenuItem(windows.Viewport.Name, nullptr, &windows.Viewport.Visible);
                EndMenu();
            }
            EndMainMenuBar();
        }

        if (windows.Debug.Visible) {
            if (Begin(windows.Debug.Name, &windows.Debug.Visible)) {
                if (BeginTabBar("Debug")) {
                    if (BeginTabItem("ImGui")) {
                        Text("Dear ImGui %s (%d)", IMGUI_VERSION, IMGUI_VERSION_NUM);
                        Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
                        Text("%d vertices, %d indices (%d triangles)", io.MetricsRenderVertices, io.MetricsRenderIndices, io.MetricsRenderIndices / 3);
                        const auto &g = *GImGui;
                        Text("%d visible windows, %d current allocations", io.MetricsRenderWindows, g.DebugAllocInfo.TotalAllocCount - g.DebugAllocInfo.TotalFreeCount);
                        Separator();
                        Text("See [Windows->%s] for more details.", windows.ImGuiDemo.Name);
                        EndTabItem();
                    }
                    if (BeginTabItem("Vulkan")) {
                        SeparatorText("General");
                        const auto &props = vc->PhysicalDevice.getProperties();
                        Text("API version: %d.%d.%d", VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion));
                        Text("Driver version: %d.%d.%d", VK_VERSION_MAJOR(props.driverVersion), VK_VERSION_MINOR(props.driverVersion), VK_VERSION_PATCH(props.driverVersion));
                        Text("Vendor ID: 0x%04X", props.vendorID);
                        SeparatorText("Device");
                        Text("ID: 0x%04X", props.deviceID);
                        Text("Type: %s", vk::to_string(props.deviceType).c_str());
                        Text("Name: %s", props.deviceName.data());

                        SeparatorText("ImGui_ImplVulkanH_Window");
                        Text("Dimensions: %dx%d", wd.Width, wd.Height);
                        Text(
                            "Surface\n\tFormat: %s\n\tColor space: %s",
                            vk::to_string(vk::Format(wd.SurfaceFormat.format)).c_str(),
                            vk::to_string(vk::ColorSpaceKHR(wd.SurfaceFormat.colorSpace)).c_str()
                        );
                        Text("Present mode: %s", vk::to_string(vk::PresentModeKHR(wd.PresentMode)).c_str());
                        Text("Image count: %d", wd.ImageCount);
                        Text("Semaphore count: %d", wd.SemaphoreCount);
                        EndTabItem();
                    }
                    if (BeginTabItem("Engine")) {
                        SeparatorText("Buffer memory");
                        TextUnformatted(DebugBufferHeapUsage(r).c_str());
                        SeparatorText("Action");
                        Text("sizeof(Action): %zu bytes", action::ActionSize());
                        EndTabItem();
                    }
                    EndTabBar();
                }
            }
            End();
        }
        if (windows.ImGuiDemo.Visible) ImGui::ShowDemoWindow(&windows.ImGuiDemo.Visible);
        if (windows.ImSpinnerDemo.Visible) {
            if (Begin(windows.ImSpinnerDemo.Name, &windows.ImSpinnerDemo.Visible)) ImSpinner::demoSpinners();
            End();
        }
        if (windows.ImPlotDemo.Visible) ImPlot::ShowDemoWindow(&windows.ImPlotDemo.Visible);

        if (windows.SceneControls.Visible) {
            if (Begin(windows.SceneControls.Name, &windows.SceneControls.Visible)) RenderControls(r, viewport);
            End();
        }

        bool scrubbing = false; // Timeline frame marker held this frame; gates motion blur.
        if (windows.Animation.Visible) {
            PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
            if (Begin(windows.Animation.Name, &windows.Animation.Visible, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
                PushStyleVar(ImGuiStyleVar_FramePadding, {6, 4});
                Indent(6);
                Spacing();
                RenderClipPickers(r);
                Unindent(6);
                PopStyleVar();
                const auto scene_e = viewport;
                if (auto a = RenderAnimationTimeline(r.get<const TimelineRange>(scene_e), r.get<const TimelinePlayback>(scene_e), r.get<const AnimationTimelineView>(scene_e), r.ctx().get<const ViewportIcons>().Anim, scrubbing)) {
                    std::visit([](auto leaf) { action::Emit(leaf); }, std::move(*a));
                }
            }
            End();
            PopStyleVar();
        }
        r.ctx().get<FrameState>().Scrubbing = scrubbing;

        // Keep the viewport window open across the apply/derive/render below so the image draws back into it before End().
        uvec2 new_logical_extent{};
        bool viewport_open = false;
        if (windows.Viewport.Visible) {
            PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
            viewport_open = Begin(windows.Viewport.Name, &windows.Viewport.Visible);
            if (viewport_open) {
                Interact(r, viewport, r.ctx().get<FrameState>());
                auto &dl = *ImGui::GetWindowDrawList();
                dl.ChannelsSplit(2);
                dl.ChannelsSetCurrent(1);
                InteractOverlay(r, viewport, r.ctx().get<FrameState>());
                DrawModalJobsOverlay(r);
                const auto content_region = ImGui::GetContentRegionAvail();
                new_logical_extent = {uint32_t(std::max(content_region.x, 0.f)), uint32_t(std::max(content_region.y, 0.f))};
            }
        }
        const bool viewport_settled = new_logical_extent != uvec2{} && new_logical_extent == r.ctx().get<const ViewportExtent>().Value;
        // Remaining emits go after Interact so it wins the single-action buffer.
        driver.EmitFrameActions(r, viewport, viewport_settled, new_logical_extent);
        // Record viewport resizes so replay restores the same render extent.
        // A resize drag spans many frames: stage it and commit on mouse-up so the drag records a single SetExtent.
        // Staged after the frame's other emits so those win the single-action buffer.
        if (new_logical_extent != uvec2{} && r.ctx().get<const ViewportExtent>().Value != new_logical_extent) {
            action::EmitStaged(action::view::SetExtent{new_logical_extent});
            viewport_resizing = true;
        } else if (viewport_resizing && !IsMouseDown(ImGuiMouseButton_Left)) {
            action::Commit(); // Drag finished: flush the gesture's final staged SetExtent.
            viewport_resizing = false;
        }

        action::ApplyEmitted(r, viewport);
        ReportActionErrors(r);

        // Derive this frame's applied actions and submit the GPU render (nonblocking). WaitForRender() runs later, before RenderFrame() samples the image.
        SubmitViewport(r, viewport, GetFrameCount() > 1 ? vk::Fence{wd.Frames[wd.FrameIndex].Fence} : vk::Fence{});

        if (viewport_open) {
            // Draw the rendered image and overlays into the open viewport window.
            DisplayViewport(r, viewport);
            ImGui::GetWindowDrawList()->ChannelsMerge();
        }
        if (windows.Viewport.Visible) {
            End();
            PopStyleVar();
        }

        ImGui::Render();
        auto *draw_data = GetDrawData();
        if (const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f); !is_minimized) {
            WaitForRender(r); // ImGui samples final image
            if (driver.CaptureFrame(r, viewport, viewport_settled)) done = true;
            RenderFrame(*vc->Device, vc->Queue, wd, draw_data);
            PresentFrame(vc->Queue, wd);
        }
    }

    action::StopLog();
    vc->Device->waitIdle();

    r.ctx().erase<AudioDeviceResource>(); // Stops and uninitializes the output device.

    // Tear down the viewport and its ctx-resident GPU stores in order before clearing the registry,
    // so GpuBuffers (and its VMA allocator) outlives the MeshStore allocations that retire into it.
    DeinitViewportMedia(r); // App-only media (icons/modal audio/ImGui texture), while the device + GpuBuffers are alive.
    DeinitViewport(r, viewport);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    ImGui_ImplVulkanH_DestroyWindow(*vc->Instance, *vc->Device, &wd, nullptr);
    vkDestroySurfaceKHR(*vc->Instance, wd.Surface, nullptr);
    wd.Surface = {};
    imgui_descriptor_pool.reset();
    vc.reset();

    SDL_DestroyWindow(window);
    SDL_Quit();
}

// Seed the scene, run the fixed-step capture loop, and finish the session log.
void RunHeadlessScene(entt::registry &r, entt::entity viewport, const char *initial_file, bool empty, const CaptureRequest &capture) {
    auto driver = BeginCaptureSession(r, viewport, capture, initial_file, empty, /*fixed_step=*/true);
    // Emitted, not Performed: the resize must happen inside the first tick's SubmitViewport for that
    // frame to render the recreated images correctly.
    action::Emit(action::view::SetExtent{DefaultWindowSize});

    auto &frame_state = r.ctx().get<FrameState>();
    frame_state.DeltaTime = driver.RenderDt;
    int bench_frames = capture.BenchFrames;
    bool profile_cleared{false};
    bool done{false};
    while (!done) {
        if (driver.DurationElapsed(r, viewport)) break;
        const auto extent = r.ctx().get<const ViewportExtent>().Value;
        const bool settled = ViewportImageReady(r);
        // Scene-load work (mesh and texture upload) dwarfs a frame: keep it out of the profile.
        if (settled && !profile_cleared) {
            r.ctx().get<Profile>().ClearStats();
            profile_cleared = true;
        }
        {
            const CpuScope scope{r.ctx().get<Profile>(), "Frame"};
            driver.EmitFrameActions(r, viewport, settled, extent);
            action::ApplyEmitted(r, viewport);
            ReportActionErrors(r);
            SubmitViewport(r, viewport);
            WaitForRender(r);
        }
        if (bench_frames > 0) {
            // Benchmark: force a render every settled tick (direct request write) and exit after the requested count.
            if (settled && --bench_frames == 0) {
                if (driver.ScreenshotMode()) {
                    if (auto saved = SaveScreenshot(r, driver.ScreenshotPath); saved) std::println("Saved screenshot: {}", saved->string());
                    else std::println(stderr, "Screenshot: {}", saved.error());
                }
                done = true;
            }
            r.ctx().get<PendingRenderRequest>().Value = RenderRequest::ReRecord;
        } else {
            if (driver.CaptureFrame(r, viewport, settled)) done = true;
            // Headless has no window to close: without anything to capture or play, one settled frame is the whole run.
            if (!driver.Presenting() && settled) done = true;
        }
        driver.ElapsedPlayTime += frame_state.DeltaTime;
    }
    action::StopLog();
}

// Run without a window: no SDL video, ImGui, audio, or file dialogs. The viewport renders offscreen
// on a fixed-step, GPU-paced clock, and capture reads it back. Initializes the engine, runs `scenes`,
// and tears down.
void RunHeadlessEngine(bool quiet, auto &&scenes) {
    LogEnabled = !quiet;

    if (!SDL_Init(0)) throw std::runtime_error(std::format("SDL_Init error: {}", SDL_GetError())); // Base path only, no subsystems.
    InitPaths();

    auto vc = std::make_unique<VulkanContext>(std::vector<const char *>{}, /*with_swapchain=*/false);
    entt::registry r;
    const auto viewport = InitEngine(r, vc->Resources());
    SetupScene(r, viewport);
    r.ctx().get<FrameState>().DisplayFramebufferScale = {2, 2}; // Match the app's retina rendering (pixel density and DPI-scaled GPU state like edge-line width).
    ProcessComponentEvents(r, viewport); // Prime derived state before the first frame reads it.

    scenes(r, viewport);

    vc->Device->waitIdle();
    DeinitViewport(r, viewport);
    vc.reset();
    SDL_Quit();
}

// Headless single-scene run. Exits after one rendered frame when there is nothing to capture or play.
void RunHeadless(const char *initial_file, bool quiet, bool empty, const CaptureRequest &capture) {
    RunHeadlessEngine(quiet, [&](entt::registry &r, entt::entity viewport) {
        RunHeadlessScene(r, viewport, initial_file, empty, capture);
    });
}

// A corpus render job spooled by `script/Render`: one `.job` file per scene, holding
// "<output basename>\t<scene arg>" (scene arg: a file path, "--empty", or empty for the default scene).
struct RenderJob {
    fs::path OutBasename;
    std::string SceneArg;
};

// Claim the next pending job by renaming it to `.claimed`. The rename is atomic, so parallel
// workers pulling from one spool never render the same scene twice.
std::optional<RenderJob> ClaimRenderJob(const fs::path &spool) {
    std::vector<fs::path> pending;
    std::error_code ec;
    for (const auto &entry : fs::directory_iterator{spool, ec}) {
        if (entry.path().extension() == ".job") pending.emplace_back(entry.path());
    }
    std::ranges::sort(pending);
    for (const auto &path : pending) {
        auto claimed = path;
        claimed += ".claimed";
        std::error_code rename_ec;
        fs::rename(path, claimed, rename_ec);
        if (rename_ec) continue; // Another worker claimed it first.
        std::ifstream in{claimed};
        std::string line;
        if (!std::getline(in, line)) continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        return RenderJob{line.substr(0, tab), line.substr(tab + 1)};
    }
    return std::nullopt;
}

// Render every job in the spool with one engine, clearing the scene between jobs.
// Each scene's console output goes to its `.log`, and stdout gets one line per finished scene.
void RunHeadlessQueue(const fs::path &spool, bool quiet) {
    RunHeadlessEngine(quiet, [&](entt::registry &r, entt::entity viewport) {
        const int launcher_out = ::dup(STDOUT_FILENO), launcher_err = ::dup(STDERR_FILENO);
        while (const auto job = ClaimRenderJob(spool)) {
            const auto out = job->OutBasename.string();
            std::fflush(stdout);
            std::fflush(stderr);
            if (const int log_fd = ::open((out + ".log").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644); log_fd >= 0) {
                ::dup2(log_fd, STDOUT_FILENO);
                ::dup2(log_fd, STDERR_FILENO);
                ::close(log_fd);
            }
            const bool empty = job->SceneArg == "--empty";
            const char *initial_file = !empty && !job->SceneArg.empty() ? job->SceneArg.c_str() : nullptr;
            RunHeadlessScene(r, viewport, initial_file, empty, CaptureRequest{.RenderBasename = job->OutBasename});
            // Reset for the next job, finalizing any in-progress recording.
            QuiesceScene(r, viewport);
            ClearScene(r, viewport);
            ProcessComponentEvents(r, viewport); // Settle the reset so the next scene loads from the same baseline as a fresh engine.
            std::fflush(stdout);
            std::fflush(stderr);
            ::dup2(launcher_out, STDOUT_FILENO);
            ::dup2(launcher_err, STDERR_FILENO);
            if (fs::exists(out + ".webp") || fs::exists(out + ".mp4")) std::println("ok   {}", out);
            else std::println("SKIP {} (no output; load failed or unsupported encoding)", out);
            std::fflush(stdout); // Stdout is typically a block-buffered pipe, so make the line visible now.
        }
        ::close(launcher_out);
        ::close(launcher_err);
    });
}
} // namespace

int main(int argc, char **argv) {
    // VideoRecorder pipes frames to ffmpeg via popen; ignore SIGPIPE so writes return EPIPE instead of killing us.
    std::signal(SIGPIPE, SIG_IGN);

    std::set_terminate([]() {
        try {
            if (auto eptr = std::current_exception()) std::rethrow_exception(eptr);
        } catch (const std::exception &e) {
            std::println(stderr, "{}", e.what());
        } catch (...) {
            std::println(stderr, "Unhandled unknown exception");
        }
        std::abort();
    });

    const std::span args{argv + 1, argv + argc};
    const auto looks_numeric = [](const char *a) {
        const std::string_view s{a};
        return !s.empty() && std::isdigit(uint8_t(s[0]));
    };

    const char *initial_file = nullptr;
    bool empty = false, headless = false;
    fs::path render_queue;
    CaptureRequest capture;
#ifdef QUIET
    bool quiet = true;
#else
    bool quiet = false;
#endif
    for (auto it = args.begin(); it != args.end(); ++it) {
        const std::string_view a{*it};
        if (a == "--quiet" || a == "-q") quiet = true;
        else if (a == "--play") {
            capture.Play = true;
            if (std::next(it) != args.end() && looks_numeric(*std::next(it))) capture.PlayDuration = std::atof(*++it);
        } else if (a == "--record" && std::next(it) != args.end()) capture.RecordPath = *++it;
        else if (a == "--screenshot" && std::next(it) != args.end()) capture.ScreenshotPath = *++it;
        else if (a == "--render" && std::next(it) != args.end()) capture.RenderBasename = *++it;
        else if (a == "--render-queue" && std::next(it) != args.end()) render_queue = *++it;
        else if (a == "--empty") empty = true;
        else if (a == "--headless") headless = true;
        else if (a == "--fps" && std::next(it) != args.end()) capture.Fps = std::atoi(*++it);
        else if (a == "--motion-blur" && std::next(it) != args.end()) capture.MotionBlurSteps = uint8_t(std::max(1, std::atoi(*++it)));
        else if (a == "--frames" && std::next(it) != args.end()) capture.BenchFrames = std::atoi(*++it);
        else if (a == "--profile") Profile::Enabled = true;
        else if (!a.starts_with('-') && !initial_file) initial_file = *it;
    }
    if (capture.Fps <= 0) capture.Fps = 60;
    // Render mode derives its own output paths from the basename.
    if (!capture.RenderBasename.empty() && (!capture.RecordPath.empty() || !capture.ScreenshotPath.empty())) {
        std::println(stderr, "--render cannot be combined with --record or --screenshot");
        return 1;
    }
    // Queue mode is headless and derives everything from each job.
    if (!render_queue.empty() && (initial_file || !capture.RenderBasename.empty() || !capture.RecordPath.empty() || !capture.ScreenshotPath.empty())) {
        std::println(stderr, "--render-queue cannot be combined with a scene file or capture flags");
        return 1;
    }

    if (!render_queue.empty()) RunHeadlessQueue(render_queue, quiet);
    else if (headless) RunHeadless(initial_file, quiet, empty, capture);
    else run(initial_file, quiet, empty, capture);
    return 0;
}
