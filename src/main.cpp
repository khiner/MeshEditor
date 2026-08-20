#include "Compress.h"
#include "File.h"
#include "FileDialog.h"
#include "LogEnabled.h"
#include "MacPlatform.h"
#include "Paths.h"
#include "ProcessEvents.h"
#include "TransformMath.h"
#include "VideoRecorder.h"
#include "Window.h"
#include "action/ActionApply.h"
#include "action/ActionIndex.h"
#include "action/Build.h"
#include "action/Emit.h"
#include "action/Errors.h"
#include "action/Io.h"
#include "action/Log.h"
#include "action/Selection.h"
#include "action/View.h"
#include "animation/AnimationData.h"
#include "animation/TimelineUi.h"
#include "armature/ArmatureComponents.h"
#include "audio/AudioDevice.h"
#include "audio/AudioSystem.h"
#include "audio/AudioTypes.h"
#include "audio/ModalModelFile.h"
#include "audio/ModalModes.h"
#include "gizmo/TransformGizmoTypes.h"
#include "gltf/GltfScene.h"
#include "image/ImageEncode.h"
#include "mesh/MeshComponents.h"
#include "metal/MetalContext.h"
#include "metal/RenderTarget.h"
#include "metal/Shader.h"
#include "object/ObjectOps.h"
#include "physics/PhysicsTypes.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
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

#include "imgui_impl_metal.h"
#include "imgui_internal.h"
#include "implot.h"
#include "imspinner_demo.h"
#include <Foundation/NSAutoreleasePool.hpp>
#include <entt/entity/registry.hpp>

#include <chrono>
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
using SteadyClock = std::chrono::steady_clock;

// #define IMGUI_UNLIMITED_FRAME_RATE

namespace {
// Skip rather than block when every drawable is in flight.
MTL::CommandBuffer *RenderAndPresentFrame(const mtl::Context &ctx, CA::MetalLayer *layer, ImDrawData *draw_data) {
    const profile::CpuScope scope{"ImGuiRenderSubmit"};
    auto *drawable = layer->nextDrawable();
    if (!drawable) return nullptr;

    const std::array colors{mtl::ClearColor(drawable->texture(), {0.45, 0.55, 0.60, 1.0})};
    auto *const pass = mtl::MakePassDescriptor(colors);
    auto *command_buffer = ctx.Queue->commandBuffer();
    ImGui_ImplMetal_NewFrame(pass);
    auto *encoder = command_buffer->renderCommandEncoder(pass);
    ImGui_ImplMetal_RenderDrawData(draw_data, command_buffer, encoder);
    encoder->endEncoding();
    {
        const profile::CpuScope present_scope{"Present"};
        command_buffer->presentDrawable(drawable);
    }
    command_buffer->commit();
    return command_buffer;
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
    GltfSampleTree Examples, Benchmarks, SampleAssets, Physics, PhysicalAudio;
    std::set<std::string> SampleAssetsExtensions;
};

GltfSampleTrees BuildSampleTrees() {
    GltfSampleTrees t;
    t.Examples = BuildGltfSampleTree(Paths::Res() / "examples");
    t.Benchmarks = BuildGltfSampleTree(Paths::Res() / "benchmarks");
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
#ifdef GLTF_PHYSICAL_AUDIO_DIR
    t.PhysicalAudio = BuildGltfSampleTree(GLTF_PHYSICAL_AUDIO_DIR);
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
    WaitForRender(r);
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
    WaitForRender(r);
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

// Fit the scene into the middle half of the view without changing camera orientation.
// Returns false while instance bounds are still pending from the GPU.
bool FrameScene(entt::registry &r, entt::entity viewport, float aspect_ratio) {
    const auto &cam = r.get<const ViewCamera>(viewport);
    const auto *persp = std::get_if<Perspective>(&cam.Data); // The launch view camera is always perspective.
    if (!persp) return true;

    // Keep the current orientation; measure each vertex against this camera's basis.
    const vec3 right = cam.Orientation * vec3{1, 0, 0}, up = cam.Orientation * vec3{0, 1, 0}, away = cam.Forward();
    // Half the real frustum tangents, so a vertex reaching the (narrower) edge only fills the middle half of the true view.
    const float ty = 0.5f * std::tan(persp->FieldOfViewRad * 0.5f), tx = aspect_ratio * ty;
    constexpr float lowest = std::numeric_limits<float>::lowest();
    float top = lowest, bottom = lowest, rgt = lowest, lft = lowest;
    const auto &buffers = r.ctx().get<const GpuBuffers>();
    AABB scene;
    bool any_bounded_instance = false;
    for (const auto [e, ri, wt] : r.view<const RenderInstance, const WorldTransform>().each()) {
        if (ri.BufferIndex == UINT32_MAX) continue;
        any_bounded_instance = true;
        // Extras (gizmo/wireframe) instances hold an empty AABB and fail the validity check.
        const auto &local = buffers.Instances.GetBounds(ri.BufferIndex);
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
    if (glm::any(glm::greaterThan(scene.Min, scene.Max))) return !any_bounded_instance;

    const auto center = (scene.Min + scene.Max) * 0.5f;
    const float ca = glm::dot(center, right), cb = glm::dot(center, up), cf = glm::dot(center, away);
    const float distance = std::max({top - cb / ty, bottom + cb / ty, rgt - ca / tx, lft + ca / tx}) - cf;
    if (distance <= 0.f) return true; // Framing cannot help a scene the camera already sits inside.

    // Clip planes bracket the scene depth so nothing is z-clipped.
    const float plane_reach = 6 * glm::length(scene.Max - scene.Min);
    auto fit = *persp;
    fit.FarClip = distance + plane_reach;
    fit.NearClip = std::max(distance - plane_reach, *fit.FarClip / 10000.f);

    r.replace<ViewCamera>(viewport, ViewCamera{center + distance * away, center, Camera{fit}});
    return true;
}

// The default window size, doubling as the headless viewport extent.
constexpr uvec2 DefaultWindowSize{1280, 800};

// Capture options from the CLI. `--render` is a preset for the full scene corpus; `--screenshot`/`--record` target one output.
struct CaptureRequest {
    enum class BenchmarkAction { Steady, Orbit, Transform, Visibility };

    bool Play{false};
    float PlayDuration{0}; // 0 = run until playback completes one loop.
    int Fps{60};
    bool RecordAudio{false}; // Mux the master output into the recording. Off so the render corpus stays video only.
    fs::path RecordPath{}, ScreenshotPath{};
    fs::path RenderBasename{}; // Output basename, no extension.
    std::optional<uint8_t> MotionBlurSteps{}; // Disengaged = leave the viewport's own setting alone.
    float TimelineEnd{0}; // Seconds. Positive: set the timeline's end frame, so a long play runs without looping.
    int BenchFrames{0}; // Headless: re-render every tick and exit after this many frames.
    BenchmarkAction BenchAction{BenchmarkAction::Steady};
    uint32_t BenchActionCount{64};
    std::string CameraName{};
    std::optional<ViewportShadingMode> Shading{}; // Disengaged = leave the viewport's own setting alone.
    bool Overlays{false}; // Keep overlays on through a capture, which presentation otherwise turns off.
    bool Edit{false}; // Select mesh objects and enter vertex edit mode.
};

struct BenchmarkDriver {
    CaptureRequest::BenchmarkAction Action;
    std::vector<std::pair<entt::entity, Transform>> Transforms;
    uint32_t Frame{};

    BenchmarkDriver(entt::registry &r, const CaptureRequest &capture) : Action(capture.BenchAction) {
        if (Action != CaptureRequest::BenchmarkAction::Transform && Action != CaptureRequest::BenchmarkAction::Visibility) return;
        std::vector<entt::entity> entities;
        for (const auto [entity, kind] : r.view<const ObjectKind>(entt::exclude<SubElementOf>).each()) {
            if (kind.Value == ObjectType::Mesh && r.all_of<Instance, Transform, RenderInstance>(entity)) entities.emplace_back(entity);
        }
        std::ranges::sort(entities);
        entities.resize(std::min<size_t>(entities.size(), capture.BenchActionCount));
        Transforms.reserve(entities.size());
        for (const auto entity : entities) Transforms.emplace_back(entity, r.get<const Transform>(entity));
    }

    void Apply(entt::registry &r) {
        switch (Action) {
            case CaptureRequest::BenchmarkAction::Steady: break;
            case CaptureRequest::BenchmarkAction::Orbit:
                action::Emit(action::view::OrbitViewCamera{{0.01f, 0.f}});
                break;
            case CaptureRequest::BenchmarkAction::Transform: {
                const float offset = Frame % 2 == 0 ? 0.05f : 0.f;
                for (const auto &[entity, base] : Transforms) {
                    r.patch<Transform>(entity, [&](auto &transform) {
                        transform = base;
                        transform.P.y += offset;
                    });
                }
                break;
            }
            case CaptureRequest::BenchmarkAction::Visibility:
                for (const auto &[entity, _] : Transforms) {
                    if (Frame % 2 == 0) Hide(r, entity);
                    else Show(r, entity);
                }
                break;
        }
        ++Frame;
    }
};

bool SelectSceneCamera(entt::registry &r, entt::entity viewport, std::string_view name) {
    if (name.empty()) return true;
    for (const auto [entity, _, camera_name] : r.view<const Camera, const CameraName>().each()) {
        if (camera_name.Value != name) continue;
        SetLookThrough(r, viewport, entity);
        ProcessComponentEvents(r, viewport);
        return true;
    }
    std::println(stderr, "No scene camera named '{}'.", name);
    return false;
}

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
    const fs::path path = initial_file ? initial_file : "";
    bool loaded{true};
    if (path.extension() == ProjectExt) OpenProjectFile(r, viewport, path);
    else if (path.extension() == ActionsExt) ReplayLogIntoNewSession(r, viewport, path);
    else {
        if (capture.RenderBasename.empty()) {
            Paths::SetProject(action::ReserveRestoreSession());
            action::StartLog(Paths::Project() / SessionLogName);
        } else {
            Paths::SetProject(capture.RenderBasename.parent_path());
            action::StartLog(fs::path{capture.RenderBasename.string() + ".actions"});
        }
        if (initial_file) {
            Perform(r, viewport, action::io::Load{.Path = initial_file});
            loaded = !ReportActionErrors(r);
        } else if (!empty) Perform(r, viewport, action::io::LoadDefaultScene{});
    }
    return loaded && SelectSceneCamera(r, viewport, capture.CameraName);
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
        // A render records the scene's own audio, so a scene with sound objects gets a track and every other stays video only.
        RecordAudio = capture.RecordAudio || (RenderMode() && !r.view<const ModalModes>().empty());
    }

    bool RenderMode() const { return !RenderBasename.empty(); }
    bool RecordingMode() const { return !RecordPath.empty(); }
    bool ScreenshotMode() const { return !ScreenshotPath.empty(); }
    // A wav recording consumes no images, so its run can skip GPU frames once recording is underway.
    bool AudioOnly() const { return RecordPath.extension() == ".wav" && !ScreenshotMode(); }
    bool Presenting() const { return Play || ScreenshotMode() || RecordingMode(); }
    bool Framed(bool settled) const { return settled && (ViewFramed || !Presenting()); }

    bool DurationElapsed(const entt::registry &r, entt::entity viewport) const {
        if (PlayDuration <= 0) return false;
        const float elapsed = RecordingMode() ? float(CapturedFrameCount(r, viewport)) / float(RecordFps) : ElapsedPlayTime;
        return elapsed >= PlayDuration;
    }

    // Emit this frame's capture-driven actions. Call before ApplyEmitted, with `settled` true once
    // the viewport is at its final extent with the image built.
    void EmitFrameActions(entt::registry &r, entt::entity viewport, bool settled, uvec2 extent) {
        if (!ViewFramed && Presenting() && extent != uvec2{}) {
            // The launch camera waits for GPU-produced bounds; a scene camera frames itself.
            const bool framed = !r.view<const Camera>().empty() ||
                FrameScene(r, viewport, float(extent.x) / float(extent.y));
            ViewFramed = settled && framed;
        }
        // Fixed-step recording waits one more tick, until recording has begun, so the start frame is captured.
        const bool ready = RenderMode() || (FixedStep && RecordingMode()) ? IsRecording(r, viewport) : (Play || RecordingMode());
        if (!PlaybackStarted && Framed(settled) && ready) {
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
        // The recording starts once.
        // An encoder that fails under it takes the recording with it, and starting another would fail the same way, so the run ends instead.
        if (RecordingMode() && settled) {
            if (!RecordingStarted) {
                RecordingStarted = true;
                StartRecording(r, viewport, RecordPath, RecordFps, RecordAudio);
                if (IsRecording(r, viewport)) NextCapture = SteadyClock::now();
            }
            if (!IsRecording(r, viewport)) done = true;
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
            } else if (SteadyClock::now() >= NextCapture) {
                CaptureRecordFrame(r, viewport);
                NextCapture += std::chrono::nanoseconds{1'000'000'000 / RecordFps};
            }
        } else if (FixedStep && !RecordingMode() && PlaybackStarted && PlayDuration <= 0 && loop_end) {
            // A duration-less fixed-step play run (headless --play) ends after one timeline loop.
            done = true;
        }
        return done;
    }

    bool Play;
    bool SeedFailed{false}; // The initial scene failed to load, so a headless run has nothing to render.
    float PlayDuration;
    bool FixedStep;
    fs::path RecordPath, ScreenshotPath, RenderBasename;
    float RenderDt; // Fixed-step seconds per tick (one timeline frame).
    int RecordFps;
    bool RecordAudio;
    SteadyClock::time_point NextCapture; // Wall-clock recording: next capture time, initialized when recording starts.
    float ElapsedPlayTime{0}; // Caller-accumulated sim seconds, for the play-duration cap.
    uint32_t NextRenderClip{1}; // Next clip to capture once the current loop finishes.
    uint32_t NextRenderVariant{0}; // Next material variant to capture once the default image saves.
    bool PlaybackStarted{false}, ScreenshotSaved{false}, ViewFramed{false}, RecordingStarted{false};
};

// Seed the scene and its session log, then enter presentation mode so the first rendered frame matches the capture.
CaptureDriver BeginCaptureSession(entt::registry &r, entt::entity viewport, const CaptureRequest &capture, const char *initial_file, bool empty, bool fixed_step) {
    const bool seeded = SeedScene(r, viewport, capture, initial_file, empty);
    const bool play = seeded && capture.Play;
    // After the load, whose end frame comes from the scene's own animation durations.
    if (capture.TimelineEnd > 0) {
        const float fps = r.get<const TimelineRange>(viewport).Fps;
        Perform(r, viewport, action::timeline::SetEndFrame{int(std::ceil(capture.TimelineEnd * fps))});
    }
    CaptureDriver driver{r, viewport, capture, play, fixed_step};
    driver.SeedFailed = !seeded;
    // A benchmark run keeps the editor view, so frames and screenshots cover what the editor draws.
    if (driver.Presenting() && capture.BenchFrames == 0) Perform(r, viewport, action::timeline::EnterPresentation{});
    // Presentation disables overlays unless --overlays restores them.
    if (capture.Overlays) Perform(r, viewport, action::UpdateOf<&ViewportDisplay::ShowOverlays>(viewport, true));
    r.ctx().get<FrameState>().FixedFrameStep = driver.FixedStep;
    // Force motion blur on for the whole recording run, leaving still-screenshot renders and audio-only recordings, whose frames nothing reads, alone.
    r.ctx().get<FrameState>().Capturing = driver.RecordingMode() && !driver.AudioOnly();
    if (capture.MotionBlurSteps) {
        Perform(r, viewport, action::UpdateOf<&ViewportDisplay::MotionBlur>(viewport, std::optional{MotionBlur{.Steps = *capture.MotionBlurSteps}}));
    }
    if (capture.Shading) {
        Perform(r, viewport, action::view::SetViewportShading{*capture.Shading});
    }
    return driver;
}

void run(const char *initial_file, bool quiet, bool empty, const CaptureRequest &capture) {
    LogEnabled = !quiet;

    MacPlatform::InitPaths();

    MacPlatform::Window window;

    entt::registry r;
    const auto &ctx = r.ctx().emplace<mtl::Context>();

    auto *const layer = window.Layer();
    layer->setDevice(ctx.Device.get());
    layer->setPixelFormat(mtl::Format::Color);
    // Render mode is GPU-paced: content is fixed-step per tick, so present pacing only affects
    // wall time. Benchmark frames measure throughput, which vsync pacing would hide.
    layer->setDisplaySyncEnabled(capture.RenderBasename.empty() && capture.BenchFrames == 0);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    auto &io = GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows
    io.IniFilename = nullptr; // Disable ImGui's .ini file saving
    io.ConfigDebugIgnoreFocusLoss = true; // Keep input state across Cmd+Tab so in-flight gizmo drags survive focus loss.
    io.ConfigDragClickToInputText = true; // A click-release without dragging turns a Drag field into a text input.

    StyleColorsDark();

    window.InitImGui();
    ImGui_ImplMetal_Init(ctx.Device.get());

    InitFonts();

    const auto viewport = InitEngine(r);
    InitViewportMedia(r);
    SetupScene(r, viewport); // Before the first frame reads viewport state.
    // Capture the DPI scale (only set during NewFrame) before priming DPI-scaled GPU state like edge-line width.
    window.NewImGuiFrame();
    r.ctx().get<FrameState>().DisplayFramebufferScale = {io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y};
    ProcessComponentEvents(r, viewport); // Prime derived state before the first frame reads it.

    auto &audio_device = r.ctx().emplace<AudioDeviceResource>(r, viewport);
    ReconcileAudioDevice(audio_device, r.get<const AudioOutputConfig>(viewport), r.get<const AudioOutputMix>(viewport));

    SampleTreesFuture = std::async(std::launch::async, BuildSampleTrees);

    auto driver = BeginCaptureSession(r, viewport, capture, initial_file, empty, /*fixed_step=*/false);

    bool viewport_resizing{false}; // True while a resize drag is staged but not yet committed.
    bool done{false};
    MTL::CommandBuffer *last_frame{nullptr}; // Resize waits for resources sampled by the last submitted UI frame.
    WindowsState windows;
    int bench_ticks{0};
    while (!done) {
        // Scene-load work (mesh and texture upload) dwarfs a frame: keep it out of the profile.
        if (bench_ticks == 1) profile::ClearStats();
        const profile::CpuScope frame_scope{"Frame"};
        auto events = window.PollEvents();
        r.ctx().get<FrameState>().PreciseWheelDelta += vec2{events.ScrollX, events.ScrollY};
        for (const auto &path : events.DroppedFiles) OpenFile(r, viewport, path);
        done = events.Quit;
        if (driver.DurationElapsed(r, viewport)) done = true;

        window.NewImGuiFrame();
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
                    FileDialog::ShowOpen("project;state;actions", [&](const fs::path &path) { OpenFile(r, viewport, path); });
                }
                const auto save_project_as = [&] {
                    FileDialog::ShowSave("project", "scene.project", [&](const fs::path &picked) {
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
                const auto import_dialog = [](const char *extensions) {
                    FileDialog::ShowOpen(extensions, [](const fs::path &path) { action::Emit(action::io::Load{.Path = path}); });
                };
                if (BeginMenu("Import")) {
                    if (MenuItem("glTF 2.0 (.glb/.gltf)")) {
                        import_dialog("gltf;glb");
                    }
                    if (MenuItem("Wavefront (.obj)")) {
                        import_dialog("obj");
                    }
                    if (MenuItem("Stanford PLY (.ply)")) {
                        import_dialog("ply");
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
                    render_submenu("Benchmarks", trees->Benchmarks);
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
                    render_submenu("glTF_PhysicalAudio Samples", trees->PhysicalAudio);
                }
                if (MenuItem("Save glTF", nullptr)) {
                    FileDialog::ShowSave("gltf;glb", "scene.gltf", [](const fs::path &path) { action::Emit(action::io::SaveGltf{.Path = path}); });
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
                    if (BeginTabItem("Metal")) {
                        SeparatorText("Device");
                        Text("Name: %s", ctx.Device->name()->utf8String());
                        Text("Unified memory: %s", ctx.Device->hasUnifiedMemory() ? "yes" : "no");
                        Text("Argument buffer tier: %ld", long(ctx.Device->argumentBuffersSupport()) + 1);
                        Text("BC texture compression: %s", ctx.Device->supportsBCTextureCompression() ? "yes" : "no");
                        Text("Max buffer length: %llu MB", (unsigned long long)(ctx.Device->maxBufferLength() / (1024 * 1024)));

                        SeparatorText("Window surface");
                        const auto drawable_size = layer->drawableSize();
                        Text("Drawable: %.0fx%.0f", drawable_size.width, drawable_size.height);
                        Text("Display sync: %s", layer->displaySyncEnabled() ? "on" : "off");
                        EndTabItem();
                    }
                    if (BeginTabItem("Engine")) {
                        SeparatorText("Buffer memory");
                        TextUnformatted(DebugBufferHeapUsage(r).c_str());
                        SeparatorText("Action");
                        Text("sizeof(Action): %zu bytes", action::ActionSize());
                        EndTabItem();
                    }
                    if (BeginTabItem("Audio")) {
                        DrawAudioDebug(r);
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
        if (capture.BenchFrames > 0 && viewport_settled) {
            if (capture.BenchAction == CaptureRequest::BenchmarkAction::Orbit) action::Emit(action::view::OrbitViewCamera{{0.01f, 0.f}});
            if (++bench_ticks >= capture.BenchFrames) done = true;
        }
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
        SubmitViewport(r, viewport, GetFrameCount() > 1 ? last_frame : nullptr);

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
            if (driver.CaptureFrame(r, viewport, driver.Framed(viewport_settled))) done = true;
            if (auto *frame = RenderAndPresentFrame(ctx, layer, draw_data)) last_frame = frame;
        }
    }

    action::StopLog();
    if (last_frame) last_frame->waitUntilCompleted();

    r.ctx().erase<AudioDeviceResource>(); // Stops and uninitializes the output device.

    // GpuBuffers must outlive MeshStore allocations retired during teardown.
    DeinitViewportMedia(r); // App-only media (icons/modal audio/ImGui texture), while the device + GpuBuffers are alive.
    DeinitViewport(r, viewport);

    ImGui_ImplMetal_Shutdown();
    window.ShutdownImGui();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
}

// Seed the scene, run the fixed-step capture loop, and finish the session log.
bool RunHeadlessScene(entt::registry &r, entt::entity viewport, const char *initial_file, bool empty, const CaptureRequest &capture) {
    // Headless has no output device, so the audio system is created only for runs that may capture audio, keeping other renders free of the modal render pool's threads.
    // A render decides that from the scene it loads, and the load fills the bank, so it is created first.
    if (capture.RecordAudio || !capture.RenderBasename.empty()) InitAudioSystem(r);
    auto driver = BeginCaptureSession(r, viewport, capture, initial_file, empty, /*fixed_step=*/true);
    // A scene that failed to load has nothing to render, so the run ends here with the failure already on stderr rather than recording silence and reporting a clean exit.
    if (driver.SeedFailed) {
        action::StopLog();
        return false;
    }
    // Emitted, not Performed: the resize must happen inside the first tick's SubmitViewport for that
    // frame to render the recreated images correctly.
    action::Emit(action::view::SetExtent{DefaultWindowSize});
    if (capture.Edit) {
        std::vector<entt::entity> meshes;
        for (const auto [entity, kind, _] : r.view<const ObjectKind, const Instance>().each()) {
            if (kind.Value == ObjectType::Mesh) meshes.emplace_back(entity);
        }
        if (!meshes.empty()) {
            Perform(r, viewport, action::selection::ApplyTreeSelection{
                                     .ToSelect = meshes,
                                     .ToDeselect = {},
                                     .NavToActive = meshes.front(),
                                     .Clear = action::selection::ApplyTreeSelection::ClearKind::All,
                                 });
            Perform(r, viewport, action::view::SetInteractionMode{InteractionMode::Edit});
            Perform(r, viewport, action::view::SetEditMode{Element::Vertex});
        }
    }

    auto &frame_state = r.ctx().get<FrameState>();
    frame_state.DeltaTime = driver.RenderDt;
    int bench_frames = capture.BenchFrames;
    BenchmarkDriver benchmark{r, capture};
    bool profile_cleared{false};
    bool submitted{false};
    bool done{false};
    while (!done) {
        if (driver.DurationElapsed(r, viewport)) break;
        const auto extent = r.ctx().get<const ViewportExtent>().Value;
        // Queue workers keep render resources across scenes, so an image can be ready before this scene has rendered.
        // Settling also requires this scene's first submit, which fills the instance bounds FrameScene reads.
        const bool settled = submitted && ViewportImageReady(r);
        // Scene-load work (mesh and texture upload) dwarfs a frame: keep it out of the profile.
        if (settled && !profile_cleared) {
            profile::ClearStats();
            profile_cleared = true;
        }
        {
            const profile::CpuScope scope{"Frame"};
            if (bench_frames > 0 && settled) benchmark.Apply(r);
            driver.EmitFrameActions(r, viewport, settled, extent);
            action::ApplyEmitted(r, viewport);
            ReportActionErrors(r);
            // An audio-only recording consumes no images, so once it is underway the tick drops its render request and the sim still steps through SubmitViewport's event processing while the offline audio renders in CaptureRecordFrame with no GPU frame behind it.
            // The first frames still render, since settling and the scene framing read the built image.
            if (driver.AudioOnly() && driver.RecordingStarted) r.ctx().get<PendingRenderRequest>().Value = RenderRequest::None;
            SubmitViewport(r, viewport);
            WaitForRender(r);
            submitted = true;
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
            auto &pending = r.ctx().get<PendingRenderRequest>().Value;
            pending = std::max(pending, RenderRequest::Submit);
        } else {
            if (driver.CaptureFrame(r, viewport, driver.Framed(settled))) done = true;
            // Headless has no window to close: without anything to capture or play, one settled frame is the whole run.
            if (!driver.Presenting() && settled) done = true;
        }
        driver.ElapsedPlayTime += frame_state.DeltaTime;
    }
    action::StopLog();
    return true;
}

// Run without a window, ImGui, audio, or file dialogs. The viewport renders offscreen
// on a fixed-step, GPU-paced clock, and capture reads it back. Initializes the engine, runs `scenes`,
// and tears down.
void RunHeadlessEngine(bool quiet, auto &&scenes) {
    LogEnabled = !quiet;

    MacPlatform::InitPaths();

    entt::registry r;
    r.ctx().emplace<mtl::Context>();
    const auto viewport = InitEngine(r);
    SetupScene(r, viewport);
    r.ctx().get<FrameState>().DisplayFramebufferScale = {2, 2}; // Match the app's retina rendering (pixel density and DPI-scaled GPU state like edge-line width).
    ProcessComponentEvents(r, viewport); // Prime derived state before the first frame reads it.

    scenes(r, viewport);

    WaitForRender(r);
    DeinitViewport(r, viewport);
}

// Headless single-scene run.
// Exits after one rendered frame when there is nothing to capture or play, and reports failure when the scene itself failed to load.
bool RunHeadless(const char *initial_file, bool quiet, bool empty, const CaptureRequest &capture) {
    bool ok = true;
    RunHeadlessEngine(quiet, [&](entt::registry &r, entt::entity viewport) {
        ok = RunHeadlessScene(r, viewport, initial_file, empty, capture);
    });
    return ok;
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
void RunHeadlessQueue(const fs::path &spool, bool quiet, const CaptureRequest &harness) {
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
            RunHeadlessScene(r, viewport, initial_file, empty, CaptureRequest{.RenderBasename = job->OutBasename, .Overlays = harness.Overlays, .Edit = harness.Edit});
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
    const auto autorelease_pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

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
        else if (a == "--record-audio") capture.RecordAudio = true;
        else if (a == "--screenshot" && std::next(it) != args.end()) capture.ScreenshotPath = *++it;
        else if (a == "--render" && std::next(it) != args.end()) capture.RenderBasename = *++it;
        else if (a == "--render-queue" && std::next(it) != args.end()) render_queue = *++it;
        else if (a == "--empty") empty = true;
        else if (a == "--headless") headless = true;
        else if (a == "--overlays") capture.Overlays = true;
        else if (a == "--edit") capture.Edit = true;
        else if (a == "--fps" && std::next(it) != args.end()) capture.Fps = std::atoi(*++it);
        else if (a == "--timeline-end" && std::next(it) != args.end()) capture.TimelineEnd = std::atof(*++it);
        else if (a == "--motion-blur" && std::next(it) != args.end()) capture.MotionBlurSteps = uint8_t(std::max(1, std::atoi(*++it)));
        else if (a == "--frames" && std::next(it) != args.end()) capture.BenchFrames = std::atoi(*++it);
        else if (a == "--bench-action" && std::next(it) != args.end()) {
            const std::string_view action{*++it};
            if (action == "steady") capture.BenchAction = CaptureRequest::BenchmarkAction::Steady;
            else if (action == "orbit") capture.BenchAction = CaptureRequest::BenchmarkAction::Orbit;
            else if (action == "transform") capture.BenchAction = CaptureRequest::BenchmarkAction::Transform;
            else if (action == "visibility") capture.BenchAction = CaptureRequest::BenchmarkAction::Visibility;
            else {
                std::println(stderr, "Unknown benchmark action '{}'.", action);
                return 1;
            }
        } else if (a == "--bench-action-count" && std::next(it) != args.end()) {
            capture.BenchActionCount = uint32_t(std::max(1, std::atoi(*++it)));
        }
        else if (a == "--camera" && std::next(it) != args.end()) capture.CameraName = *++it;
        else if (a == "--shading" && std::next(it) != args.end()) {
            const std::string_view mode{*++it};
            capture.Shading = mode == "wireframe" ? ViewportShadingMode::Wireframe :
                mode == "solid"                   ? ViewportShadingMode::Solid :
                mode == "preview"                 ? ViewportShadingMode::MaterialPreview :
                                                    ViewportShadingMode::Rendered;
        } else if (a == "--profile") profile::Enabled = true;
        else if (a == "--profile-json" && std::next(it) != args.end()) {
            profile::Enabled = true;
            profile::JsonPath = *++it;
        } else if (!a.starts_with('-') && !initial_file) initial_file = *it;
    }
    if (capture.Fps <= 0) capture.Fps = 60;
    // A wav target records audio only, so the audio flag is implied.
    if (capture.RecordPath.extension() == ".wav") capture.RecordAudio = true;
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

    bool headless_ok = true;
    if (!render_queue.empty()) RunHeadlessQueue(render_queue, quiet, capture);
    else if (headless) headless_ok = RunHeadless(initial_file, quiet, empty, capture);
    else run(initial_file, quiet, empty, capture);
    // A capture that ffmpeg rejected leaves a truncated or missing file, and a headless scene that failed to load rendered nothing, so a clean status would hide either.
    return !headless_ok || VideoRecorder::AnyFailed() ? 1 : 0;
}
