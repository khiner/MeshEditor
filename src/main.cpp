#include "Compress.h"
#include "File.h"
#include "FileDialog.h"
#include "LogEnabled.h"
#include "MacPlatform.h"
#include "Paths.h"
#include "ProcessEvents.h"
#include "Profile.h"
#include "TransformMath.h"
#include "VideoRecorder.h"
#include "Window.h"
#include "WorkspaceState.h"
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
#include "gltf/GltfScene.h"
#include "image/ImageEncode.h"
#include "mesh/MeshComponents.h"
#include "metal/Image.h"
#include "metal/MetalContext.h"
#include "metal/RenderTarget.h"
#include "object/ObjectOps.h"
#include "physics/PhysicsTypes.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "scene/SceneControlsUi.h"
#include "scene/WorldTransform.h"
#include "snapshot/ReplayTestFixture.h"
#include "snapshot/SaveState.h"
#include "snapshot/SceneSnapshot.h"
#include "viewport/FrameState.h"
#include "viewport/ViewCamera.h"
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

#include <array>
#include <bit>
#include <chrono>
#include <csignal>
#include <cstdlib>
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
// Skip a frame when every drawable is in flight.
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

constexpr float FontAtlasScale = 2; // Supersample the font atlas for sharper text.
void AddFont(FontFamily family, const std::string_view font_file) {
    static const auto FontsPath = Paths::Res() / "fonts";
    static constexpr auto PixelsForFamily = [] {
        std::array<uint, size_t(FontFamily::Count)> v{};
        v[size_t(FontFamily::Main)] = 15;
        v[size_t(FontFamily::Monospace)] = 17;
        return v;
    }();
    GetIO().Fonts->AddFontFromFileTTF((FontsPath / font_file).c_str(), PixelsForFamily[size_t(family)] * FontAtlasScale);
}
void InitFonts(float scale = 1.f) {
    AddFont(FontFamily::Main, "Inter-Regular.ttf");
    AddFont(FontFamily::Monospace, "JetBrainsMono-Regular.ttf");
    ImGui::GetIO().FontGlobalScale = scale / FontAtlasScale;
}

} // namespace

namespace {
struct GltfSample {
    std::string Label;
    fs::path Path;
    std::set<std::string> Extensions;
};

struct GltfSampleTree {
    std::map<std::string, GltfSampleTree> Children;
    std::vector<GltfSample> Files;
};

// Return a glTF or GLB file's top-level extensionsUsed names without constructing an Asset.
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

// Collect every .glb and .gltf under root, preserving files with duplicate stems in variant directories.
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

// Mirror root while collapsing redundant directory levels:
// Merge an empty directory with its single child (AnimatedCube/glTF/ -> AnimatedCube/).
// Flatten a directory with one file whose stem repeats the directory name (AnimatedCube/AnimatedCube.gltf -> AnimatedCube.gltf).
// Preserve directories that contain multiple format variants.
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

// Apply an action and update derived scene state outside the main loop.
template<typename ActionType> void Perform(entt::registry &r, entt::entity viewport, ActionType action) {
    action::ApplyNow(r, viewport, std::move(action));
    ProcessComponentEvents(r, viewport);
}

// Finish GPU work and stop playback before modifying scene structure.
void QuiesceScene(entt::registry &r, entt::entity viewport) {
    WaitForRender(r);
    const auto &playback = r.get<const TimelinePlayback>(viewport);
    if (playback.Playing) action::ApplyNow(r, viewport, action::timeline::TogglePlay{playback.CurrentFrame});
}

constexpr std::string_view SessionLogName{"session.actions"}, ProjectStateName{"project.state"}, ProjectExt{".project"}, ActionsExt{".actions"};
fs::path CurrentProjectPath;
uint64_t RestoreGeneration{};
fs::path CachedWorkspacePath;
std::vector<std::byte> CachedWorkspaceBytes;

workspace::State CaptureWorkspace(entt::registry &r, entt::entity viewport) {
    return workspace::Capture(r, viewport, r.ctx().get<const WindowsState>());
}

void SaveWorkspace(entt::registry &r, entt::entity viewport, bool force = true) {
    if (Paths::Project().empty()) return;
    const auto path = Paths::Project() / workspace::FileName;
    auto bytes = workspace::Serialize(CaptureWorkspace(r, viewport));
    if (!force && path == CachedWorkspacePath && bytes == CachedWorkspaceBytes) return;
    if (!workspace::Save(path, bytes)) {
        std::println(stderr, "Failed to save workspace state.");
        return;
    }
    CachedWorkspacePath = path;
    CachedWorkspaceBytes = std::move(bytes);
}

bool UiGestureSettled() {
    if (GImGui->MovingWindow || GImGui->DragDropActive) return false;
    const auto &io = GetIO();
    for (int button = 0; button < ImGuiMouseButton_COUNT; ++button) {
        if (io.MouseDown[button] || io.MouseReleased[button]) return false;
    }
    return true;
}

void StartScratchSession(entt::registry &r, entt::entity viewport) {
    QuiesceScene(r, viewport);
    SaveWorkspace(r, viewport);
    action::StopLog();
    Paths::SetProject(action::ReserveRestoreSession());
    ClearScene(r, viewport);
    ++RestoreGeneration;
    action::StartLog(Paths::Project() / SessionLogName);
    CurrentProjectPath.clear();
}

// Start a scratch session with the default or empty scene.
void NewScene(entt::registry &r, entt::entity viewport, bool empty) {
    StartScratchSession(r, viewport);
    if (!empty) action::Emit(action::io::LoadDefaultScene{});
}

// Clear the scene and replay an action log in the current project directory.
void ReplayLogInPlace(entt::registry &r, entt::entity viewport, const fs::path &log_path) {
    auto live_workspace = CaptureWorkspace(r, viewport);
    QuiesceScene(r, viewport);
    action::FlushLog();
    ClearScene(r, viewport);
    std::error_code ec;
    const bool replaying_session_log = fs::equivalent(log_path, Paths::Project() / SessionLogName, ec);
    action::ReplayLog(
        r, viewport, log_path, &PresentViewport, 0, std::numeric_limits<uint64_t>::max(),
        /*record=*/!replaying_session_log
    );
    workspace::Apply(r, viewport, r.ctx().get<WindowsState>(), live_workspace);
    PresentViewport(r, viewport);
    ++RestoreGeneration;
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

// Restore a project from its base snapshot and a bounded suffix of its action log.
void RestoreProject(
    entt::registry &r, entt::entity viewport, const fs::path &working_dir,
    uint64_t action_end = std::numeric_limits<uint64_t>::max(), const workspace::State *workspace_override = nullptr
) {
    const auto state_path = working_dir / ProjectStateName, log_path = working_dir / SessionLogName;
    uint64_t skip = 0;
    if (std::error_code ec; fs::exists(state_path, ec)) skip = LoadStateBase(r, viewport, state_path);
    else ClearScene(r, viewport);

    const auto fallback_workspace = CaptureWorkspace(r, viewport);
    const auto count = action_end > skip ? action_end - skip : 0;
    action::ReplayLog(r, viewport, log_path, &PresentViewport, skip, count);
    const auto stored_workspace = workspace_override ? std::optional<workspace::State>{*workspace_override} : workspace::Load(working_dir / workspace::FileName);
    workspace::Apply(r, viewport, r.ctx().get<WindowsState>(), stored_workspace.value_or(fallback_workspace));
    PresentViewport(r, viewport);
}

// Restore a session from its base snapshot and subsequent action log.
void OpenProjectDir(entt::registry &r, entt::entity viewport, const fs::path &working_dir) {
    QuiesceScene(r, viewport);
    SaveWorkspace(r, viewport);
    action::StopLog();
    CurrentProjectPath.clear();
    Paths::SetProject(working_dir);
    RestoreProject(r, viewport, working_dir);
    ++RestoreGeneration;
    action::StartLog(working_dir / SessionLogName, /*append=*/true);
}

// Open a .project archive in a new working directory.
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
    else if (ext == ActionsExt) ReplayLogInPlace(r, viewport, path);
    else action::Emit(action::io::Load{.Path = path});
}

// Save a scene snapshot and project archive at archive_path.
void SaveProjectFile(entt::registry &r, entt::entity viewport, const fs::path &archive_path) {
    Perform(r, viewport, action::io::SaveState{.Path = Paths::Project() / ProjectStateName});
    SaveWorkspace(r, viewport);
    const auto log_path = Paths::Project() / SessionLogName;
    action::StopLog(); // Flush the log before archiving.
    const bool ok = Compress(Paths::Project(), archive_path);
    action::StartLog(log_path, /*append=*/true);
    if (!ok) {
        std::println(stderr, "Failed to save project '{}'", archive_path.string());
        return;
    }
    CurrentProjectPath = archive_path;
}

// Replace the session history with a snapshot of the current scene.
void ClearHistory(entt::registry &r, entt::entity viewport) {
    r.emplace_or_replace<ActionIndex>(viewport); // Reset session bookkeeping outside action Apply.
    Perform(r, viewport, action::io::SaveState{.Path = Paths::Project() / ProjectStateName});
    SaveWorkspace(r, viewport);
    action::StopLog();
    std::error_code ec;
    fs::remove(Paths::Project() / SessionLogName, ec);
    fs::remove_all(ModalModelsDir(), ec);
    action::StartLog(Paths::Project() / SessionLogName);
}

void BuildDefaultDockLayout(const WindowsState &windows, ImGuiID dockspace_id) {
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

void RenderDebugWindow(entt::registry &r, const mtl::Context &ctx, CA::MetalLayer *layer, WindowsState &windows, const ImGuiIO &io) {
    if (!windows.Debug.Visible) return;
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

struct EditorWindowsFrame {
    uvec2 ViewportExtent{};
    bool ViewportBegun{false};
    bool ViewportOpen{false};
};

EditorWindowsFrame BeginEditorWindows(
    entt::registry &r, entt::entity viewport, const mtl::Context &ctx, CA::MetalLayer *layer,
    WindowsState &windows, const ImGuiIO &io, bool interactive
) {
    RenderDebugWindow(r, ctx, layer, windows, io);
    if (windows.ImGuiDemo.Visible) ShowDemoWindow(&windows.ImGuiDemo.Visible);
    if (windows.ImSpinnerDemo.Visible) {
        if (Begin(windows.ImSpinnerDemo.Name, &windows.ImSpinnerDemo.Visible)) ImSpinner::demoSpinners();
        End();
    }
    if (windows.ImPlotDemo.Visible) ImPlot::ShowDemoWindow(&windows.ImPlotDemo.Visible);
    if (windows.SceneControls.Visible) {
        if (Begin(windows.SceneControls.Name, &windows.SceneControls.Visible)) RenderControls(r, viewport);
        End();
    }

    bool scrubbing = false;
    if (windows.Animation.Visible) {
        PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        if (Begin(windows.Animation.Name, &windows.Animation.Visible, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
            PushStyleVar(ImGuiStyleVar_FramePadding, {6, 4});
            Indent(6);
            Spacing();
            RenderClipPickers(r);
            Unindent(6);
            PopStyleVar();
            if (auto action = RenderAnimationTimeline(
                    r.get<const TimelineRange>(viewport), r.get<const TimelinePlayback>(viewport),
                    r.get<const AnimationTimelineView>(viewport), r.ctx().get<const ViewportIcons>().Anim, scrubbing
                );
                interactive && action) {
                std::visit([](auto leaf) { action::Emit(leaf); }, std::move(*action));
            }
        }
        End();
        PopStyleVar();
    }
    r.ctx().get<FrameState>().Scrubbing = scrubbing;

    EditorWindowsFrame frame;
    if (windows.Viewport.Visible) {
        frame.ViewportBegun = true;
        PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        frame.ViewportOpen = Begin(windows.Viewport.Name, &windows.Viewport.Visible);
        if (frame.ViewportOpen) {
            if (interactive) Interact(r, viewport, r.ctx().get<FrameState>());
            auto &draw_list = *GetWindowDrawList();
            draw_list.ChannelsSplit(2);
            draw_list.ChannelsSetCurrent(1);
            InteractOverlay(r, viewport, r.ctx().get<FrameState>());
            DrawModalJobsOverlay(r);
            const auto content_region = GetContentRegionAvail();
            frame.ViewportExtent = {
                uint32_t(std::max(content_region.x, 0.f)),
                uint32_t(std::max(content_region.y, 0.f)),
            };
        }
    }
    return frame;
}

void EndEditorViewport(const EditorWindowsFrame &frame) {
    if (!frame.ViewportBegun) return;
    End();
    PopStyleVar();
}

#ifdef DEBUG_BUILD
struct ValidationImage {
    std::vector<std::byte> Pixels;
    uint32_t Width{}, Height{};
};

fs::path WriteValidationImage(std::string_view name, const ValidationImage &image) {
    auto rgba = image.Pixels;
    for (size_t i = 0; i < rgba.size(); i += 4) std::swap(rgba[i], rgba[i + 2]);
    const auto encoded = EncodeImagePngRgba8(rgba, image.Width, image.Height, name);
    if (!encoded) return {};
    const auto path = fs::temp_directory_path() / std::format("MeshEditor-validation-{}.png", name);
    std::ofstream out{path, std::ios::binary};
    out.write(reinterpret_cast<const char *>(encoded->data()), std::streamsize(encoded->size()));
    return out ? path : fs::path{};
}

ValidationImage RenderAppImage(const mtl::Context &ctx, ImDrawData *draw_data) {
    const auto pixel_width = uint32_t(std::ceil(draw_data->DisplaySize.x * draw_data->FramebufferScale.x));
    const auto pixel_height = uint32_t(std::ceil(draw_data->DisplaySize.y * draw_data->FramebufferScale.y));
    if (pixel_width == 0 || pixel_height == 0) return {};

    auto target = mtl::CreateUntrackedTexture2D(
        ctx, mtl::Format::Color, {pixel_width, pixel_height},
        MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead, MTL::StorageModeShared
    );
    const std::array colors{mtl::ClearColor(*target, {0.45, 0.55, 0.60, 1.0})};
    auto *pass = mtl::MakePassDescriptor(colors);
    ImGui_ImplMetal_NewFrame(pass);

    auto *command_buffer = ctx.Queue->commandBuffer();
    auto *encoder = command_buffer->renderCommandEncoder(pass);
    ImGui_ImplMetal_RenderDrawData(draw_data, command_buffer, encoder);
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    if (const auto *error = command_buffer->error()) {
        throw std::runtime_error(std::format("Failed to render the validation image: {}", error->localizedDescription()->utf8String()));
    }
    return {ReadbackImageRgba8(ctx, target, 0, 0, target.Extent), pixel_width, pixel_height};
}

struct ValidationInputs {
    fs::path WorkingDir;
    uint64_t ActionEnd{};
    workspace::State Workspace;
    CA::MetalLayer *Layer{};
    ImVec2 DisplaySize{}, FramebufferScale{}, MousePos{};
    ImGuiID HoveredIdPreviousFrame{};
    float HoveredIdTimer{}, HoveredIdNotActiveTimer{};
    std::string FocusedWindow;
};

ValidationImage RenderValidationApp(
    entt::registry &r, entt::entity viewport, const ValidationInputs &inputs
) {
    auto &ctx = r.ctx().get<const mtl::Context>();
    auto &io = GetIO();
    auto &windows = r.ctx().get<WindowsState>();
    io.MousePos = {-FLT_MAX, -FLT_MAX};
    const auto render_frame = [&](bool restore_hover = false) {
        io.DisplaySize = inputs.DisplaySize;
        io.DisplayFramebufferScale = inputs.FramebufferScale;
        io.DeltaTime = 1.f / 60.f;
        NewFrame();
        if (restore_hover) {
            GImGui->HoveredIdPreviousFrame = inputs.HoveredIdPreviousFrame;
            GImGui->HoveredIdTimer = inputs.HoveredIdTimer;
            GImGui->HoveredIdNotActiveTimer = inputs.HoveredIdNotActiveTimer;
        }

        auto dockspace_id = DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_AutoHideTabBar);
        if (!windows.LayoutLoaded) {
            BuildDefaultDockLayout(windows, dockspace_id);
            windows.LayoutLoaded = true;
        }
        if (BeginMainMenuBar()) {
            if (BeginMenu("File")) EndMenu();
            if (BeginMenu("Windows")) EndMenu();
            EndMainMenuBar();
        }
        const auto frame = BeginEditorWindows(r, viewport, ctx, inputs.Layer, windows, io, /*interactive=*/false);
        if (frame.ViewportOpen) {
            DisplayViewport(r, viewport);
            GetWindowDrawList()->ChannelsMerge();
        }
        EndEditorViewport(frame);
        workspace::ApplyPendingTabs(windows);
        Render();
    };

    render_frame(); // Instantiate the windows referenced by the restored dock layout.
    SetWindowFocus(inputs.FocusedWindow.empty() ? nullptr : inputs.FocusedWindow.c_str());
    render_frame(); // Bind those windows to their dock nodes.
    render_frame(); // Settle docked child sizes and scrollbars.
    io.MousePos = inputs.MousePos;
    render_frame(/*restore_hover=*/true); // Apply live hover history after the restored hit regions settle.
    return RenderAppImage(ctx, GetDrawData());
}

struct ValidationResult {
    std::vector<std::byte> State;
    std::vector<std::byte> SceneState;
    std::vector<std::byte> Workspace;
    ValidationImage App;
    float Milliseconds{};
};

ValidationResult RestoreForValidation(
    const ValidationInputs &inputs, const std::vector<std::byte> *snapshot_state
) {
    entt::registry restored;
    restored.ctx().emplace<mtl::Context>();
    const auto restored_viewport = InitEngine(restored);
    InitAudioSystem(restored);

    auto *live_imgui = ImGui::GetCurrentContext();
    auto *live_implot = ImPlot::GetCurrentContext();
    auto *live_fonts = GetIO().Fonts;
    const auto live_font_scale = GetIO().FontGlobalScale;
    auto *validation_imgui = ImGui::CreateContext(live_fonts);
    ImGui::SetCurrentContext(validation_imgui);
    auto *validation_implot = ImPlot::CreateContext();
    ImPlot::SetCurrentContext(validation_implot);
    auto &io = GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
    io.IniFilename = nullptr;
    io.FontGlobalScale = live_font_scale;
    StyleColorsDark();
    ImGui_ImplMetal_Init(restored.ctx().get<const mtl::Context>().Device.get());
    InitViewportMedia(restored);
    SetupScene(restored, restored_viewport);
    restored.ctx().get<FrameState>().DisplayFramebufferScale = std::bit_cast<vec2>(inputs.FramebufferScale);
    ProcessComponentEvents(restored, restored_viewport);

    const auto begin = SteadyClock::now();
    if (snapshot_state) {
        snapshot::LoadState(restored, *snapshot_state);
        ProcessComponentEvents(restored, restored_viewport);
        workspace::Apply(restored, restored_viewport, restored.ctx().get<WindowsState>(), inputs.Workspace);
        PresentViewport(restored, restored_viewport);
    } else {
        RestoreProject(restored, restored_viewport, inputs.WorkingDir, inputs.ActionEnd, &inputs.Workspace);
    }
    auto app_image = RenderValidationApp(restored, restored_viewport, inputs);
    const auto elapsed = std::chrono::duration<float, std::milli>(SteadyClock::now() - begin).count();
    auto state = snapshot::SaveState(restored);
    auto scene_state = snapshot::SnapshotSceneState(restored);
    auto restored_workspace = workspace::Serialize(CaptureWorkspace(restored, restored_viewport));

    WaitForRender(restored);
    DeinitViewportMedia(restored);
    DeinitAudioSystem(restored);
    DeinitViewport(restored, restored_viewport);
    ImGui_ImplMetal_Shutdown();
    ImPlot::DestroyContext(validation_implot);
    ImGui::DestroyContext(validation_imgui);
    ImGui::SetCurrentContext(live_imgui);
    ImPlot::SetCurrentContext(live_implot);

    return {
        std::move(state), std::move(scene_state), std::move(restored_workspace),
        std::move(app_image), elapsed
    };
}

void RequireEqual(std::string_view what, std::span<const std::byte> expected, std::span<const std::byte> actual) {
    if (const auto diff = snapshot::Compare(expected, actual); !diff.Equal) {
        std::println(
            stderr, "[validation] {} DIVERGED at byte {} (expected {} / actual {})",
            what, diff.FirstDifferingByte, expected.size(), actual.size()
        );
        std::abort();
    }
}

void RequireEqualImage(std::string_view what, const ValidationImage &expected, const ValidationImage &actual) {
    if (expected.Width != actual.Width || expected.Height != actual.Height) {
        std::println(stderr, "[validation] {} extent DIVERGED ({}x{} / {}x{})", what, expected.Width, expected.Height, actual.Width, actual.Height);
        std::abort();
    }
    if (const auto diff = snapshot::Compare(expected.Pixels, actual.Pixels); !diff.Equal) {
        const auto pixel = diff.FirstDifferingByte / 4;
        std::println(
            stderr, "[validation] {} DIVERGED at pixel ({}, {}), channel {} (byte {} of {})",
            what, pixel % expected.Width, pixel / expected.Width, diff.FirstDifferingByte % 4,
            diff.FirstDifferingByte, expected.Pixels.size()
        );
        const auto expected_path = WriteValidationImage("live", expected);
        const auto actual_path = WriteValidationImage(what, actual);
        if (!expected_path.empty() && !actual_path.empty()) {
            std::println(stderr, "[validation] wrote {} and {}", expected_path.string(), actual_path.string());
        }
        std::abort();
    }
}

// Require independent replay and snapshot restoration to reproduce canonical state and the live composited app image.
void ValidateRoundTrip(
    entt::registry &r, entt::entity viewport, CA::MetalLayer *layer,
    ImVec2 display_size, ImVec2 framebuffer_scale, const ValidationImage &live_app
) {
    QuiesceScene(r, viewport);
    action::FlushLog();

    const auto live_state = snapshot::SaveState(r);
    const auto live_scene_state = snapshot::SnapshotSceneState(r);
    const ValidationInputs inputs{
        .WorkingDir = Paths::Project(),
        .ActionEnd = r.get_or_emplace<ActionIndex>(viewport).Index,
        .Workspace = CaptureWorkspace(r, viewport),
        .Layer = layer,
        .DisplaySize = display_size,
        .FramebufferScale = framebuffer_scale,
        .MousePos = GetIO().MousePos,
        .HoveredIdPreviousFrame = GImGui->HoveredIdPreviousFrame,
        .HoveredIdTimer = GImGui->HoveredIdTimer,
        .HoveredIdNotActiveTimer = GImGui->HoveredIdNotActiveTimer,
        .FocusedWindow = GImGui->NavWindow ? std::string{GImGui->NavWindow->Name} : std::string{},
    };

    const auto replay = RestoreForValidation(inputs, nullptr);
    const auto restored = RestoreForValidation(inputs, &live_state);

    if (const auto diff = snapshot::Compare(live_scene_state, replay.SceneState); !diff.Equal) {
        std::println(
            stderr, "[validation] replay scene DIVERGED at byte {} (expected {} / actual {})",
            diff.FirstDifferingByte, live_scene_state.size(), replay.SceneState.size()
        );
        if (const auto fixture_dir = snapshot::WriteReplayTestFixture(
                inputs.WorkingDir / SessionLogName, live_scene_state, replay.SceneState
            );
            !fixture_dir.empty()) {
            std::println(stderr, "[validation] wrote replay-test fixture to {}", fixture_dir.string());
        }
        std::abort();
    }
    RequireEqual("replay state", live_state, replay.State);
    RequireEqual("snapshot state", live_state, restored.State);
    RequireEqual("replay/snapshot workspace", replay.Workspace, restored.Workspace);
    RequireEqualImage("replay app", live_app, replay.App);
    RequireEqualImage("snapshot app", live_app, restored.App);
    std::println("[validation] replay restore {:.1f} ms; snapshot restore {:.1f} ms", replay.Milliseconds, restored.Milliseconds);
}
#endif

// Write the viewport to path and return the resolved output path on success.
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

// Fit the scene into the middle half of the view and return false while GPU bounds are pending.
bool FrameScene(entt::registry &r, entt::entity viewport, float aspect_ratio) {
    const auto &cam = r.get<const ViewCamera>(viewport);
    const auto *persp = std::get_if<Perspective>(&cam.Data);
    if (!persp) return true;

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
        // Exclude empty bounds from gizmo and wireframe instances.
        const auto &local = buffers.Instances.GetBounds(ri.BufferIndex);
        if (local.Min.x > local.Max.x || local.Min.y > local.Max.y || local.Min.z > local.Max.z) continue;

        const auto m = ToMatrix(wt);
        for (int c = 0; c < 8; ++c) {
            const vec3 v{m * vec4{(c & 1) ? local.Max.x : local.Min.x, (c & 2) ? local.Max.y : local.Min.y, (c & 4) ? local.Max.z : local.Min.z, 1.f}};
            scene.Min = numeric::Min(scene.Min, v);
            scene.Max = numeric::Max(scene.Max, v);
            const float a = numeric::Dot(v, right), b = numeric::Dot(v, up), f = numeric::Dot(v, away);
            top = std::max(top, b / ty + f);
            bottom = std::max(bottom, -b / ty + f);
            rgt = std::max(rgt, a / tx + f);
            lft = std::max(lft, -a / tx + f);
        }
    }
    if (scene.Min.x > scene.Max.x || scene.Min.y > scene.Max.y || scene.Min.z > scene.Max.z) return !any_bounded_instance;

    const auto center = (scene.Min + scene.Max) * 0.5f;
    const float ca = numeric::Dot(center, right), cb = numeric::Dot(center, up), cf = numeric::Dot(center, away);
    const float distance = std::max({top - cb / ty, bottom + cb / ty, rgt - ca / tx, lft + ca / tx}) - cf;
    if (distance <= 0.f) return true;

    // Clip planes bracket the scene depth so nothing is z-clipped.
    const float plane_reach = 6 * numeric::Length(scene.Max - scene.Min);
    auto fit = *persp;
    fit.FarClip = distance + plane_reach;
    fit.NearClip = std::max(distance - plane_reach, *fit.FarClip / 10000.f);

    r.replace<ViewCamera>(viewport, ViewCamera{center + distance * away, center, Camera{fit}});
    return true;
}

// Use the default window extent for headless rendering.
constexpr uvec2 DefaultWindowSize{1280, 800};

constexpr std::string_view Usage{
    R"(Usage: MeshEditor [scene] [options]

Loads `scene` (a .gltf/.glb/project file) into the editor, or the default scene when none is given.

Scene:
  --empty                     Start with an empty scene
  --camera NAME               Frame the named camera
  --shading MODE              wireframe | solid | preview | rendered
  --edit ELEMENT              Enter edit mode on vertex | edge | face
  --select-all                Select every element of the edited mesh
  --overlays                  Draw the editor overlays
  --lod-error PIXELS          Screen-space error budget for the cluster LOD cut
  --display LIST              Comma-separated: vertex-normals, face-normals, bounds, tet-wireframe

Capture:
  --screenshot PATH           Write one image and exit
  --record PATH               Record a video, or audio alone for a .wav path
  --record-audio              Record audio alongside the video
  --render BASENAME           Write one scene's corpus artifacts under BASENAME
  --render-queue DIR          Claim jobs from a render spool until it empties (headless)
  --fps N                     Capture frame rate (default 60)
  --timeline-end SECONDS      Stop the captured timeline at SECONDS
  --motion-blur STEPS         Accumulate STEPS sub-frames per captured frame
  --play [SECONDS]            Play the timeline, optionally for SECONDS

Benchmarking:
  --headless                  Run without a window
  --frames N                  Render N frames and exit
  --bench-action ACTION       steady | orbit | transform | visibility | box-select
  --bench-action-count N      Actions per benchmark run
  --profile                   Print the profile report on exit
  --profile-json PATH         Write the profile report to PATH

Other:
  --quiet, -q                 Suppress progress output
  --help, -h                  Show this message)"
};

// Capture options parsed from the command line.
struct CaptureRequest {
    enum class BenchmarkAction { Steady,
                                 Orbit,
                                 Transform,
                                 Visibility,
                                 BoxSelect };

    bool Play{false};
    float PlayDuration{0};
    int Fps{60};
    bool RecordAudio{false}; // Mux the master output into the recording. Off so the render corpus stays video only.
    fs::path RecordPath{}, ScreenshotPath{};
    fs::path RenderBasename{};
    std::optional<uint8_t> MotionBlurSteps{};
    float TimelineEnd{0}; // Seconds. Positive: set the timeline's end frame, so a long play runs without looping.
    int BenchFrames{0};
    BenchmarkAction BenchAction{BenchmarkAction::Steady};
    uint32_t BenchActionCount{64};
    std::string CameraName{};
    std::optional<ViewportShadingMode> Shading{};
    bool Overlays{false}; // Keep overlays on through a capture, which presentation otherwise turns off.
    std::optional<Element> EditMode{}; // Engaged: select mesh objects and enter this element edit mode.
    bool SelectAll{false};
    float LodErrorPixels{-1.f}; // Screen-space error budget override for the cluster LOD cut. Negative leaves the viewport setting untouched.
    uint8_t NormalOverlays{0};
    bool BoundingBoxes{false}, TetWireframe{false};
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

    void Apply(entt::registry &r, entt::entity viewport, uvec2 extent) {
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
            case CaptureRequest::BenchmarkAction::BoxSelect: {
                if (extent == uvec2{}) break;
                const auto &camera = r.get<const ViewCamera>(viewport);
                const float aspect = float(extent.x) / float(extent.y);
                const uint32_t inset = Frame % 2 == 0 ? 4u : 8u;
                action::Emit(action::selection::ApplyBoxSelect{
                    .BoxPx = {{inset, inset}, {extent.x - inset - 1, extent.y - inset - 1}},
                    .Additive = false,
                    .ViewProj = std::make_unique<mat4>(camera.Projection(aspect) * camera.View()),
                });
                break;
            }
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

// Report and clear action failures, returning whether any occurred.
bool ReportActionErrors(entt::registry &r) {
    auto &errors = r.ctx().get<action::Errors>().Messages;
    if (errors.empty()) return false;
    for (const auto &message : errors) std::cerr << message << std::endl;
    errors.clear();
    return true;
}

// Initialize the scene and session log, returning false when the initial file fails to load.
bool SeedScene(entt::registry &r, entt::entity viewport, const CaptureRequest &capture, const char *initial_file, bool empty) {
    const fs::path path = initial_file ? initial_file : "";
    bool loaded{true};
    if (path.extension() == ProjectExt) OpenProjectFile(r, viewport, path);
    else if (path.extension() == ActionsExt) {
        const auto imported = File::Read(path);
        if (!imported) {
            std::println(stderr, "{}", imported.error());
            loaded = false;
        } else {
            Paths::SetProject(action::ReserveRestoreSession());
            const auto session_log = Paths::Project() / SessionLogName;
            {
                std::ofstream out{session_log, std::ios::binary | std::ios::trunc};
                out.write(reinterpret_cast<const char *>(imported->data()), std::streamsize(imported->size()));
                loaded = bool(out);
            }
            if (loaded) {
                ReplayLogInPlace(r, viewport, session_log);
                action::StartLog(session_log, /*append=*/true);
            } else {
                std::println(stderr, "Failed to import action log '{}'.", path.string());
            }
        }
    } else {
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

// Coordinate scene framing, playback, screenshots, recording, and completion for both run loops.
struct CaptureDriver {
    // Fixed-step mode captures one timeline frame per GPU-paced tick; wall-clock mode samples at 1/Fps seconds.
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
        // Include audio in corpus videos only for scenes with sound objects.
        RecordAudio = capture.RecordAudio || (RenderMode() && !r.view<const ModalModes>().empty());
    }

    bool RenderMode() const { return !RenderBasename.empty(); }
    bool RecordingMode() const { return !RecordPath.empty(); }
    bool ScreenshotMode() const { return !ScreenshotPath.empty(); }
    // Skip GPU frames after audio-only recording begins.
    bool AudioOnly() const { return RecordPath.extension() == ".wav" && !ScreenshotMode(); }
    bool Presenting() const { return Play || ScreenshotMode() || RecordingMode(); }
    bool Framed(bool settled) const { return settled && (ViewFramed || !Presenting()); }

    bool DurationElapsed(const entt::registry &r, entt::entity viewport) const {
        if (PlayDuration <= 0) return false;
        const float elapsed = RecordingMode() ? float(CapturedFrameCount(r, viewport)) / float(RecordFps) : ElapsedPlayTime;
        return elapsed >= PlayDuration;
    }

    // Emit capture actions before ApplyEmitted after the viewport reaches its final extent.
    void EmitFrameActions(entt::registry &r, entt::entity viewport, bool settled, uvec2 extent) {
        if (!ViewFramed && Presenting() && extent != uvec2{}) {
            // Wait for GPU bounds before framing the launch camera.
            const bool framed = !r.view<const Camera>().empty() ||
                FrameScene(r, viewport, float(extent.x) / float(extent.y));
            ViewFramed = settled && framed;
        }
        // Start fixed-step playback after recording begins so the first frame is captured.
        const bool ready = RenderMode() || (FixedStep && RecordingMode()) ? IsRecording(r, viewport) : (Play || RecordingMode());
        if (!PlaybackStarted && Framed(settled) && ready) {
            action::Emit(action::timeline::StartPresentation{});
            PlaybackStarted = true;
        }
    }

    // Capture a completed render and return whether the run should end.
    bool CaptureFrame(entt::registry &r, entt::entity viewport, bool settled) {
        bool done = false;
        if (ScreenshotMode() && !ScreenshotSaved && settled) {
            if (auto saved = SaveScreenshot(r, ScreenshotPath); saved) std::println("Saved screenshot: {}", saved->string());
            else std::println(stderr, "Screenshot: {}", saved.error());
            // Save the default image before each material variant.
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
        // End the run if the single recording startup fails.
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
                // Emit clip switches to avoid an extra playback tick from mid-loop Perform calls.
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
            // End duration-less fixed-step playback after one timeline loop.
            done = true;
        }
        return done;
    }

    bool Play;
    bool SeedFailed{false};
    float PlayDuration;
    bool FixedStep;
    fs::path RecordPath, ScreenshotPath, RenderBasename;
    float RenderDt; // Fixed-step seconds per tick (one timeline frame).
    int RecordFps;
    bool RecordAudio;
    SteadyClock::time_point NextCapture;
    float ElapsedPlayTime{0}; // Caller-accumulated sim seconds, for the play-duration cap.
    uint32_t NextRenderClip{1}; // Next clip to capture once the current loop finishes.
    uint32_t NextRenderVariant{0}; // Next material variant to capture once the default image saves.
    bool PlaybackStarted{false}, ScreenshotSaved{false}, ViewFramed{false}, RecordingStarted{false};
};

// Initialize a capture session and configure its presentation state.
CaptureDriver BeginCaptureSession(entt::registry &r, entt::entity viewport, const CaptureRequest &capture, const char *initial_file, bool empty, bool fixed_step) {
    const bool seeded = SeedScene(r, viewport, capture, initial_file, empty);
    if (seeded && capture.EditMode) {
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
            Perform(r, viewport, action::view::SetEditMode{*capture.EditMode});
        }
    }
    if (seeded && capture.SelectAll) Perform(r, viewport, action::selection::SelectAll{});
    const bool play = seeded && capture.Play;
    // After the load, whose end frame comes from the scene's own animation durations.
    if (capture.TimelineEnd > 0) {
        const float fps = r.get<const TimelineRange>(viewport).Fps;
        Perform(r, viewport, action::timeline::SetEndFrame{int(std::ceil(capture.TimelineEnd * fps))});
    }
    CaptureDriver driver{r, viewport, capture, play, fixed_step};
    driver.SeedFailed = !seeded;
    // Preserve the editor view for benchmark frames and screenshots.
    if (driver.Presenting() && capture.BenchFrames == 0) Perform(r, viewport, action::timeline::EnterPresentation{});
    // Apply the explicit overlay override after enabling presentation mode.
    if (capture.Overlays) Perform(r, viewport, action::UpdateOf<&ViewportDisplay::ShowOverlays>(viewport, true));
    if (capture.LodErrorPixels >= 0.f) {
        Perform(r, viewport, action::UpdateOf<&ViewportDisplay::LodErrorPixels>(viewport, capture.LodErrorPixels));
    }
    r.ctx().get<FrameState>().FixedFrameStep = driver.FixedStep;
    // Enable motion blur for video recording and preserve the current setting for still or audio-only captures.
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
    // Disable present pacing for fixed-step rendering and throughput benchmarks.
    layer->setDisplaySyncEnabled(capture.RenderBasename.empty() && capture.BenchFrames == 0);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    auto &io = GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows
    io.IniFilename = nullptr;
    io.ConfigDebugIgnoreFocusLoss = true; // Keep input state across Cmd+Tab so in-flight gizmo drags survive focus loss.
    io.ConfigDragClickToInputText = true; // A click-release without dragging turns a Drag field into a text input.

    StyleColorsDark();

    window.InitImGui();
    ImGui_ImplMetal_Init(ctx.Device.get());

    InitFonts();

    const auto viewport = InitEngine(r);
    profile::Init(ctx);
    InitAudioSystem(r);
    InitViewportMedia(r);
    SetupScene(r, viewport);
    // Read the DPI scale from NewFrame before initializing DPI-scaled GPU state.
    window.NewImGuiFrame();
    r.ctx().get<FrameState>().DisplayFramebufferScale = std::bit_cast<vec2>(io.DisplayFramebufferScale);
    ProcessComponentEvents(r, viewport);

    auto &audio_device = r.ctx().emplace<AudioDeviceResource>(r, viewport);
    ReconcileAudioDevice(audio_device, r.get<const AudioOutputConfig>(viewport), r.get<const AudioOutputMix>(viewport));

    SampleTreesFuture = std::async(std::launch::async, BuildSampleTrees);

    auto driver = BeginCaptureSession(r, viewport, capture, initial_file, empty, /*fixed_step=*/false);

    bool viewport_resizing{false};
#ifdef DEBUG_BUILD
    bool validate_requested{false};
#endif
#ifdef VALIDATE_ACTIONS
    uint64_t validated_action_index{0};
    uint64_t validated_restore_generation{RestoreGeneration};
#endif
    bool done{false};
    uint8_t startup_frames_remaining{2};
    MTL::CommandBuffer *last_frame{nullptr}; // Resize waits for resources sampled by the last submitted UI frame.
    auto &windows = r.ctx().get<WindowsState>();
    int bench_ticks{0};
    while (!done) {
        // Report scene loading separately from frame timing.
        if (bench_ticks == 1) {
            profile::ReportCpuPhase("Scene load CPU timings");
            profile::ClearStats();
        }
        const profile::CpuScope frame_scope{"Frame"};
        auto events = window.PollEvents();
        r.ctx().get<FrameState>().PreciseWheelDelta += vec2{events.ScrollX, events.ScrollY};
        for (const auto &path : events.DroppedFiles) OpenFile(r, viewport, path);
        done = events.Quit;
        if (driver.DurationElapsed(r, viewport)) done = true;

#ifdef DEBUG_BUILD
        const auto ui_action_index = r.get_or_emplace<ActionIndex>(viewport).Index;
        const auto ui_restore_generation = RestoreGeneration;
#endif

        window.NewImGuiFrame();
        driver.ElapsedPlayTime += io.DeltaTime;
        // Scene-affecting code reads FrameState::DeltaTime. `io.DeltaTime` is wall-clock, UI-only.
        r.ctx().get<FrameState>().DeltaTime = driver.FixedStep ? driver.RenderDt : io.DeltaTime;
        NewFrame();

        auto dockspace_id = DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_AutoHideTabBar);
        if (!windows.LayoutLoaded) {
            BuildDefaultDockLayout(windows, dockspace_id);
            windows.LayoutLoaded = true;
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
                if (MenuItem("[Debug] Roundtrip")) validate_requested = true;
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

        // Keep the viewport window open across apply/derive/render so its image is inserted before End().
        const auto editor_windows = BeginEditorWindows(r, viewport, ctx, layer, windows, io, /*interactive=*/true);
        const auto new_logical_extent = editor_windows.ViewportExtent;
        const bool viewport_settled = new_logical_extent != uvec2{} && new_logical_extent == r.ctx().get<const ViewportExtent>().Value;
        if (capture.BenchFrames > 0 && viewport_settled) {
            if (capture.BenchAction == CaptureRequest::BenchmarkAction::Orbit) action::Emit(action::view::OrbitViewCamera{{0.01f, 0.f}});
            if (++bench_ticks >= capture.BenchFrames) done = true;
        }
        // Give interaction actions priority in the single-action buffer.
        driver.EmitFrameActions(r, viewport, viewport_settled, new_logical_extent);
        // Stage resize drags and commit one SetExtent on mouse-up for deterministic replay.
        if (new_logical_extent != uvec2{} && r.ctx().get<const ViewportExtent>().Value != new_logical_extent) {
            action::EmitStaged(action::view::SetExtent{new_logical_extent});
            viewport_resizing = true;
        } else if (viewport_resizing && !IsMouseDown(ImGuiMouseButton_Left)) {
            action::Commit();
            viewport_resizing = false;
        }

        action::ApplyEmitted(r, viewport);
        ReportActionErrors(r);

        // Submit derived state and rendering before the later WaitForRender synchronizes image sampling.
        SubmitViewport(r, viewport, GetFrameCount() > 1 ? last_frame : nullptr);

        if (editor_windows.ViewportOpen) {
            DisplayViewport(r, viewport);
            GetWindowDrawList()->ChannelsMerge();
        }
        EndEditorViewport(editor_windows);
        workspace::ApplyPendingTabs(windows);

        ImGui::Render();
        window.HonorMouseWarp();
        auto *draw_data = GetDrawData();
        const bool ui_gesture_settled = UiGestureSettled();
        if (const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f); !is_minimized) {
            WaitForRender(r); // Synchronize before ImGui samples the final image.
            if (driver.CaptureFrame(r, viewport, driver.Framed(viewport_settled))) done = true;

#ifdef DEBUG_BUILD
            const auto action_index = r.get_or_emplace<ActionIndex>(viewport).Index;
            bool validate = validate_requested;
#ifdef VALIDATE_ACTIONS
            if (action_index != validated_action_index || RestoreGeneration != validated_restore_generation) {
                validate = true;
            }
#endif
            const bool ui_matches_scene =
                ui_action_index == action_index &&
                ui_restore_generation == RestoreGeneration;
            const bool validation_ready = validate && ui_gesture_settled;
            const bool present_frame = !validation_ready || ui_matches_scene;
#else
            constexpr bool present_frame{true};
#endif

            MTL::CommandBuffer *presented_frame{nullptr};
            if (present_frame) {
                if (auto *frame = RenderAndPresentFrame(ctx, layer, draw_data)) {
                    // ImGui makes newly bound dock nodes visible on the following frame.
                    if (startup_frames_remaining > 0 && --startup_frames_remaining == 0) {
                        frame->waitUntilCompleted();
                        window.Show();
                    }
                    last_frame = presented_frame = frame;
                }
            }

#ifdef DEBUG_BUILD
            if (validation_ready && presented_frame && GetFrameCount() > 1 && viewport_settled && ViewportImageReady(r)) {
                presented_frame->waitUntilCompleted();
                const auto live_app = RenderAppImage(ctx, draw_data);
                ValidateRoundTrip(r, viewport, layer, draw_data->DisplaySize, draw_data->FramebufferScale, live_app);
                validate_requested = false;
#ifdef VALIDATE_ACTIONS
                validated_action_index = action_index;
                validated_restore_generation = RestoreGeneration;
#endif
            }
#endif
        }

        static auto next_workspace_save = SteadyClock::now();
        if (ui_gesture_settled && (io.WantSaveIniSettings || SteadyClock::now() >= next_workspace_save)) {
            SaveWorkspace(r, viewport, /*force=*/false);
            io.WantSaveIniSettings = false;
            next_workspace_save = SteadyClock::now() + std::chrono::milliseconds{250};
        }
    }

    SaveWorkspace(r, viewport);
    action::StopLog();
    if (last_frame) last_frame->waitUntilCompleted();

    r.ctx().erase<AudioDeviceResource>();
    DeinitAudioSystem(r);

    // GpuBuffers must outlive MeshStore allocations retired during teardown.
    DeinitViewportMedia(r);
    profile::Report();
    profile::Deinit();
    DeinitViewport(r, viewport);

    ImGui_ImplMetal_Shutdown();
    window.ShutdownImGui();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
}

// Run one fixed-step capture session and finalize its action log.
bool RunHeadlessScene(entt::registry &r, entt::entity viewport, const char *initial_file, bool empty, const CaptureRequest &capture) {
    const auto scene_start = std::chrono::steady_clock::now();
    auto driver = BeginCaptureSession(r, viewport, capture, initial_file, empty, /*fixed_step=*/true);
    // Stop after reporting an initial scene-load failure.
    if (driver.SeedFailed) {
        action::StopLog();
        return false;
    }
    // Emit the resize so the first SubmitViewport recreates images before rendering.
    action::Emit(action::view::SetExtent{DefaultWindowSize});
    if (capture.NormalOverlays != 0) Perform(r, viewport, action::UpdateOf<&ViewportDisplay::NormalOverlays>(viewport, capture.NormalOverlays));
    if (capture.BoundingBoxes) Perform(r, viewport, action::UpdateOf<&ViewportDisplay::ShowBoundingBoxes>(viewport, true));
    if (capture.TetWireframe) Perform(r, viewport, action::UpdateOf<&ViewportDisplay::ShowTetWireframe>(viewport, true));

    auto &frame_state = r.ctx().get<FrameState>();
    frame_state.DeltaTime = driver.RenderDt;
    int bench_frames = capture.BenchFrames;
    BenchmarkDriver benchmark{r, capture};
    bool profile_cleared{false};
    bool submitted{false};
    bool done{false};
    // Record readiness after the first submitted frame has complete meshlet data.
    std::chrono::steady_clock::time_point first_frame_at{};
    while (!done) {
        if (driver.DurationElapsed(r, viewport)) break;
        const auto extent = r.ctx().get<const ViewportExtent>().Value;
        // Require this scene's first submit because queue workers preserve ready images across scenes.
        const bool settled = submitted && ViewportImageReady(r);
        // Report scene loading separately from frame timing.
        if (settled && !profile_cleared) {
            if (profile::Enabled) {
                const auto ms_since_start = [scene_start](auto point) { return std::chrono::duration<float, std::milli>(point - scene_start).count(); };
                std::println(
                    "Scene ready: first frame {:.1f} ms, settled {:.1f} ms",
                    ms_since_start(first_frame_at), ms_since_start(std::chrono::steady_clock::now())
                );
            }
            profile::ReportCpuPhase("Scene load CPU timings");
            profile::ClearStats();
            profile_cleared = true;
        }
        {
            const profile::CpuScope scope{"Frame"};
            if (bench_frames > 0 && settled) benchmark.Apply(r, viewport, extent);
            driver.EmitFrameActions(r, viewport, settled, extent);
            action::ApplyEmitted(r, viewport);
            ReportActionErrors(r);
            // Continue event processing while audio-only capture suppresses GPU rendering after initial framing.
            if (driver.AudioOnly() && driver.RecordingStarted) r.ctx().get<PendingRenderRequest>().Value = RenderRequest::None;
            SubmitViewport(r, viewport);
            WaitForRender(r);
            if (!submitted) first_frame_at = std::chrono::steady_clock::now();
            submitted = true;
        }
        if (bench_frames > 0) {
            // Render every ready benchmark tick and stop after the requested count.
            if (settled && --bench_frames == 0) {
                if (driver.ScreenshotMode()) {
                    if (auto saved = SaveScreenshot(r, driver.ScreenshotPath); saved) std::println("Saved screenshot: {}", saved->string());
                    else std::println(stderr, "Screenshot: {}", saved.error());
                }
                done = true;
            }
            auto &pending = r.ctx().get<PendingRenderRequest>().Value;
            pending = std::max(pending, RenderRequest::Reuse);
        } else {
            if (driver.CaptureFrame(r, viewport, driver.Framed(settled))) done = true;
            // End an idle headless run after one completed frame.
            if (!driver.Presenting() && settled) done = true;
        }
        driver.ElapsedPlayTime += frame_state.DeltaTime;
    }
    action::StopLog();
    return true;
}

// Run scenes offscreen on a fixed-step, GPU-paced clock.
void RunHeadlessEngine(bool quiet, auto &&scenes) {
    LogEnabled = !quiet;

    MacPlatform::InitPaths();

    entt::registry r;
    const auto &ctx = r.ctx().emplace<mtl::Context>();
    const auto viewport = InitEngine(r);
    profile::Init(ctx);
    InitAudioSystem(r);
    SetupScene(r, viewport);
    r.ctx().get<FrameState>().DisplayFramebufferScale = {2, 2}; // Match the app's retina rendering (pixel density and DPI-scaled GPU state like edge-line width).
    ProcessComponentEvents(r, viewport);

    scenes(r, viewport);

    WaitForRender(r);
    DeinitAudioSystem(r);
    profile::Report();
    profile::Deinit();
    DeinitViewport(r, viewport);
}

// Run one headless scene and return its capture or load status.
bool RunHeadless(const char *initial_file, bool quiet, bool empty, const CaptureRequest &capture) {
    bool ok = true;
    RunHeadlessEngine(quiet, [&](entt::registry &r, entt::entity viewport) {
        ok = RunHeadlessScene(r, viewport, initial_file, empty, capture);
    });
    return ok;
}

// Parse a corpus job containing "<output basename>\t<scene argument>".
struct RenderJob {
    fs::path OutBasename;
    std::string SceneArg;
};

// Atomically rename and return the next pending job.
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
        if (rename_ec) continue;
        std::ifstream in{claimed};
        std::string line;
        if (!std::getline(in, line)) continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        return RenderJob{line.substr(0, tab), line.substr(tab + 1)};
    }
    return std::nullopt;
}

// Render every queued job with one engine and write each scene's console output to its log.
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
            RunHeadlessScene(r, viewport, initial_file, empty, CaptureRequest{.RenderBasename = job->OutBasename, .Overlays = harness.Overlays, .EditMode = harness.EditMode, .SelectAll = harness.SelectAll, .LodErrorPixels = harness.LodErrorPixels, .NormalOverlays = harness.NormalOverlays, .BoundingBoxes = harness.BoundingBoxes, .TetWireframe = harness.TetWireframe});
            // Finalize capture and restore engine state between jobs.
            QuiesceScene(r, viewport);
            ClearScene(r, viewport);
            ProcessComponentEvents(r, viewport);
            std::fflush(stdout);
            std::fflush(stderr);
            ::dup2(launcher_out, STDOUT_FILENO);
            ::dup2(launcher_err, STDERR_FILENO);
            if (fs::exists(out + ".webp") || fs::exists(out + ".mp4")) std::println("ok   {}", out);
            else std::println("SKIP {} (no output; load failed or unsupported encoding)", out);
            std::fflush(stdout);
        }
        ::close(launcher_out);
        ::close(launcher_err);
    });
}
} // namespace

int main(int argc, char **argv) {
    const auto autorelease_pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

    // Convert ffmpeg pipe closure into EPIPE for VideoRecorder error handling.
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
        if (a == "--help" || a == "-h") {
            std::println("{}", Usage);
            return 0;
        }
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
        else if (a == "--edit" && std::next(it) != args.end()) {
            const std::string_view mode{*++it};
            if (mode == "vertex") capture.EditMode = Element::Vertex;
            else if (mode == "edge") capture.EditMode = Element::Edge;
            else if (mode == "face") capture.EditMode = Element::Face;
            else {
                std::println(stderr, "Unknown edit element '{}'.", mode);
                return 1;
            }
        } else if (a == "--lod-error" && std::next(it) != args.end()) {
            capture.LodErrorPixels = std::stof(std::string{*++it});
        } else if (a == "--select-all") capture.SelectAll = true;
        else if (a == "--display" && std::next(it) != args.end()) {
            const std::string_view names{*++it};
            for (size_t start = 0; start <= names.size();) {
                const auto end = std::min(names.find(',', start), names.size());
                const auto item = names.substr(start, end - start);
                start = end + 1;
                if (item == "vertex-normals") capture.NormalOverlays |= uint8_t(Element::Vertex);
                else if (item == "face-normals") capture.NormalOverlays |= uint8_t(Element::Face);
                else if (item == "bounds") capture.BoundingBoxes = true;
                else if (item == "tet-wireframe") capture.TetWireframe = true;
                else {
                    std::println(stderr, "Unknown display overlay '{}'.", item);
                    return 1;
                }
            }
        } else if (a == "--fps" && std::next(it) != args.end()) capture.Fps = std::atoi(*++it);
        else if (a == "--timeline-end" && std::next(it) != args.end()) capture.TimelineEnd = std::atof(*++it);
        else if (a == "--motion-blur" && std::next(it) != args.end()) capture.MotionBlurSteps = uint8_t(std::max(1, std::atoi(*++it)));
        else if (a == "--frames" && std::next(it) != args.end()) capture.BenchFrames = std::atoi(*++it);
        else if (a == "--bench-action" && std::next(it) != args.end()) {
            const std::string_view action{*++it};
            if (action == "steady") capture.BenchAction = CaptureRequest::BenchmarkAction::Steady;
            else if (action == "orbit") capture.BenchAction = CaptureRequest::BenchmarkAction::Orbit;
            else if (action == "transform") capture.BenchAction = CaptureRequest::BenchmarkAction::Transform;
            else if (action == "visibility") capture.BenchAction = CaptureRequest::BenchmarkAction::Visibility;
            else if (action == "box-select") capture.BenchAction = CaptureRequest::BenchmarkAction::BoxSelect;
            else {
                std::println(stderr, "Unknown benchmark action '{}'.", action);
                return 1;
            }
        } else if (a == "--bench-action-count" && std::next(it) != args.end()) {
            capture.BenchActionCount = uint32_t(std::max(1, std::atoi(*++it)));
        } else if (a == "--camera" && std::next(it) != args.end()) capture.CameraName = *++it;
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
        } else if (a.starts_with('-')) {
            std::println(stderr, "Unknown option '{}'. Run with --help for the option list.", a);
            return 1;
        } else if (!initial_file) initial_file = *it;
    }
    if (capture.Fps <= 0) capture.Fps = 60;
    // Enable audio capture for WAV output.
    if (capture.RecordPath.extension() == ".wav") capture.RecordAudio = true;
    // Derive corpus output paths from the render basename.
    if (!capture.RenderBasename.empty() && (!capture.RecordPath.empty() || !capture.ScreenshotPath.empty())) {
        std::println(stderr, "--render cannot be combined with --record or --screenshot");
        return 1;
    }
    // Derive queue capture options from each job.
    if (!render_queue.empty() && (initial_file || !capture.RenderBasename.empty() || !capture.RecordPath.empty() || !capture.ScreenshotPath.empty())) {
        std::println(stderr, "--render-queue cannot be combined with a scene file or capture flags");
        return 1;
    }

    bool headless_ok = true;
    if (!render_queue.empty()) RunHeadlessQueue(render_queue, quiet, capture);
    else if (headless) headless_ok = RunHeadless(initial_file, quiet, empty, capture);
    else run(initial_file, quiet, empty, capture);
    // Report failed captures and headless scene loads through the process status.
    return !headless_ok || VideoRecorder::AnyFailed() ? 1 : 0;
}
