#include "metal/AutoreleaseScope.h"
#include "viewport/ViewportSchema.h"

#include "Compress.h"

#include "FileDialog.h"
#include "MacPlatform.h"
#include "Paths.h"
#include "Profile.h"
#include "TransformMath.h"
#include "WorkspaceState.h"
#include "action/Build.h"
#include "action/Errors.h"
#include "audio/AudioDevice.h"
#include "audio/AudioOutputConfig.h"
#include "audio/AudioUi.h"
#include "audio/ModalModes.h"
#include "editor/AudioIntegration.h"
#include "editor/Engine.h"
#include "gizmo/GizmoInteraction.h"
#include "gltf/GltfScene.h"
#include "image/ImageEncode.h"
#include "mesh/MeshComponents.h"
#include "metal/MetalCpp.h"
#include "metal/PassChain.h"
#include "metal/RenderTarget.h"
#include "metal/Shader.h"
#include "procedure/NodeEditor.h"
#include "project/HistoryUi.h"
#include "project/Sessions.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "scene/CameraLens.h"
#include "scene/Entity.h"
#include "scene/SceneControlsUi.h"
#include "scene/WorldTransform.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "timeline/TimelineTheme.h"
#include "timeline/TimelineUi.h"
#include "ui/CtrlShortcut.h"
#include "ui/FieldEdit.h"
#include "ui/InterfaceTheme.h"
#include "ui/InterfaceUi.h"
#include "ui/MacBackend.h"
#include "viewport/FrameState.h"
#include "viewport/RenderExtent.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportComponents.h"
#include "viewport/ViewportConsumerFence.h"
#include "viewport/ViewportIcons.h"
#include "viewport/ViewportUi.h"

#include "imgui_impl_metal.h"
#include "implot.h"
#include "imspinner_demo.h"

#include <csignal>
#include <future>
#include <print>
#include <set>

using std::ranges::any_of, std::ranges::all_of;

namespace fs = std::filesystem;

namespace {
using SteadyClock = std::chrono::steady_clock;
double ElapsedMs(SteadyClock::time_point begin) { return std::chrono::duration<double, std::milli>(SteadyClock::now() - begin).count(); }

// Skip a frame when every drawable is in flight.
NS::SharedPtr<MTL::CommandBuffer> RenderAndPresentFrame(const mtl::Context &ctx, CA::MetalLayer *layer, ImDrawData *draw_data) {
    const mtl::AutoreleaseScope native_scope;
    const profile::CpuScope scope{"ImGuiRenderSubmit"};
    const auto *drawable = layer->nextDrawable();
    if (!drawable) return {};

    const std::array colors{mtl::ClearColor(drawable->texture(), {0.45, 0.55, 0.60, 1.0})};
    const auto pass = mtl::MakePassDescriptor(colors);
    auto *command_buffer = ctx.Queue->commandBuffer();
    ImGui_ImplMetal_NewFrame(pass.get());
    auto *encoder = command_buffer->renderCommandEncoder(pass.get());
    ImGui_ImplMetal_RenderDrawData(draw_data, command_buffer, encoder);
    encoder->endEncoding();
    {
        const profile::CpuScope present_scope{"Present"};
        command_buffer->presentDrawable(drawable);
    }
    mtl::Commit(command_buffer);
    return NS::RetainPtr(command_buffer);
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
    GltfSampleTrees t{
        .Examples = BuildGltfSampleTree(Paths::Res() / "examples"),
        .Benchmarks = BuildGltfSampleTree(Paths::Res() / "benchmarks"),
#ifdef GLTF_SAMPLE_ASSETS_DIR
        .SampleAssets = BuildGltfSampleTree(fs::path{GLTF_SAMPLE_ASSETS_DIR} / "Models"),
#endif
#ifdef GLTF_PHYSICS_DIR
        .Physics = BuildGltfSampleTree(GLTF_PHYSICS_DIR),
#endif
#ifdef GLTF_PHYSICAL_AUDIO_DIR
        .PhysicalAudio = BuildGltfSampleTree(GLTF_PHYSICAL_AUDIO_DIR),
#endif
    };
#ifdef GLTF_SAMPLE_ASSETS_DIR
    [&](this auto &&self, const GltfSampleTree &n) -> void {
        for (const auto &f : n.Files) t.SampleAssetsExtensions.insert_range(f.Extensions);
        for (const auto &[_, c] : n.Children) self(c);
    }(t.SampleAssets);
#endif
    return t;
}

std::future<GltfSampleTrees> SampleTreesFuture;

using project::Session;

// Applies the action, whose settle takes a view setup it asks for before the next one reads or overrides the view.
void Perform(state::Registry &r, auto action) {
    Session(r).Do(action::MakeAction(std::move(action)));
}

// Finish GPU work and stop playback before modifying scene structure.
void QuiesceScene(state::Registry &r) {
    WaitForRender(r);
    const auto &playback = r.get<const TimelinePlayback>(state::Document);
    if (playback.Playing) Perform(r, action::timeline::TogglePlay{playback.CurrentFrame});
}

constexpr std::string_view ProjectExt{".project"}, ActionsExt{".actions"};
fs::path HeadlessDirectory;
uint64_t RestoreGeneration{};
fs::path CachedWorkspacePath;
std::vector<std::byte> CachedWorkspaceBytes;

workspace::State CaptureWorkspace(state::Registry &r) {
    return workspace::Capture(r, r.Resources.get<const PanelsState>());
}

bool SaveWorkspace(state::Registry &r, bool force = true) {
    if (Paths::Project().empty()) return true;
    const auto path = Paths::Project() / workspace::FileName;
    auto bytes = workspace::Serialize(CaptureWorkspace(r));
    if (!force && path == CachedWorkspacePath && bytes == CachedWorkspaceBytes) return true;
    if (!workspace::Save(path, bytes)) {
        action::Fail(r, "Failed to save workspace state.");
        return false;
    }
    CachedWorkspacePath = path;
    CachedWorkspaceBytes = std::move(bytes);
    return true;
}

bool UiGestureSettled(const ImGuiWindow *frame_focus) {
    // Dock decorations can still show the old focus when menus or widgets change it during the frame.
    // A closed menu leaves focus on its undrawn popup for one frame until ImGui refocuses under it.
    const auto *focus = GImGui->NavWindow;
    if (focus != frame_focus || (focus && !focus->Active) || GImGui->ActiveId || GImGui->ActiveIdPreviousFrame || !GImGui->OpenPopupStack.empty() || GImGui->MovingWindow || GImGui->DragDropActive) return false;
    const auto &io = GetIO();
    for (int button = 0; button < ImGuiMouseButton_COUNT; ++button) {
        if (io.MouseDown[button] || io.MouseReleased[button]) return false;
    }
    return true;
}

fs::path NewWorkingDirectory() {
    if (HeadlessDirectory.empty()) return project::ReserveRestoreSession();
    File::TemporaryDirectory directory{HeadlessDirectory};
    return std::exchange(directory.Path, {});
}

bool NewScene(state::Registry &r, bool empty) {
    if (!SaveWorkspace(r)) return false;
    const auto dir = NewWorkingDirectory();
    if (dir.empty() || !Session(r).New(dir, empty)) return false;
    Paths::SetProject(dir);
    ++RestoreGeneration;
    return true;
}

struct RestoreTimings {
    double ResetMs{}, RestoreMs{}, RenderMs{}, CaptureMs{};
};

bool OpenProjectDir(state::Registry &r, const fs::path &working_dir, const fs::path &saved = {}) {
    auto &session = Session(r);
    if (session.History.Present && (!SaveWorkspace(r) || !session.Save())) return false;
    WaitForRender(r);
    const auto old_dir = Paths::Project(); // NOLINT(performance-unnecessary-copy-initialization) SetProject replaces the referenced path.
    Paths::SetProject(working_dir);
    if (!session.Open(working_dir, saved)) {
        Paths::SetProject(old_dir);
        return false;
    }
    ++RestoreGeneration;
    const auto stored = saved.empty() ? workspace::Load(working_dir / workspace::FileName) : workspace::Deserialize(session.RestoredWorkspace);
    if (stored) workspace::Apply(r, r.Resources.get<PanelsState>(), *stored);
    return true;
}

bool OpenProjectFile(state::Registry &r, const fs::path &path) {
    std::error_code ec;
    const bool directory = fs::is_directory(path, ec);
    if (directory && HeadlessDirectory.empty()) return OpenProjectDir(r, path / "working", path / "Saved.project");
    const auto working = NewWorkingDirectory();
    if (working.empty()) return false;
    bool copied;
    File::DirectoryLock source_lock;
    if (directory) {
        source_lock = File::DirectoryLock{path / "working"};
        if (source_lock) fs::copy(path / "working", working, fs::copy_options::recursive, ec);
        copied = source_lock && !ec;
    } else {
        copied = Decompress(path, working);
        if (copied && path.extension() == ActionsExt) {
            const auto workspace_bytes = ReadArchiveMetadata(path);
            copied = workspace_bytes && workspace::Deserialize(*workspace_bytes) && workspace::Save(working / workspace::FileName, *workspace_bytes);
        }
    }
    if (!copied || !OpenProjectDir(r, working, directory ? path / "Saved.project" : fs::path{})) {
        if (Paths::Project() != working) fs::remove_all(working, ec);
        action::Fail(r, std::format("Failed to open project '{}'.", path.string()));
        return false;
    }
    if (directory) Session(r).SavedPath.clear();
    return true;
}

void OpenFile(state::Registry &r, const fs::path &path) {
    std::error_code ec;
    if (const auto ext = path.extension(); fs::is_directory(path, ec) || ext == ProjectExt || ext == ActionsExt) OpenProjectFile(r, path);
    else action::Emit(action::io::Load{.Path = path});
}

bool SaveProjectFile(state::Registry &r, const fs::path &archive_path) {
    const auto form = archive_path.extension() == ActionsExt ? project::ArchiveForm::Actions : project::ArchiveForm::Project;
    return SaveWorkspace(r) && Session(r).SaveArchive(archive_path, form, CachedWorkspaceBytes);
}

void SaveProjectAs(state::Registry &r) {
    const auto &saved = Session(r).SavedPath;
    FileDialog::ShowSave(nullptr, saved.empty() ? fs::path{"Untitled"} : saved.parent_path(), [&r](const fs::path &directory) {
        if (SaveWorkspace(r) && Session(r).SaveAs(directory, CachedWorkspaceBytes)) Paths::SetProject(Session(r).History.Dir);
    });
}

void SaveProject(state::Registry &r) {
    if (Session(r).SavedPath.empty()) SaveProjectAs(r);
    else SaveProjectFile(r, Session(r).SavedPath);
}

void RevertSavedProject(state::Registry &r) {
    if (!Session(r).RevertSaved()) return;
    if (const auto stored = workspace::Deserialize(Session(r).RestoredWorkspace)) workspace::Apply(r, r.Resources.get<PanelsState>(), *stored);
    ++RestoreGeneration;
}

void BuildDefaultDockLayout(const PanelsState &panels, ImGuiID dockspace_id) {
    auto controls_node_id = DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.3f, nullptr, &dockspace_id);
    auto extra_node_id = DockBuilderSplitNode(controls_node_id, ImGuiDir_Down, 0.4f, nullptr, &controls_node_id);
    const auto info_node_id = DockBuilderSplitNode(extra_node_id, ImGuiDir_Down, 0.25f, nullptr, &extra_node_id);
    auto nodes_node_id = DockBuilderSplitNode(dockspace_id, ImGuiDir_Down, 0.3f, nullptr, &dockspace_id);
    const auto timeline_node_id = DockBuilderSplitNode(nodes_node_id, ImGuiDir_Down, 0.35f, nullptr, &nodes_node_id);
    const auto parameters_node_id = DockBuilderSplitNode(nodes_node_id, ImGuiDir_Right, 0.25f, nullptr, &nodes_node_id);
    DockBuilderDockWindow(panels.History.Name, extra_node_id);
    DockBuilderDockWindow(panels.Style.Name, extra_node_id);
    DockBuilderDockWindow(panels.Debug.Name, extra_node_id);
    DockBuilderDockWindow(panels.ImGuiDemo.Name, extra_node_id);
    DockBuilderDockWindow(panels.ImSpinnerDemo.Name, extra_node_id);
    DockBuilderDockWindow(panels.ImPlotDemo.Name, extra_node_id);
    DockBuilderDockWindow(panels.SceneControls.Name, controls_node_id);
    DockBuilderDockWindow(panels.Timeline.Name, timeline_node_id);
    DockBuilderDockWindow(panels.Nodes.Name, nodes_node_id);
    DockBuilderDockWindow(panels.Parameters.Name, parameters_node_id);
    DockBuilderDockWindow(panels.Info.Name, info_node_id);
    DockBuilderGetNode(extra_node_id)->SelectedTabId = ImHashStr(panels.History.Name);
}

void RenderDebugPanel(state::Registry &r, const mtl::Context &ctx, CA::MetalLayer *layer, PanelsState &panels, const ImGuiIO &io) {
    if (!panels.Debug.Visible) return;
    if (Begin(panels.Debug.Name, &panels.Debug.Visible)) {
        if (BeginTabBar("Debug")) {
            if (BeginTabItem("ImGui")) {
                Text("Dear ImGui %s (%d)", IMGUI_VERSION, IMGUI_VERSION_NUM);
                Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
                Text("%d vertices, %d indices (%d triangles)", io.MetricsRenderVertices, io.MetricsRenderIndices, io.MetricsRenderIndices / 3);
                const auto &g = *GImGui;
                Text("%d visible windows, %d current allocations", io.MetricsRenderWindows, g.DebugAllocInfo.TotalAllocCount - g.DebugAllocInfo.TotalFreeCount);
                Separator();
                Text("See [Panels->%s] for more details.", panels.ImGuiDemo.Name);
                EndTabItem();
            }
            if (BeginTabItem("Metal")) {
                SeparatorText("Device");
                Text("Name: %s", ctx.Device->name()->utf8String());
                Text("Unified memory: %s", ctx.Device->hasUnifiedMemory() ? "yes" : "no");
                Text("Argument buffer tier: %ld", long(ctx.Device->argumentBuffersSupport()) + 1);
                Text("BC texture compression: %s", ctx.Device->supportsBCTextureCompression() ? "yes" : "no");
                Text("Max buffer length: %llu MB", (unsigned long long)(ctx.Device->maxBufferLength() / (1024 * 1024)));
                if (layer) {
                    SeparatorText("Window surface");
                    const auto drawable_size = layer->drawableSize();
                    Text("Drawable: %.0fx%.0f", drawable_size.width, drawable_size.height);
                    Text("Display sync: %s", layer->displaySyncEnabled() ? "on" : "off");
                }
                EndTabItem();
            }
            if (BeginTabItem("Engine")) {
                SeparatorText("Buffer memory");
                TextUnformatted(DebugBufferHeapUsage(r).c_str());
                SeparatorText("Action");
                Text("sizeof(Action): %zu bytes", sizeof(action::Action));
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

void DrawStylePanel(state::Registry &r, Panel &panel) {
    if (!panel.Visible) return;
    if (Begin(panel.Name, &panel.Visible) && BeginTabBar("Style")) {
        if (BeginTabItem("Viewport")) {
            DrawViewportStyle(r);
            EndTabItem();
        }
        if (BeginTabItem("Nodes")) {
            procedure::DrawNodeStyle(r);
            EndTabItem();
        }
        if (BeginTabItem("Timeline")) {
            DrawTimelineStyle(r);
            EndTabItem();
        }
        if (BeginTabItem("Interface")) {
            ui::DrawInterfaceStyle(r);
            EndTabItem();
        }
        EndTabBar();
    }
    End();
}

// A whole number prints without decimals, any other value in its shortest form, and both carry the unit symbol.
void FormatBound(std::span<char> out, double value, FieldUnit unit) {
    snprintf(out.data(), out.size(), value == std::floor(value) ? "%.0f%s" : "%g%s", value, UnitSymbol(unit).data());
}

void DrawFieldTypeLine(const char *type, const FieldSpec *spec) {
    if (!spec || !spec->Bounded()) {
        Text("Type: %s", type);
        return;
    }
    const bool radians = spec->Unit == FieldUnit::Radians;
    const auto bound = [radians](double value) { return radians ? double(numeric::Degrees(float(value))) : value; };
    char lo[32] = "-inf", hi[32] = "inf";
    if (spec->HasMin()) FormatBound(lo, bound(spec->Min), spec->Unit);
    if (spec->HasMax()) FormatBound(hi, bound(spec->Max), spec->Unit);
    Text("Type: %s | Range: %c%s, %s%c", type, spec->HasMin() ? '[' : '(', lo, hi, spec->HasMax() ? ']' : ')');
}

// Draws after every other panel, so it describes the item hovered in this frame.
void DrawInfoPanel(state::Registry &r, Panel &panel) {
    const auto info = std::exchange(ui::Hovered, {});
    if (!panel.Visible) return;
    if (Begin(panel.Name, &panel.Visible)) {
        if (!info.Label) {
            TextDisabled("Hover a field for its description.");
        } else {
            TextUnformatted(info.Label);
            if (info.Path) TextDisabled("%s", info.Path);
            if (info.Type) DrawFieldTypeLine(info.Type, info.Spec);
            if (info.Help) TextWrapped("%s%s", info.Help, std::string_view{info.Help}.ends_with('.') ? "" : ".");
            if (info.Status) TextUnformatted(info.Status);
            if (info.Selection && SelectedIn(r).size() > 1) TextUnformatted("Alt applies the edit to every selected object");
            if (info.Key && info.Key->HasChannel) {
                Text("%s%s", info.Key->KeyAtFrame ? "Keyed on this frame" : "Animated", info.Key->Changed ? ", changed since the key" : "");
            }
        }
    }
    End();
}

// The viewport's panel shows the viewport's name under an ImGui ID its entity keeps.
std::string ViewportPanelName(const state::Registry &r, state::Entity viewport) {
    return std::format("{}###Viewport{}", GetName(r, viewport), state::Index(viewport));
}

// The viewport panel under the mouse becomes the active viewport while no gesture, button, or popup holds the input.
void FollowHoveredViewport(state::Registry &r) {
    const auto &frame = r.Resources.get<const FrameState>();
    if (frame.MeshDrag || frame.BoxDrag || r.get<const GizmoInteraction>(state::Document).IsUsing()) return;
    if (IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) return;
    const auto &io = GetIO();
    for (int button = 0; button < ImGuiMouseButton_COUNT; ++button) {
        if (io.MouseDown[button] || io.MouseReleased[button]) return;
    }
    const auto *hovered = GImGui->HoveredWindow;
    if (!hovered) return;
    const auto viewports = r.view<const Viewport>();
    const auto it = std::ranges::find(viewports, hovered->RootWindow->ID, [&](state::Entity viewport) { return ImHashStr(ViewportPanelName(r, viewport).c_str()); });
    if (it != viewports.end()) r.replace_if_changed(state::Document, ActiveViewport{*it});
}

// A viewport panel open this frame, with its content size.
struct OpenPanel {
    state::Entity Entity;
    uvec2 Extent;
};

struct EditorPanelsFrame {
    std::vector<OpenPanel> Viewports; // In entity index order.
};

EditorPanelsFrame BeginEditorPanels(
    state::Registry &r, const mtl::Context &ctx, CA::MetalLayer *layer,
    PanelsState &panels, const ImGuiIO &io, ImGuiID dockspace_id, bool interactive
) {
    RenderDebugPanel(r, ctx, layer, panels, io);
    if (project::DrawHistoryPanel(Session(r), panels.History, interactive) && SaveWorkspace(r)) Session(r).ClearHistory();
    DrawStylePanel(r, panels.Style);
    if (panels.ImGuiDemo.Visible) ShowDemoWindow(&panels.ImGuiDemo.Visible);
    if (panels.ImSpinnerDemo.Visible) {
        if (Begin(panels.ImSpinnerDemo.Name, &panels.ImSpinnerDemo.Visible)) ImSpinner::demoSpinners();
        End();
    }
    if (panels.ImPlotDemo.Visible) ImPlot::ShowDemoWindow(&panels.ImPlotDemo.Visible);
    if (panels.SceneControls.Visible) {
        if (Begin(panels.SceneControls.Name, &panels.SceneControls.Visible)) RenderControls(r);
        End();
    }

    bool scrubbing = false;
    if (panels.Timeline.Visible) {
        PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        if (Begin(panels.Timeline.Name, &panels.Timeline.Visible, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
            PushStyleVar(ImGuiStyleVar_FramePadding, {6, 4});
            Indent(6);
            Spacing();
            Unindent(6);
            PopStyleVar();
            const auto &keyframes = r.get<const SelectedKeyframes>(state::Document).Frames;
            if (auto action = RenderTimeline(
                    r.get<const TimelineRange>(state::Document), r.get<const TimelinePlayback>(state::Document),
                    r.get<const TimelineView>(state::Document), r, r.get<const Animations>(state::Document),
                    keyframes, r.Resources.get<const ViewportIcons>().Timeline, r.get<const InterfaceTheme>(state::Document), r.get<const TimelineTheme>(state::Document), scrubbing
                );
                interactive && action) {
                action::Emit(std::move(*action));
            }
        }
        End();
        PopStyleVar();
    }
    procedure::DrawNodesPanel(r, panels.Nodes, interactive);
    procedure::DrawParametersPanel(r, panels.Nodes, panels.Parameters, interactive);
    if (interactive) r.Resources.get<FrameState>().Scrubbing = scrubbing;

    if (interactive) FollowHoveredViewport(r);
    auto &frame_state = r.Resources.get<FrameState>();
    const auto viewports = r.view<const Viewport>();
    const auto active = FindActiveViewport(r);
    const bool closable = viewports.size() > 1;
    EditorPanelsFrame frame;
    for (const auto viewport : viewports) {
        // A panel in an unselected dock tab draws nothing, so its viewport does no work this frame.
        r.edit<Viewport>(viewport).Shown = false;
        if (!r.get<const ViewportPanel>(viewport).Open) continue;
        const auto name = ViewportPanelName(r, viewport);
        // A panel ImGui has never seen opens as a tab beside the active viewport's panel, or in the central dock node before that panel exists.
        if (const auto id = ImHashStr(name.c_str()); !FindWindowByID(id) && !FindWindowSettingsByID(id)) {
            const auto *beside = viewport == active ? nullptr : FindWindowByName(ViewportPanelName(r, active).c_str());
            const auto *central = DockBuilderGetCentralNode(dockspace_id);
            SetNextWindowDockID(beside ? beside->DockId : central ? central->ID :
                                                                    0);
        }
        PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        bool open = true;
        if (Begin(name.c_str(), closable ? &open : nullptr)) {
            r.edit<Viewport>(viewport).Shown = true;
            if (interactive && viewport == active) Interact(r, frame_state);
            auto &draw_list = *GetWindowDrawList();
            draw_list.ChannelsSplit(2);
            draw_list.ChannelsSetCurrent(1);
            InteractOverlay(r, frame_state, viewport, viewport == active);
            if (frame.Viewports.empty()) DrawModalJobsOverlay(r);
            const auto content_region = GetContentRegionAvail();
            frame.Viewports.push_back({viewport, {uint32_t(std::max(content_region.x, 0.f)), uint32_t(std::max(content_region.y, 0.f))}});
        }
        End();
        PopStyleVar();
        if (!open) action::Emit(action::view::RemoveViewport{viewport});
    }
    return frame;
}

// Inserts each open viewport panel's image beneath the overlay its panel drew, after the frame's renders are submitted.
void DisplayViewportPanels(state::Registry &r, const EditorPanelsFrame &frame) {
    const auto active = FindActiveViewport(r);
    for (const auto &panel : frame.Viewports) {
        if (!r.all_of<Viewport>(panel.Entity)) continue;
        Begin(ViewportPanelName(r, panel.Entity).c_str());
        DisplayViewport(r, panel.Entity, panel.Entity == active);
        GetWindowDrawList()->ChannelsMerge();
        End();
    }
}

#ifdef DEBUG_BUILD

struct ValidationImage {
    mtl::Texture Target;
    NS::SharedPtr<MTL::Event> Ready;
    uint64_t Generation{};
};

// A hash of every vertex, index and command the draw data holds, equal across frames whose UI is identical.
uint64_t DrawDataHash(const ImDrawData *draw_data) {
    uint64_t hash = 0;
    const auto mix = [&](const void *data, size_t bytes) { hash = hash * 0x9E3779B97F4A7C15ull + std::hash<std::string_view>{}({static_cast<const char *>(data), bytes}); };
    for (const auto *list : draw_data->CmdLists) {
        mix(list->VtxBuffer.Data, size_t(list->VtxBuffer.Size) * sizeof(ImDrawVert));
        mix(list->IdxBuffer.Data, size_t(list->IdxBuffer.Size) * sizeof(ImDrawIdx));
        for (const auto &cmd : list->CmdBuffer) {
            mix(&cmd.ClipRect, sizeof(cmd.ClipRect));
            mix(&cmd.ElemCount, sizeof(cmd.ElemCount));
            mix(&cmd.TexRef, sizeof(cmd.TexRef));
        }
    }
    return hash;
}

void WriteValidationProject(const store::History &history, std::span<const std::byte> workspace) {
    const auto path = fs::temp_directory_path() / "MeshEditor-validation.actions";
    if (project::WriteActionsArchive(history, path, workspace)) std::println(stderr, "[validation] wrote action replay archive to {}", path.string());
}

fs::path WriteValidationImage(const mtl::Context &ctx, std::string_view name, const mtl::Texture &image) {
    auto rgba = ReadbackImageRgba8(ctx, image, 0, 0, image.Extent);
    for (size_t i = 0; i < rgba.size(); i += 4) std::swap(rgba[i], rgba[i + 2]);
    const auto encoded = EncodeImagePngRgba8(rgba, image.Extent.Width, image.Extent.Height, name);
    if (!encoded) return {};
    const auto path = fs::temp_directory_path() / std::format("MeshEditor-validation-{}.png", name);
    std::ofstream out{path, std::ios::binary};
    out.write(reinterpret_cast<const char *>(encoded->data()), std::streamsize(encoded->size()));
    return out ? path : fs::path{};
}

void RenderAppImage(const mtl::Context &ctx, ImDrawData *draw_data, ValidationImage &image) {
    const mtl::AutoreleaseScope native_scope;
    const mtl::Extent2D extent{
        uint32_t(std::ceil(draw_data->DisplaySize.x * draw_data->FramebufferScale.x)),
        uint32_t(std::ceil(draw_data->DisplaySize.y * draw_data->FramebufferScale.y)),
    };
    if (!image.Ready) image.Ready = NS::TransferPtr(ctx.Device->newEvent());
    if (!image.Target || image.Target.Extent != extent) {
        image.Target = mtl::CreateUntrackedTexture2D(
            ctx, mtl::Format::Color, extent,
            MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead, MTL::StorageModePrivate
        );
    }
    const std::array colors{mtl::ClearColor(*image.Target, {0.45, 0.55, 0.60, 1.0})};
    const auto pass = mtl::MakePassDescriptor(colors);
    ImGui_ImplMetal_NewFrame(pass.get());
    auto *command_buffer = ctx.Queue->commandBuffer();
    {
        mtl::PassChain chain{command_buffer};
        auto *encoder = chain.BeginRender(pass.get(), "ValidationApp");
        ImGui_ImplMetal_RenderDrawData(draw_data, command_buffer, encoder);
    }
    command_buffer->encodeSignalEvent(image.Ready.get(), ++image.Generation);
    command_buffer->addCompletedHandler([](MTL::CommandBuffer *completed) {
        const mtl::AutoreleaseScope native_scope;
        if (const auto *error = completed->error()) {
            std::println(stderr, "[validation] app render failed: {}", error->localizedDescription()->utf8String());
            std::abort();
        }
    });
    mtl::Commit(command_buffer);
}

struct ValidationUi {
    ImGuiContext *LiveImGui{ImGui::GetCurrentContext()}, *Gui{};
    ImPlotContext *LiveImPlot{ImPlot::GetCurrentContext()}, *Plot{};

    ValidationUi(const mtl::Context &ctx) {
        auto *fonts = LiveImGui ? GetIO().Fonts : nullptr;
        const auto font_scale = LiveImGui ? GetIO().FontGlobalScale : 1.f;
        Gui = ImGui::CreateContext(fonts);
        ImGui::SetCurrentContext(Gui);
        Plot = ImPlot::CreateContext(); // NOLINT(cppcoreguidelines-prefer-member-initializer) Created under the current ImGui context.
        ImPlot::SetCurrentContext(Plot);
        auto &io = GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
        io.IniFilename = nullptr;
        io.FontGlobalScale = font_scale;
        ui::StyleColors();
        ImGui_ImplMetal_Init(ctx.Device.get());
    }
    void ActivateLive() const {
        ImGui::SetCurrentContext(LiveImGui);
        ImPlot::SetCurrentContext(LiveImPlot);
    }
    ~ValidationUi() {
        ImGui::SetCurrentContext(Gui);
        ImPlot::SetCurrentContext(Plot);
        ImGui_ImplMetal_Shutdown();
        ImPlot::DestroyContext(Plot);
        ImGui::DestroyContext(Gui);
        ActivateLive();
    }
};

void DrawMainMenuBar(state::Registry &r, PanelsState &panels) {
    if (BeginMainMenuBar()) {
        if (BeginMenu("File")) {
            if (BeginMenu("New")) {
                if (MenuItem("Default")) NewScene(r, /*empty=*/false);
                if (MenuItem("Empty")) NewScene(r, /*empty=*/true);
                EndMenu();
            }
            if (MenuItem("Open")) {
                FileDialog::ShowOpen("project;actions", [&](const fs::path &path) { OpenFile(r, path); }, /*directories=*/true);
            }
            if (MenuItem("Save", "Cmd+S")) SaveProject(r);
            if (MenuItem("Save as...", "Cmd+Shift+S")) SaveProjectAs(r);
            if (MenuItem("Revert to Saved", nullptr, false, !Session(r).SavedPath.empty())) RevertSavedProject(r);
            if (BeginMenu("Restore")) {
                const auto sessions = project::ListRestoreSessions();
                for (const auto &session : sessions) {
                    const std::time_t t = session.UnixSeconds;
                    char date[32];
                    std::strftime(date, sizeof date, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
                    const auto label = std::format("{} ({})", date, session.Path.filename().string());
                    if (MenuItem(label.c_str())) OpenProjectDir(r, session.Path);
                }
                EndMenu();
            }
            const auto import_dialog = [](const char *extensions) {
                FileDialog::ShowOpen(extensions, [](const fs::path &path) { action::Emit(action::io::Load{.Path = path}); });
            };
            if (BeginMenu("Import")) {
                if (MenuItem("glTF 2.0 (.glb/.gltf)")) import_dialog("gltf;glb");
                if (MenuItem("Wavefront (.obj)")) import_dialog("obj");
                if (MenuItem("Stanford PLY (.ply)")) import_dialog("ply");
                if (MenuItem("RealImpact")) {
                    FileDialog::ShowPickFolder([](const fs::path &path) { action::Emit(action::io::LoadRealImpact{.Path = path}); });
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
            EndMenu();
        }
        if (BeginMenu("Edit")) {
            const auto &history = Session(r).History;
            const auto undo_target = history.UndoTarget(), redo_target = history.RedoTarget();
            if (MenuItem("Undo", "Cmd+Z", false, undo_target.has_value())) Session(r).RequestNavigate(*undo_target);
            if (MenuItem("Redo", "Cmd+Shift+Z", false, redo_target.has_value())) Session(r).RequestNavigate(*redo_target);
            EndMenu();
        }
        if (BeginMenu("Panels")) {
            MenuItem(panels.History.Name, nullptr, &panels.History.Visible);
            MenuItem(panels.Nodes.Name, nullptr, &panels.Nodes.Visible);
            MenuItem(panels.Parameters.Name, nullptr, &panels.Parameters.Visible);
            MenuItem(panels.Info.Name, nullptr, &panels.Info.Visible);
            MenuItem(panels.Style.Name, nullptr, &panels.Style.Visible);
            MenuItem(panels.Debug.Name, nullptr, &panels.Debug.Visible);
            MenuItem(panels.ImGuiDemo.Name, nullptr, &panels.ImGuiDemo.Visible);
            MenuItem(panels.ImSpinnerDemo.Name, nullptr, &panels.ImSpinnerDemo.Visible);
            MenuItem(panels.ImPlotDemo.Name, nullptr, &panels.ImPlotDemo.Visible);
            MenuItem(panels.SceneControls.Name, nullptr, &panels.SceneControls.Visible);
            MenuItem(panels.Timeline.Name, nullptr, &panels.Timeline.Visible);
            Separator();
            for (const auto viewport : r.view<const Viewport>()) {
                if (bool open = r.get<const ViewportPanel>(viewport).Open; MenuItem(ViewportPanelName(r, viewport).c_str(), nullptr, &open)) {
                    r.patch<ViewportPanel>(viewport, [open](ViewportPanel &panel) { panel.Open = open; });
                }
            }
            if (MenuItem("New Viewport")) action::Emit(action::view::AddViewport{});
            EndMenu();
        }
        EndMainMenuBar();
    }
}

EditorPanelsFrame BeginEditorFrame(state::Registry &r, const mtl::Context &ctx, CA::MetalLayer *layer, PanelsState &panels, const ImGuiIO &io, bool interactive) {
    const auto dockspace_id = DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_AutoHideTabBar);
    if (!panels.LayoutLoaded) {
        BuildDefaultDockLayout(panels, dockspace_id);
        panels.LayoutLoaded = true;
    }
    DrawMainMenuBar(r, panels);
    if (interactive) {
        project::HandleHistoryShortcuts(Session(r));
        if (!GetIO().WantTextInput) {
            if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiKey_S, ImGuiInputFlags_RouteGlobal)) SaveProject(r);
            if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_S, ImGuiInputFlags_RouteGlobal)) SaveProjectAs(r);
        }
        if (auto action = HandleTimelineShortcuts(r.get<const TimelinePlayback>(state::Document))) std::visit([](auto leaf) { action::Emit(leaf); }, std::move(*action));
    }
    return BeginEditorPanels(r, ctx, layer, panels, io, dockspace_id, interactive);
}

void EndEditorFrame(state::Registry &r, PanelsState &panels, const EditorPanelsFrame &frame) {
    DisplayViewportPanels(r, frame);
    DrawInfoPanel(r, panels.Info);
    Render();
    workspace::ApplyPending(panels);
}

struct ValidationEngine {
    std::optional<Engine> Core;
    const fs::path Directory = fs::temp_directory_path() / std::format("MeshEditor-validation-{}", uintptr_t(this));
    ValidationImage App;
    std::optional<ValidationUi> Ui;

    ValidationEngine() {
        Core.emplace(true);
        InitViewportMedia(Core->R);
    }
    ~ValidationEngine() {
        WaitForRender(Core->R);
        Ui.reset();
        App = {};
        DeinitViewportMedia(Core->R);
        Core.reset();
        std::error_code ec;
        fs::remove_all(Directory, ec);
    }
};

} // namespace
// One offscreen engine restores by replay and then cold from the stored state, each compared with the live session before the next.
struct ValidationSession {
    ValidationEngine Restored;
    ValidationImage Live;
    mtl::ComputePipeline Compare;
    NS::SharedPtr<MTL::Buffer> Differences;

    ValidationSession(state::Registry &r)
        : Compare(r.Resources.get<mtl::LibraryCache>(), {"ValidationCompare.metal", "CompareValidationImages"}),
          Differences(mtl::NewBuffer(r.Resources.get<const mtl::Context>(), 2 * sizeof(uint32_t))) {}
};
namespace {

struct ValidationInputs {
    fs::path WorkingDir;

    workspace::State Workspace;
    AudioOutputConfig AudioOutput;
    AudioDeviceInfo Device;
    CA::MetalLayer *Layer{};
    ImVec2 DisplaySize{}, FramebufferScale{}, MousePos{};
    ImGuiID HoveredIdPreviousFrame{};
    float HoveredIdTimer{}, HoveredIdNotActiveTimer{};
    std::string FocusedWindow;
    ImGuiID NavId{};
    bool NavCursorVisible{}, NavHighlightItemUnderNav{};
    bool Capturing{}, Scrubbing{};
    float PlaybackFrame{};
    TimelinePlayback Playback{};
};

ImDrawData *RenderValidationApp(
    state::Registry &r, const ValidationInputs &inputs
) {
    const auto &ctx = r.Resources.get<const mtl::Context>();
    auto &io = GetIO();
    auto &panels = r.Resources.get<PanelsState>();
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
            GImGui->NavId = inputs.NavId;
            GImGui->NavCursorVisible = inputs.NavCursorVisible;
            GImGui->NavHighlightItemUnderNav = inputs.NavHighlightItemUnderNav;
        }

        EndEditorFrame(r, panels, BeginEditorFrame(r, ctx, inputs.Layer, panels, io, /*interactive=*/false));
    };

    render_frame(); // Instantiate the panels referenced by the restored dock layout.
    render_frame(); // Bind those panels to their dock nodes.
    render_frame(); // Settle docked child sizes and scrollbars.
    // Restore focus last so dock binding and the menu bar cannot move it again.
    SetWindowFocus(inputs.FocusedWindow.empty() ? nullptr : inputs.FocusedWindow.c_str());
    io.MousePos = inputs.MousePos;
    std::optional<uint64_t> previous;
    for (uint32_t frame = 0;; ++frame) {
        render_frame(/*restore_hover=*/true);
        const auto hash = DrawDataHash(GetDrawData());
        if (previous == hash) break;
        if (frame == 15) {
            std::println(stderr, "[validation] restored UI did not settle");
            std::abort();
        }
        previous = hash;
    }
    if (const std::string_view focused = GImGui->NavWindow ? GImGui->NavWindow->Name : ""; focused != inputs.FocusedWindow) {
        std::println(stderr, "[validation] focus DIVERGED (live '{}' / restored '{}')", inputs.FocusedWindow, focused);
        std::abort();
    }
    return GetDrawData();
}

struct ValidationResult {
    std::vector<std::byte> State, Workspace;
    RestoreTimings Timings;
};

ValidationResult RestoreForValidation(
    ValidationEngine &engine, const ValidationInputs &inputs, const store::History &live, bool replay
) {
    auto &restored = engine.Core->R;
    const auto &ctx = restored.Resources.get<const mtl::Context>();
    ValidationResult result;
    auto &timings = result.Timings;
    auto begin = SteadyClock::now();
    engine.Ui.emplace(ctx);
    restored.Resources.get<PanelsState>() = {};
    auto &frame = restored.Resources.get<FrameState>();
    frame = {.DisplayFramebufferScale = std::bit_cast<vec2>(inputs.FramebufferScale), .Scrubbing = inputs.Scrubbing, .Capturing = inputs.Capturing};
    engine.Core->P->Close();
    fs::create_directories(engine.Directory);
    std::error_code asset_ec;
    fs::remove(engine.Directory / "assets", asset_ec);
    if (fs::is_directory(inputs.WorkingDir / "assets")) fs::create_directory_symlink(inputs.WorkingDir / "assets", engine.Directory / "assets");
    for (const auto *name : {"leaves.log", "nodes.log", "tree.log"}) fs::copy_file(inputs.WorkingDir / name, engine.Directory / name, fs::copy_options::overwrite_existing);
    const auto reset_ms = ElapsedMs(begin);

    begin = SteadyClock::now();
    if (!engine.Core->P->Open(engine.Directory) || (replay && !engine.Core->P->Replay())) {
        for (const auto &error : restored.Resources.get<action::Errors>().Messages) std::println(stderr, "[validation] {}", error);
        std::println(stderr, "[validation] {}", engine.Core->P->History.TakeIntegrityError());
        WriteValidationProject(live, workspace::Serialize(inputs.Workspace));
        std::abort();
    }
    timings.RestoreMs = ElapsedMs(begin);
    workspace::Apply(restored, restored.Resources.get<PanelsState>(), inputs.Workspace);
    restored.Resources.get<AudioOutputConfig>() = inputs.AudioOutput;
    restored.Resources.get<AudioDeviceInfo>() = inputs.Device;
    frame.Scene = ActingScene(restored);
    begin = SteadyClock::now();
    restored.replace<TimelinePlayback>(state::Document, inputs.Playback);
    restored.edit<PlaybackFrame>(state::Document).Value = inputs.PlaybackFrame;
    for (const auto viewport : restored.view<const Viewport>()) {
        if (Displayed(restored, viewport)) PresentViewport(restored, viewport);
    }
    timings.ResetMs += reset_ms;
    RenderAppImage(ctx, RenderValidationApp(restored, inputs), engine.App);
    timings.RenderMs += ElapsedMs(begin);
    begin = SteadyClock::now();
    result.State = engine.Core->P->History.MaterializeLive();
    result.Workspace = workspace::Serialize(CaptureWorkspace(restored));
    timings.CaptureMs = ElapsedMs(begin);
    engine.Ui->ActivateLive();
    return result;
}

void RequireEqual(const store::History &live, std::string_view what, std::span<const std::byte> expected, std::span<const std::byte> actual, std::span<const std::byte> workspace) {
    if (!std::ranges::equal(expected, actual)) {
        std::println(
            stderr, "[validation] {} DIVERGED at byte {} (expected {} / actual {})",
            what, std::ranges::mismatch(expected, actual).in1 - expected.begin(), expected.size(), actual.size()
        );
        WriteValidationProject(live, workspace);
        std::abort();
    }
}

// Compares the restored engine's app and viewport images with the live ones, naming a divergence by `restore`.
void CompareValidationImages(state::Registry &r, ValidationSession &session, std::string_view restore) {
    const mtl::AutoreleaseScope native_scope;
    const auto &ctx = r.Resources.get<const mtl::Context>();
    const std::array<const mtl::Texture *, 2> expected{&session.Live.Target, &r.get<const Viewport>(FirstViewport(r)).Targets.Screen->FinalColorImage};
    const std::array<const mtl::Texture *, 2> restored{&session.Restored.App.Target, &session.Restored.Core->R.get<const Viewport>(FirstViewport(session.Restored.Core->R)).Targets.Screen->FinalColorImage};
    const std::array names{std::format("{}-app", restore), std::format("{}-viewport", restore)};
    for (uint32_t i = 0; i < restored.size(); ++i) {
        if (restored[i]->Extent != expected[i]->Extent) {
            const auto actual = restored[i]->Extent, extent = expected[i]->Extent;
            std::println(stderr, "[validation] {} extent DIVERGED ({}x{} / {}x{})", names[i], extent.Width, extent.Height, actual.Width, actual.Height);
            std::abort();
        }
    }
    auto *differences = static_cast<uint32_t *>(session.Differences->contents());
    std::fill_n(differences, restored.size(), UINT32_MAX);
    auto *command_buffer = ctx.Queue->commandBuffer();
    // Each app event follows its engine's viewport render; queues may finish in any order.
    for (const auto *image : {&session.Live, &session.Restored.App}) command_buffer->encodeWait(image->Ready.get(), image->Generation);
    {
        mtl::PassChain chain{command_buffer};
        auto *encoder = chain.BeginCompute("ValidationCompare");
        encoder->setComputePipelineState(session.Compare.State());
        for (uint32_t i = 0; i < restored.size(); ++i) {
            encoder->setTexture(**expected[i], 0);
            encoder->setTexture(**restored[i], 1);
            encoder->setBuffer(session.Differences.get(), i * sizeof(uint32_t), 0);
            const auto extent = expected[i]->Extent;
            encoder->dispatchThreads(MTL::Size(extent.Width, extent.Height, 1), MTL::Size(16, 16, 1));
        }
    }
    mtl::Commit(command_buffer);
    command_buffer->waitUntilCompleted();
    if (const auto *error = command_buffer->error()) throw std::runtime_error(error->localizedDescription()->utf8String());
    for (uint32_t i = 0; i < restored.size(); ++i) {
        if (differences[i] == UINT32_MAX) continue;
        const auto byte = differences[i], pixel = byte / 4;
        const auto extent = expected[i]->Extent;
        std::println(stderr, "[validation] {} DIVERGED at pixel ({}, {}), channel {} (byte {} of {})", names[i], pixel % extent.Width, pixel / extent.Width, byte % 4, byte, uint64_t(extent.Width) * extent.Height * 4);
        const auto expected_path = WriteValidationImage(ctx, i == 0 ? "live-app" : "live-viewport", *expected[i]);
        const auto actual_path = WriteValidationImage(ctx, names[i], *restored[i]);
        if (!expected_path.empty() && !actual_path.empty()) std::println(stderr, "[validation] wrote {} and {}", expected_path.string(), actual_path.string());
        WriteValidationProject(Session(r).History, workspace::Serialize(CaptureWorkspace(r)));
        std::abort();
    }
}

// Compare Persistent state, viewport pixels, and composed UI pixels after action replay and cold restoration.
void ValidateRoundTrip(
    state::Registry &r, CA::MetalLayer *layer,
    ImDrawData *draw_data, std::unique_ptr<ValidationSession> &session, MTL::CommandBuffer *presented_frame = nullptr
) {
    const auto total_begin = SteadyClock::now();
    auto begin = total_begin;
    if (!session) session = std::make_unique<ValidationSession>(r);
    const auto init_ms = ElapsedMs(begin);
    begin = SteadyClock::now();
    if (presented_frame) presented_frame->waitUntilCompleted();
    RenderAppImage(r.Resources.get<const mtl::Context>(), draw_data, session->Live);
    WaitForRender(r);
    if (!Session(r).History.Save()) {
        std::println(stderr, "[validation] {}", Session(r).History.TakeIntegrityError());
        std::abort();
    }
    if (std::string why; !Session(r).Audit(why)) {
        std::println(stderr, "[validation] live audit failed: {}", why);
        std::abort();
    }

    const auto live_state = Session(r).History.MaterializeLive();
    const ValidationInputs inputs{
        .WorkingDir = Paths::Project(),
        .Workspace = CaptureWorkspace(r),
        .AudioOutput = r.Resources.get<const AudioOutputConfig>(),
        .Device = r.Resources.get<const AudioDeviceInfo>(),
        .Layer = layer,
        .DisplaySize = draw_data->DisplaySize,
        .FramebufferScale = draw_data->FramebufferScale,
        .MousePos = GetIO().MousePos,
        .HoveredIdPreviousFrame = GImGui->HoveredIdPreviousFrame,
        .HoveredIdTimer = GImGui->HoveredIdTimer,
        .HoveredIdNotActiveTimer = GImGui->HoveredIdNotActiveTimer,
        .FocusedWindow = GImGui->NavWindow ? std::string{GImGui->NavWindow->Name} : std::string{},
        .NavId = GImGui->NavId,
        .NavCursorVisible = GImGui->NavCursorVisible,
        .NavHighlightItemUnderNav = GImGui->NavHighlightItemUnderNav,
        .Capturing = r.Resources.get<FrameState>().Capturing,
        .Scrubbing = r.Resources.get<FrameState>().Scrubbing,
        .PlaybackFrame = r.get<const PlaybackFrame>(state::Document).Value,
        .Playback = r.get<const TimelinePlayback>(state::Document),
    };

    const auto capture_ms = ElapsedMs(begin);
    const auto &live = Session(r).History;
    const auto replay = RestoreForValidation(session->Restored, inputs, live, true);
    begin = SteadyClock::now();
    const auto workspace_bytes = workspace::Serialize(inputs.Workspace);
    RequireEqual(live, "replay state", live_state, replay.State, workspace_bytes);
    CompareValidationImages(r, *session, "replay");
    auto compare_ms = ElapsedMs(begin);
    const auto restored = RestoreForValidation(session->Restored, inputs, live, false);
    begin = SteadyClock::now();
    RequireEqual(live, "stored state", live_state, restored.State, workspace_bytes);
    RequireEqual(live, "replay/stored workspace", replay.Workspace, restored.Workspace, workspace_bytes);
    CompareValidationImages(r, *session, "stored");
    compare_ms += ElapsedMs(begin);
    begin = SteadyClock::now();
    session->Restored.Ui.reset();
    const auto cleanup_ms = ElapsedMs(begin);
    std::println("[validation] total {:.2f} ms (init {:.2f}; capture {:.2f}; compare {:.2f}; cleanup {:.2f})", ElapsedMs(total_begin), init_ms, capture_ms, compare_ms, cleanup_ms);
    const auto report = [](std::string_view name, const RestoreTimings &v) {
        std::println("[validation] {} reset {:.2f}, restore {:.2f}, render {:.2f}, capture {:.2f} ms", name, v.ResetMs, v.RestoreMs, v.RenderMs, v.CaptureMs);
    };
    report("replay", replay.Timings);
    report("stored", restored.Timings);
}
#endif

std::expected<fs::path, std::string> SaveScreenshot(state::Registry &r, state::Entity viewport, const fs::path &path) {
    auto image = ReadbackViewportImage(r, viewport);
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
bool FrameScene(state::Registry &r, state::Entity viewport, float aspect_ratio) {
    const auto &cam = r.get<const ViewCameraState>(viewport).Active;
    const auto *persp = std::get_if<Perspective>(&cam.Data);
    if (!persp) return true;

    const vec3 right = cam.Orientation * vec3{1, 0, 0}, up = cam.Orientation * vec3{0, 1, 0}, away = cam.Forward();
    // Half the real frustum tangents, so a vertex reaching the (narrower) edge only fills the middle half of the true view.
    const float ty = 0.5f * std::tan(persp->FieldOfViewRad * 0.5f), tx = aspect_ratio * ty;
    constexpr float lowest = std::numeric_limits<float>::lowest();
    float top = lowest, bottom = lowest, rgt = lowest, lft = lowest;
    const auto &buffers = r.Resources.get<const GpuBuffers>();
    AABB scene;
    bool any_bounded_instance = false;
    for (const auto [e, ri] : r.view<const RenderInstance>(state::Exclude<Hidden>).each()) {
        const auto *world = WorldTransformOf(r, e);
        if (ri.BufferIndex == UINT32_MAX || !world) continue;
        any_bounded_instance = true;
        // Exclude empty bounds from gizmo and wireframe instances.
        const auto &local = buffers.Instances.GetBounds(ri.BufferIndex);
        if (local.Min.x > local.Max.x || local.Min.y > local.Max.y || local.Min.z > local.Max.z) continue;

        const auto m = ToMatrix(*world);
        for (int c = 0; c < 8; ++c) {
            const vec3 v{m * vec4{(c & 1) ? local.Max.x : local.Min.x, (c & 2) ? local.Max.y : local.Min.y, (c & 4) ? local.Max.z : local.Min.z, 1.f}};
            scene.Min = Min(scene.Min, v);
            scene.Max = Max(scene.Max, v);
            const float a = Dot(v, right), b = Dot(v, up), f = Dot(v, away);
            top = std::max(top, b / ty + f);
            bottom = std::max(bottom, -b / ty + f);
            rgt = std::max(rgt, a / tx + f);
            lft = std::max(lft, -a / tx + f);
        }
    }
    if (scene.Min.x > scene.Max.x || scene.Min.y > scene.Max.y || scene.Min.z > scene.Max.z) return !any_bounded_instance;

    const auto center = (scene.Min + scene.Max) * 0.5f;
    const float ca = Dot(center, right), cb = Dot(center, up), cf = Dot(center, away);
    const float distance = std::max({top - cb / ty, bottom + cb / ty, rgt - ca / tx, lft + ca / tx}) - cf;
    if (distance <= 0.f) return true;

    // Clip planes bracket the scene depth so nothing is z-clipped.
    const float plane_reach = 6 * Length(scene.Max - scene.Min);
    auto fit = *persp;
    fit.FarClip = distance + plane_reach;
    fit.NearClip = std::max(distance - plane_reach, fit.FarClip / 10000.f);

    r.patch<ViewCameraState>(viewport, [&](ViewCameraState &camera) { camera.Active = ViewCamera{center + distance * away, center, CameraLens{fit}}; });
    return true;
}

constexpr uvec2 DefaultWindowSize{1280, 800};

constexpr std::string_view Usage{
    R"(Usage: MeshEditor [scene] [options]

Loads a scene file, project directory, or .project/.actions archive, or the default scene when none is given.

Scene:
  --empty                     Start with an empty scene
  --camera NAME               Frame the named camera
  --shading MODE              wireframe | solid | preview | rendered
  --edit ELEMENT              Enter edit mode on vertex | edge | face
  --select-all                Select all objects or edited elements
  --xray                      Turn on X-ray for the capture's shading mode
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
  --motion-blur fast|SAMPLES  Fast velocity blur (default), or 1-64 full shutter samples
  --play [SECONDS]            Play the timeline, optionally for SECONDS

Benchmarking:
  --headless                  Run without a window
  --frames N                  Render N frames and exit
  --bench-action ACTION       steady | orbit | transform | visibility | box-select | box-select-orbit | pick-cycle
  --bench-action-count N      Objects the transform and visibility actions touch (default 64)
  --profile                   Print the profile report on exit
  --profile-json PATH         Write the profile report to PATH

Other:
  --help, -h                  Show this message)"
};

struct CaptureRequest {
    enum class BenchmarkAction { Steady,
                                 Orbit,
                                 Transform,
                                 Visibility,
                                 BoxSelect,
                                 BoxSelectOrbit,
                                 PickCycle };

    bool Play{false};
    float PlayDuration{0};
    int Fps{60};
    bool RecordAudio{false}; // Mux master audio; disabled for the video-only render corpus.
    fs::path RecordPath{}, ScreenshotPath{};
    fs::path RenderBasename{};
    std::optional<MotionBlur> Blur{};
    float TimelineEnd{0}; // Seconds. Positive: set the timeline's end frame, so a long play runs without looping.
    int BenchFrames{0};
    BenchmarkAction BenchAction{BenchmarkAction::Steady};
    uint32_t BenchActionCount{64};
    std::string CameraName{};
    std::optional<ViewportShadingMode> Shading{};
    bool Overlays{false}; // Keep overlays on through a capture, which presentation otherwise turns off.
    std::optional<Element> EditMode{}; // Engaged: select mesh objects and enter this element edit mode.
    bool XRay{false};
    bool SelectAll{false};
    float LodErrorPixels{-1.f}; // Cluster LOD error override in pixels; negative uses the viewport setting.
    SelectionOverlays Selection{};
};

struct BenchmarkDriver {
    CaptureRequest::BenchmarkAction Action;
    std::vector<std::pair<state::Entity, Transform>> Transforms;
    uint32_t Frame{};

    BenchmarkDriver(state::Registry &r, const CaptureRequest &capture) : Action(capture.BenchAction) {
        if (Action != CaptureRequest::BenchmarkAction::Transform && Action != CaptureRequest::BenchmarkAction::Visibility) return;
        std::vector<state::Entity> entities;
        for (const auto [entity, kind] : r.view<const ObjectKind>(state::Exclude<SubElementOf>).each()) {
            if (kind.Value == ObjectType::Mesh && r.all_of<Instance, Transform>(entity) && !r.all_of<Hidden>(entity)) entities.emplace_back(entity);
        }
        std::ranges::sort(entities);
        entities.resize(std::min<size_t>(entities.size(), capture.BenchActionCount));
        Transforms.reserve(entities.size());
        for (const auto entity : entities) Transforms.emplace_back(entity, r.get<const Transform>(entity));
    }

    void Apply(state::Registry &r, state::Entity viewport, uvec2 extent) {
        switch (Action) {
            case CaptureRequest::BenchmarkAction::Steady: break;
            case CaptureRequest::BenchmarkAction::Orbit:
                action::Emit(action::view::OrbitViewCamera{viewport, {0.01f, 0.f}});
                break;
            case CaptureRequest::BenchmarkAction::Transform: {
                const float offset = Frame % 2 == 0 ? 0.05f : 0.f;
                std::vector<action::Action> actions;
                for (const auto &[entity, base] : Transforms) actions.push_back(action::MakeAction(action::UpdateOf<&Transform::P>(entity, base.P + vec3{0, offset, 0})));
                action::Emit(std::move(actions));
                break;
            }
            case CaptureRequest::BenchmarkAction::Visibility: {
                std::vector<action::Action> actions;
                if (Frame == 0 && !Transforms.empty()) {
                    std::vector<state::Entity> entities;
                    for (const auto &[entity, _] : Transforms) entities.push_back(entity);
                    actions.push_back(action::MakeAction(action::selection::ApplyTreeSelection{.Entities = entities, .SelectCount = uint32_t(entities.size()), .NavToActive = entities.front(), .Clear = action::selection::ApplyTreeSelection::ClearKind::All}));
                }
                actions.push_back(action::MakeAction(action::object::SetVisible{Frame % 2 != 0}));
                action::Emit(std::move(actions));
                break;
            }
            case CaptureRequest::BenchmarkAction::BoxSelectOrbit:
                if (Frame % 2 != 0) {
                    // Keep both benchmark viewpoints on the scene throughout long runs.
                    action::Emit(action::view::OrbitViewCamera{viewport, {Frame % 4 == 1 ? 0.6f : -0.6f, 0.f}});
                    break;
                }
                [[fallthrough]];
            case CaptureRequest::BenchmarkAction::BoxSelect: {
                if (extent == uvec2{}) break;
                // The box alternates between two insets from the viewport edge so every frame changes the selection.
                const float inset = Frame % 2 == 0 ? 4.f : 8.f;
                const vec2 size{float(extent.x), float(extent.y)};
                action::Emit(action::selection::ApplyBoxSelect{
                    .Box = {vec2{inset} / size, (size - vec2{inset + 1.f}) / size},
                    .Additive = false,
                });
                break;
            }
            case CaptureRequest::BenchmarkAction::PickCycle: {
                const auto center = SelectionPixel(r.get<const Viewport>(viewport).FrameView.Extent, {.5f, .5f});
                action::Emit(action::selection::Pick{false, PickObject(r, viewport, center, Frame != 0)});
                break;
            }
        }
        ++Frame;
    }
};

bool SelectSceneCamera(state::Registry &r, std::string_view name) {
    if (name.empty()) return true;
    for (const auto [entity, node] : r.view<const GltfNode>().each()) {
        if (node.CameraName != name || !HasLens(r, entity)) continue;
        Perform(r, action::view::SetLookThroughCamera{FirstViewport(r), entity});
        return true;
    }
    std::println(stderr, "No scene camera named '{}'.", name);
    return false;
}

// Ordered by severity so combined statuses preserve real failures.
enum class SceneStatus {
    Ok,
    Unsupported,
    Failed
};

// Report and clear action errors.
// A real failure takes precedence over unsupported extensions.
SceneStatus ReportActionErrors(state::Registry &r) {
    auto &errors = r.Resources.get<action::Errors>();
    if (auto error = Session(r).History.TakeIntegrityError(); !error.empty()) action::Fail(r, std::move(error));
    if (errors.Messages.empty()) return SceneStatus::Ok;
    const auto status = errors.OnlyUnsupportedExtensions ? SceneStatus::Unsupported : SceneStatus::Failed;
    for (const auto &message : errors.Messages) std::println(stderr, "{}", message);
    errors.Messages.clear();
    return status;
}

// Preserve the initial load's error classification for capture and queue results.
SceneStatus SeedScene(state::Registry &r, const CaptureRequest &capture, const char *initial_file, bool empty) {
    const fs::path path = initial_file ? initial_file : "";
    bool loaded;
    std::error_code ec;
    if (fs::is_directory(path, ec) || path.extension() == ProjectExt || path.extension() == ActionsExt) loaded = OpenProjectFile(r, path);
    else {
        loaded = NewScene(r, initial_file || empty);
        if (loaded && initial_file) {
            Perform(r, action::io::Load{.Path = initial_file});
            if (const auto status = ReportActionErrors(r); status != SceneStatus::Ok) return status;
        }
    }
    return loaded && SelectSceneCamera(r, capture.CameraName) ? SceneStatus::Ok : SceneStatus::Failed;
}

struct CaptureDriver {
    // Fixed-step mode captures one timeline frame per GPU-paced tick; wall-clock mode samples at 1/Fps seconds.
    CaptureDriver(state::Registry &r, const CaptureRequest &capture, bool play, bool fixed_step)
        : Play(play), PlayDuration(capture.PlayDuration),
          FixedStep(fixed_step || !capture.RenderBasename.empty()),
          RecordPath(capture.RecordPath), ScreenshotPath(capture.ScreenshotPath), RenderBasename(capture.RenderBasename) {
        if (RenderMode()) {
            const auto with = [&](const char *ext) { return fs::path{capture.RenderBasename.string() + ext}; };
            const bool dynamic = !r.view<const PhysicsMotion>().empty() || !r.view<const AnimationClips>().empty();
            if (dynamic) RecordPath = with(".mp4");
            else ScreenshotPath = with(".webp");
        }
        const float timeline_fps = r.get<const TimelineRange>(state::Document).Fps;
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

    bool DurationElapsed(const Viewport &viewport) const {
        if (PlayDuration <= 0) return false;
        const float elapsed = RecordingMode() ? float(CapturedFrameCount(viewport)) / float(RecordFps) : ElapsedPlayTime;
        return elapsed >= PlayDuration;
    }

    // Emit capture actions before Project::Frame, after sizing the viewport.
    void EmitFrameActions(state::Registry &r, state::Entity viewport, bool settled, uvec2 extent) {
        if (!ViewFramed && Presenting() && extent != uvec2{}) {
            // Wait for GPU bounds before framing the launch camera.
            const bool framed = !r.view<const Perspective>().empty() || !r.view<const Orthographic>().empty() ||
                FrameScene(r, viewport, float(extent.x) / float(extent.y));
            ViewFramed = settled && framed;
        }
        // Start fixed-step playback after recording begins so the first frame is captured.
        const bool ready = RenderMode() || (FixedStep && RecordingMode()) ? IsRecording(r.get<const Viewport>(viewport)) : (Play || RecordingMode());
        if (!PlaybackStarted && Framed(settled) && ready) {
            const auto &playback = r.get<const TimelinePlayback>(state::Document);
            if (!playback.Playing) action::Emit(action::timeline::TogglePlay{playback.CurrentFrame});
            PlaybackStarted = true;
        }
    }

    bool CaptureFrame(state::Registry &r, state::Entity viewport, bool settled) {
        bool done = false;
        if (ScreenshotMode() && !ScreenshotSaved && settled) {
            if (auto saved = SaveScreenshot(r, viewport, ScreenshotPath); saved) std::println("Saved screenshot: {}", saved->string());
            else std::println(stderr, "Screenshot: {}", saved.error());
            // Save the default image before each material variant.
            const auto *mv = RenderMode() ? r.try_get<const MaterialVariants>(state::Document) : nullptr;
            if (mv && NextRenderVariant < mv->Names.size()) {
                auto name = mv->Names[NextRenderVariant].empty() ? std::format("Variant {}", NextRenderVariant) : mv->Names[NextRenderVariant];
                std::ranges::replace(name, '/', '-');
                ScreenshotPath = fs::path{RenderBasename.string() + "." + name + ".webp"};
                action::Emit(action::UpdateOf<&MaterialVariants::Active>(state::Document, std::optional{NextRenderVariant}));
                ++NextRenderVariant;
            } else {
                ScreenshotSaved = true;
                if (!RecordingMode() && PlayDuration <= 0) done = true;
            }
        }
        // End the run if the single recording startup fails.
        const auto &live = r.get<const Viewport>(viewport);
        if (RecordingMode() && settled) {
            if (!RecordingStarted) {
                RecordingStarted = true;
                StartRecording(r, viewport, RecordPath, RecordFps, RecordAudio);
                if (IsRecording(live)) NextCapture = SteadyClock::now();
            }
            if (!IsRecording(live)) done = true;
        }
        const bool loop_end = r.get<const TimelinePlayback>(state::Document).CurrentFrame == r.get<const TimelineRange>(state::Document).EndFrame;
        if (IsRecording(live)) {
            if (FixedStep) {
                // Emit clip switches to avoid an extra playback tick from mid-loop Perform calls.
                CaptureRecordFrame(r, viewport);
                if (loop_end) {
                    if (RenderMode()) {
                        if (NextRenderClip < r.get<const Animations>(state::Document).Names.size()) action::Emit(action::animation::SelectAnimation{NextRenderClip++});
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
    SceneStatus SeedStatus{SceneStatus::Ok};
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

CaptureDriver BeginCaptureSession(state::Registry &r, const CaptureRequest &capture, const char *initial_file, bool empty, bool fixed_step) {
    const auto seeded = SeedScene(r, capture, initial_file, empty);
    std::error_code ec;
    if (const auto parent = capture.RenderBasename.parent_path(); !parent.empty()) fs::create_directories(parent, ec);
    if (seeded != SceneStatus::Ok || ec) {
        if (ec) action::Fail(r, "Cannot create capture directory: " + ec.message());
        const auto errors = ReportActionErrors(r);
        CaptureDriver failed{r, capture, false, fixed_step};
        failed.SeedStatus = std::max(seeded, errors);
        return failed;
    }
    if (capture.EditMode) {
        std::vector<state::Entity> meshes;
        for (const auto [entity, kind, _] : r.view<const ObjectKind, const Instance>().each()) {
            if (kind.Value == ObjectType::Mesh) meshes.emplace_back(entity);
        }
        std::ranges::sort(meshes);
        if (!meshes.empty()) {
            Perform(r, action::selection::ApplyTreeSelection{
                           .Entities = meshes,
                           .SelectCount = uint32_t(meshes.size()),
                           .NavToActive = meshes.front(),
                           .Clear = action::selection::ApplyTreeSelection::ClearKind::All,
                       });
            Perform(r, action::view::SetInteractionMode{InteractionMode::Edit});
            Perform(r, action::view::SetEditMode{*capture.EditMode});
        }
    }
    if (capture.SelectAll) Perform(r, action::selection::SelectAll{});
    // A render plays the active animation once.
    if (const auto *animations = !capture.RenderBasename.empty() ? r.try_get<const Animations>(state::Document) : nullptr; animations && !animations->Names.empty()) {
        Perform(r, action::animation::SelectAnimation{animations->Active});
    }
    if (capture.TimelineEnd > 0) {
        const float fps = r.get<const TimelineRange>(state::Document).Fps;
        Perform(r, action::timeline::SetEndFrame{int(std::ceil(capture.TimelineEnd * fps))});
    }
    CaptureDriver driver{r, capture, capture.Play, fixed_step};
    // Preserve the editor view for benchmark frames and screenshots.
    if (driver.Presenting() && capture.BenchFrames == 0) Perform(r, action::timeline::EnterPresentation{});
    // Apply the explicit overlay override after enabling presentation mode.
    const auto viewport = FirstViewport(r);
    if (capture.Overlays) Perform(r, action::UpdateOf<&ViewportDisplay::ShowOverlays>(viewport, true));
    if (capture.LodErrorPixels >= 0.f) {
        Perform(r, action::UpdateOf<&ViewportDisplay::LodErrorPixels>(viewport, capture.LodErrorPixels));
    }
    r.Resources.get<FrameState>().FixedFrameStep = driver.FixedStep;
    // Enable motion blur for video recording and preserve the current setting for still or audio-only captures.
    r.Resources.get<FrameState>().Capturing = driver.RecordingMode() && !driver.AudioOnly();
    if (capture.Blur) {
        Perform(r, action::UpdateOf<&ViewportDisplay::MotionBlur>(viewport, capture.Blur));
    }
    if (capture.Shading) {
        Perform(r, action::view::SetViewportShading{viewport, *capture.Shading});
    }
    if (capture.XRay && !XRayFlag(r.get<const ViewportDisplay>(viewport))) Perform(r, action::view::ToggleXRay{viewport});
    return driver;
}

void run(const char *initial_file, bool empty, const CaptureRequest &capture) {
    MacPlatform::InitPaths();

    MacPlatform::Window window;

    std::optional<Engine> engine;
    engine.emplace(true);
    auto &r = engine->R;
    const auto &ctx = r.Resources.get<const mtl::Context>();
    if (!ctx.Device->supportsFamily(MTL::GPUFamilyMetal4)) throw std::runtime_error("MeshEditor renders with Metal 4, which this device does not support.");
    InitViewportMedia(r);

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
    io.IniFilename = nullptr;
    io.ConfigDebugIgnoreFocusLoss = true; // Keep input state across Cmd+Tab so in-flight gizmo drags survive focus loss.
    io.ConfigDragClickToInputText = true; // A click-release without dragging turns a Drag field into a text input.
    // Ctrl+Tab toggles pose mode, so ImGui's window cycling on the same chord is off.
    GetCurrentContext()->ConfigNavWindowingKeyNext = GetCurrentContext()->ConfigNavWindowingKeyPrev = 0;

    ui::StyleColors();

    ui::MacBackend mac_backend{window};
    ImGui_ImplMetal_Init(ctx.Device.get());

    InitFonts();

    profile::Init(ctx);
    // Read the DPI scale from NewFrame before initializing DPI-scaled GPU state.
    mac_backend.NewFrame();
    r.Resources.get<FrameState>().DisplayFramebufferScale = std::bit_cast<vec2>(io.DisplayFramebufferScale);
    ProcessComponentEvents(r);

    if (const auto stored = LoadAudioOutputConfig()) r.Resources.get<AudioOutputConfig>() = *stored;
    ReconcileOutputDevice(r, r.Resources.emplace<AudioDeviceResource>(r));

    SampleTreesFuture = std::async(std::launch::async, BuildSampleTrees);

    auto driver = BeginCaptureSession(r, capture, initial_file, empty, /*fixed_step=*/false);

#ifdef DEBUG_BUILD
    std::unique_ptr<ValidationSession> validation_session;
    std::optional<uint64_t> previous_ui_hash;
#endif
#ifdef VALIDATE_ACTIONS
    std::pair validated_revision{uint64_t{0}, RestoreGeneration};
#endif
    bool done{driver.SeedStatus != SceneStatus::Ok};
    uint8_t startup_frames_remaining{2};
    NS::SharedPtr<MTL::CommandBuffer> last_frame; // Retain across frame autorelease pools for resize waits.
    auto &panels = r.Resources.get<PanelsState>();
    int bench_ticks{0};
    while (!done) {
        const auto frame_pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        if (bench_ticks == 1) {
            profile::ReportCpuPhase("Scene load CPU timings");
            profile::ClearStats();
        }
        const profile::CpuScope frame_scope{"Frame"};
        r.Resources.get<ViewportConsumerFence>().Value = last_frame.get();
        const auto events = window.PollEvents();
        auto &frame_state = r.Resources.get<FrameState>();
        frame_state.PreciseWheelDelta = vec2{events.ScrollX, events.ScrollY};
        frame_state.PreciseScrolling = events.PreciseScrolling;
        frame_state.Magnification = events.Magnification;
        for (const auto &path : events.DroppedFiles) OpenFile(r, path);
        done = events.Quit;
        if (driver.DurationElapsed(r.get<const Viewport>(FirstViewport(r)))) done = true;

#ifdef DEBUG_BUILD
        const auto ui_revision = std::pair{Session(r).Revision, RestoreGeneration};
        const bool ui_has_staged = Session(r).HasStaged();
        const bool ui_has_pending_solves = HasPendingModalSolves(r);
#endif

        mac_backend.NewFrame();
        driver.ElapsedPlayTime += io.DeltaTime;
        // Scene-affecting code reads FrameState::DeltaTime.
        // `io.DeltaTime` is wall-clock, UI-only.
        r.Resources.get<FrameState>().DeltaTime = driver.FixedStep ? driver.RenderDt : io.DeltaTime;
        const auto *ui_focus = GImGui->NavWindow;
        NewFrame();
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            for (const auto [viewport, camera] : r.view<const ViewCameraState>().each()) {
                if (camera.Active.IsAnimating()) r.edit<ViewCameraState>(viewport).Active.StopMoving();
            }
        }

        const auto editor_frame = BeginEditorFrame(r, ctx, layer, panels, io, /*interactive=*/true);
        // Resolve the viewports after the editor panels, since opening a project applies its workspace.
        // Capture and benchmarks drive the first viewport.
        const auto first_viewport = FirstViewport(r);
        const auto first_panel = std::ranges::find(editor_frame.Viewports, first_viewport, &OpenPanel::Entity);
        const auto new_logical_extent = first_panel != editor_frame.Viewports.end() ? first_panel->Extent : uvec2{};
        const bool viewport_settled = new_logical_extent != uvec2{} && new_logical_extent == r.get<const ViewportPanel>(first_viewport).Extent;
        if (capture.BenchFrames > 0 && viewport_settled) {
            if (capture.BenchAction == CaptureRequest::BenchmarkAction::Orbit) action::Emit(action::view::OrbitViewCamera{first_viewport, {0.01f, 0.f}});
            if (++bench_ticks >= capture.BenchFrames) done = true;
        }
        // Give interaction gestures priority in the frame buffer.
        driver.EmitFrameActions(r, first_viewport, viewport_settled, new_logical_extent);
        if (frame_state.BoxDrag) {
            const auto active = FindActiveViewport(r);
            const auto panel = std::ranges::find(editor_frame.Viewports, active, &OpenPanel::Entity);
            if (panel == editor_frame.Viewports.end() || panel->Extent != r.get<const ViewportPanel>(active).Extent || frame_state.DisplayFramebufferScale != std::bit_cast<vec2>(io.DisplayFramebufferScale)) {
                action::Commit();
                frame_state.BoxDrag.reset();
            }
        }
        // Actions can add and remove viewports, which ends every viewport reference taken above.
        Session(r).Frame(action::Drain());
        ReportActionErrors(r);
        if (!frame_state.BoxDrag) {
            for (const auto [viewport, camera] : r.view<const ViewCameraState>().each()) {
                if (camera.Active.IsAnimating()) r.patch<ViewCameraState>(viewport, [](ViewCameraState &state) { state.Active.Tick(); });
            }
        }
        // Apply the next raster mapping after actions against the displayed mapping finish.
        frame_state.DisplayFramebufferScale = std::bit_cast<vec2>(io.DisplayFramebufferScale);
        for (const auto &panel : editor_frame.Viewports) {
            if (r.all_of<ViewportPanel>(panel.Entity) && panel.Extent != uvec2{} && panel.Extent != r.get<const ViewportPanel>(panel.Entity).Extent) {
                r.patch<ViewportPanel>(panel.Entity, [&](ViewportPanel &open) { open.Extent = panel.Extent; });
            }
        }

        // Submit derived state and rendering before the later WaitForRender synchronizes image sampling.
        // A viewport resized in its frame step waits for the previous UI frame to stop sampling its image.
        SubmitViewports(r);
        r.Resources.get<ViewportConsumerFence>().Value = nullptr;
        EndEditorFrame(r, panels, editor_frame);
        mac_backend.HonorMouseWarp();
        auto *draw_data = GetDrawData();
        const bool ui_gesture_settled = UiGestureSettled(ui_focus);
        if (const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f); !is_minimized) {
            WaitForRender(r); // Synchronize before ImGui samples the final image.
            if (driver.CaptureFrame(r, FirstViewport(r), driver.Framed(viewport_settled))) done = true;

#ifdef DEBUG_BUILD
            const auto scene_revision = std::pair{Session(r).Revision, RestoreGeneration};
#ifdef VALIDATE_ACTIONS
            const bool validate = panels.History.ValidateRequested || scene_revision != validated_revision;
#else
            const bool validate = panels.History.ValidateRequested;
#endif
            // The restored UI draws until it settles, so the live frame compares only once it has settled too, which a frame identical to the one before it shows.
            const auto ui_hash = validate && !ui_has_staged && ui_revision == scene_revision ? std::optional{DrawDataHash(draw_data)} : std::nullopt;
            const bool ui_matches_scene = ui_revision == scene_revision && ui_hash && ui_hash == previous_ui_hash;
            previous_ui_hash = ui_hash;
            // Background progress is transient; compare only frames drawn after it disappears.
            const bool validation_ready = validate && !Session(r).HasStaged() && ui_gesture_settled && !ui_has_pending_solves && !HasPendingModalSolves(r);
            const bool present_frame = !validation_ready || ui_matches_scene;
#else
            constexpr bool present_frame{true};
#endif

            MTL::CommandBuffer *presented_frame{nullptr};
            if (present_frame) {
                if (auto frame = RenderAndPresentFrame(ctx, layer, draw_data)) {
                    // ImGui makes newly bound dock nodes visible on the following frame.
                    if (startup_frames_remaining > 0 && --startup_frames_remaining == 0) {
                        frame->waitUntilCompleted();
                        window.Show();
                    }
                    presented_frame = frame.get();
                    last_frame = std::move(frame);
                }
            }

#ifdef DEBUG_BUILD
            if (validation_ready && presented_frame && GetFrameCount() > 1 && viewport_settled && ViewportImageReady(r.get<const Viewport>(FirstViewport(r)))) {
                ValidateRoundTrip(r, layer, draw_data, validation_session, presented_frame);
                panels.History.ValidateRequested = false;
#ifdef VALIDATE_ACTIONS
                validated_revision = scene_revision;
#endif
            }
#endif
        }

        static auto next_workspace_save = SteadyClock::now();
        if (ui_gesture_settled && (io.WantSaveIniSettings || SteadyClock::now() >= next_workspace_save)) {
            SaveWorkspace(r, /*force=*/false);
            io.WantSaveIniSettings = false;
            next_workspace_save = SteadyClock::now() + std::chrono::milliseconds{250};
        }
    }

    SaveWorkspace(r);
    Session(r).Save();
    if (!SaveAudioOutputConfig(r.Resources.get<const AudioOutputConfig>())) action::Fail(r, "Failed to save the audio output preference.");
    ReportActionErrors(r);
    if (last_frame) last_frame->waitUntilCompleted();

    r.Resources.erase<AudioDeviceResource>();
#ifdef DEBUG_BUILD
    validation_session.reset();
#endif
    DeinitViewportMedia(r);
    profile::Report();
    profile::Deinit();
    engine.reset();

    ImGui_ImplMetal_Shutdown();
    mac_backend.Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
}

SceneStatus RunHeadlessScene(state::Registry &r, const char *initial_file, bool empty, const CaptureRequest &capture) {
    // Initialize recorded inputs independently of the previous queued scene.
    auto &frame_state = r.Resources.get<FrameState>();
    frame_state.DeltaTime = 0;
    frame_state.FixedFrameStep = false;
    const auto scene_start = std::chrono::steady_clock::now();
    auto driver = BeginCaptureSession(r, capture, initial_file, empty, /*fixed_step=*/true);
    // Stop after reporting an initial scene-load failure.
    if (driver.SeedStatus != SceneStatus::Ok) {
        return driver.SeedStatus;
    }
    // Resolve the viewport after the scene seeds, since opening a project applies its workspace, and a viewport without one takes the window size.
    const auto viewport = FirstViewport(r);
    if (r.get<const ViewportPanel>(viewport).Extent == uvec2{}) r.patch<ViewportPanel>(viewport, [](ViewportPanel &panel) { panel.Extent = DefaultWindowSize; });
    if (capture.Selection.VertexNormals) Perform(r, action::UpdateOf<&ViewportDisplay::Selection, &SelectionOverlays::VertexNormals>(viewport, true));
    if (capture.Selection.FaceNormals) Perform(r, action::UpdateOf<&ViewportDisplay::Selection, &SelectionOverlays::FaceNormals>(viewport, true));
    if (capture.Selection.BoundingBoxes) Perform(r, action::UpdateOf<&ViewportDisplay::Selection, &SelectionOverlays::BoundingBoxes>(viewport, true));
    if (capture.Selection.TetWireframe) Perform(r, action::UpdateOf<&ViewportDisplay::Selection, &SelectionOverlays::TetWireframe>(viewport, true));

    frame_state.DeltaTime = driver.RenderDt;
    int bench_frames = capture.BenchFrames;
    BenchmarkDriver benchmark{r, capture};
    bool profile_cleared{false};
    bool submitted{false};
    bool done{false};
#ifdef VALIDATE_ACTIONS
    auto &validation_session = r.Resources.get<std::unique_ptr<ValidationSession>>();
    std::optional<std::pair<uint64_t, uint64_t>> validated_revision;
#endif
    // Record readiness after the first submitted frame has complete meshlet data.
    std::chrono::steady_clock::time_point first_frame_at{};
    while (!done) {
        const auto frame_pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        if (driver.DurationElapsed(r.get<const Viewport>(viewport))) break;
        const auto extent = r.get<const ViewportPanel>(viewport).Extent;
        // Require this scene's first submit because queue workers preserve ready images across scenes.
        const bool settled = submitted && ViewportImageReady(r.get<const Viewport>(viewport));
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
            if (bench_frames > 0 && settled) benchmark.Apply(r, viewport, RenderExtentPx(r, viewport));
            driver.EmitFrameActions(r, viewport, settled, extent);
            Session(r).Frame(action::Drain());
            ReportActionErrors(r);
            // Continue event processing while audio-only capture suppresses GPU rendering after initial framing.
            if (!driver.AudioOnly() || !driver.RecordingStarted) SubmitViewports(r);
            WaitForRender(r);
            if (!submitted) first_frame_at = std::chrono::steady_clock::now();
            submitted = true;
        }
#ifdef VALIDATE_ACTIONS
        const auto revision = std::pair{Session(r).Revision, RestoreGeneration};
        if (settled && revision != validated_revision && !Session(r).HasStaged() && !HasPendingModalSolves(r)) {
            const ValidationInputs inputs{
                .WorkingDir = Paths::Project(),
                .Workspace = CaptureWorkspace(r),
                .AudioOutput = r.Resources.get<const AudioOutputConfig>(),
                .Device = r.Resources.get<const AudioDeviceInfo>(),
                .DisplaySize = {float(DefaultWindowSize.x), float(DefaultWindowSize.y)},
                .FramebufferScale = std::bit_cast<ImVec2>(frame_state.DisplayFramebufferScale),
                .MousePos = {-FLT_MAX, -FLT_MAX},
            };
            auto *draw_data = RenderValidationApp(r, inputs);
            ValidateRoundTrip(r, nullptr, draw_data, validation_session);
            validated_revision = revision;
        }
#endif
        if (bench_frames > 0) {
            // Render every ready benchmark tick and stop after the requested count.
            if (settled && --bench_frames == 0) {
                if (driver.ScreenshotMode()) {
                    if (auto saved = SaveScreenshot(r, viewport, driver.ScreenshotPath); saved) std::println("Saved screenshot: {}", saved->string());
                    else std::println(stderr, "Screenshot: {}", saved.error());
                }
                done = true;
            }
            RequestRender(r, /*disoccludes=*/false);
        } else {
            if (driver.CaptureFrame(r, viewport, driver.Framed(settled))) done = true;
            // End an idle headless run after one completed frame.
            if (!driver.Presenting() && settled) done = true;
        }
        driver.ElapsedPlayTime += frame_state.DeltaTime;
    }
    const bool saved = !capture.RenderBasename.empty() ? SaveProjectFile(r, fs::path{capture.RenderBasename.string() + ".actions"}) : Session(r).Save();
    return saved ? SceneStatus::Ok : SceneStatus::Failed;
}

// Run scenes offscreen on a fixed-step, GPU-paced clock.
void RunHeadlessEngine(auto &&scenes) {
    MacPlatform::InitPaths();
    const File::TemporaryDirectory temporary{fs::temp_directory_path()};
    if (temporary.Path.empty()) throw std::runtime_error("Cannot create headless working directory.");
    HeadlessDirectory = temporary.Path;

    Engine engine{true};
    auto &r = engine.R;
    const auto &ctx = r.Resources.get<const mtl::Context>();
#ifdef VALIDATE_ACTIONS
    ValidationUi ui{ctx};
    r.Resources.emplace<std::unique_ptr<ValidationSession>>();
    InitFonts();
    InitViewportMedia(r);
#endif
    profile::Init(ctx);
    r.Resources.get<FrameState>().DisplayFramebufferScale = {2, 2}; // Match Retina pixel density and DPI-scaled rendering.
    ProcessComponentEvents(r);

    scenes(r);

    WaitForRender(r);
#ifdef VALIDATE_ACTIONS
    r.Resources.erase<std::unique_ptr<ValidationSession>>();
    DeinitViewportMedia(r);
#endif
    profile::Report();
    profile::Deinit();
    Paths::SetProject({});
    HeadlessDirectory.clear();
}

SceneStatus FinishHeadlessScene(state::Registry &r, SceneStatus status) {
    QuiesceScene(r);
    for (const auto [_, viewport] : r.view<Viewport>().each()) viewport.Recording.reset();
    EndAudioCapture(r);
    const auto working = Paths::Project(); // NOLINT(performance-unnecessary-copy-initialization) SetProject replaces the referenced path.
    const bool closed = Session(r).Close();
    const auto errors = ReportActionErrors(r);
    Paths::SetProject({});
    std::error_code ec;
    if (!working.empty()) fs::remove_all(working, ec);
    return std::max(status, closed ? errors : SceneStatus::Failed);
}

bool RunHeadless(const char *initial_file, bool empty, const CaptureRequest &capture) {
    bool ok = true;
    RunHeadlessEngine([&](state::Registry &r) {
        ok = FinishHeadlessScene(r, RunHeadlessScene(r, initial_file, empty, capture)) == SceneStatus::Ok;
    });
    return ok;
}

struct LaunchOptions {
    std::string InitialFile{};
    bool Empty{false}, Headless{false};
    fs::path RenderQueue{};
    CaptureRequest Capture;
};

// Queue jobs and direct launches interpret capture settings identically.
std::expected<LaunchOptions, int> ParseLaunchOptions(std::span<const std::string> args, CaptureRequest capture_defaults = {}) {
    const auto looks_numeric = [](std::string_view s) {
        return !s.empty() && std::isdigit(uint8_t(s[0]));
    };

    LaunchOptions options{.Capture = std::move(capture_defaults)};
    auto &[initial_file, empty, headless, render_queue, capture] = options;
    for (auto it = args.begin(); it != args.end(); ++it) {
        const std::string_view a{*it};
        if (a == "--help" || a == "-h") {
            std::println("{}", Usage);
            return std::unexpected(0);
        }
        if (a == "--play") {
            capture.Play = true;
            if (std::next(it) != args.end() && looks_numeric(*std::next(it))) capture.PlayDuration = std::atof((++it)->c_str());
        } else if (a == "--record" && std::next(it) != args.end()) {
            capture.RecordPath = *++it;
        } else if (a == "--record-audio") {
            capture.RecordAudio = true;
        } else if (a == "--screenshot" && std::next(it) != args.end()) {
            capture.ScreenshotPath = *++it;
        } else if (a == "--render" && std::next(it) != args.end()) {
            capture.RenderBasename = *++it;
        } else if (a == "--render-queue" && std::next(it) != args.end()) {
            render_queue = *++it;
        } else if (a == "--empty") {
            empty = true;
        } else if (a == "--headless") {
            headless = true;
        } else if (a == "--overlays") {
            capture.Overlays = true;
        } else if (a == "--edit" && std::next(it) != args.end()) {
            const std::string_view mode{*++it};
            if (mode == "vertex") capture.EditMode = Element::Vertex;
            else if (mode == "edge") capture.EditMode = Element::Edge;
            else if (mode == "face") capture.EditMode = Element::Face;
            else {
                std::println(stderr, "Unknown edit element '{}'.", mode);
                return std::unexpected(1);
            }
        } else if (a == "--lod-error" && std::next(it) != args.end()) {
            capture.LodErrorPixels = std::stof(std::string{*++it});
        } else if (a == "--xray") {
            capture.XRay = true;
        } else if (a == "--select-all") {
            capture.SelectAll = true;
        } else if (a == "--display" && std::next(it) != args.end()) {
            const std::string_view names{*++it};
            for (size_t start = 0; start <= names.size();) {
                const auto end = std::min(names.find(',', start), names.size());
                const auto item = names.substr(start, end - start);
                start = end + 1;
                if (item == "vertex-normals") capture.Selection.VertexNormals = true;
                else if (item == "face-normals") capture.Selection.FaceNormals = true;
                else if (item == "bounds") capture.Selection.BoundingBoxes = true;
                else if (item == "tet-wireframe") capture.Selection.TetWireframe = true;
                else {
                    std::println(stderr, "Unknown display overlay '{}'.", item);
                    return std::unexpected(1);
                }
            }
        } else if (a == "--fps" && std::next(it) != args.end()) {
            capture.Fps = std::atoi((++it)->c_str());
        } else if (a == "--timeline-end" && std::next(it) != args.end()) {
            capture.TimelineEnd = std::atof((++it)->c_str());
        } else if (a == "--motion-blur" && std::next(it) != args.end()) {
            const std::string_view method = *++it;
            capture.Blur = method == "fast" ? MotionBlur{} : MotionBlur{.Steps = uint8_t(std::clamp(std::atoi(method.data()), 1, 64)), .Method = MotionBlurMethod::FullSampling};
        } else if (a == "--frames" && std::next(it) != args.end()) {
            capture.BenchFrames = std::atoi((++it)->c_str());
        } else if (a == "--bench-action" && std::next(it) != args.end()) {
            const std::string_view action{*++it};
            if (action == "steady") capture.BenchAction = CaptureRequest::BenchmarkAction::Steady;
            else if (action == "orbit") capture.BenchAction = CaptureRequest::BenchmarkAction::Orbit;
            else if (action == "transform") capture.BenchAction = CaptureRequest::BenchmarkAction::Transform;
            else if (action == "visibility") capture.BenchAction = CaptureRequest::BenchmarkAction::Visibility;
            else if (action == "box-select") capture.BenchAction = CaptureRequest::BenchmarkAction::BoxSelect;
            else if (action == "box-select-orbit") capture.BenchAction = CaptureRequest::BenchmarkAction::BoxSelectOrbit;
            else if (action == "pick-cycle") capture.BenchAction = CaptureRequest::BenchmarkAction::PickCycle;
            else {
                std::println(stderr, "Unknown benchmark action '{}'.", action);
                return std::unexpected(1);
            }
        } else if (a == "--bench-action-count" && std::next(it) != args.end()) {
            capture.BenchActionCount = uint32_t(std::max(1, std::atoi((++it)->c_str())));
        } else if (a == "--camera" && std::next(it) != args.end()) {
            capture.CameraName = *++it;
        } else if (a == "--shading" && std::next(it) != args.end()) {
            const std::string_view mode{*++it};
            capture.Shading = mode == "wireframe" ? ViewportShadingMode::Wireframe :
                mode == "solid"                   ? ViewportShadingMode::Solid :
                mode == "preview"                 ? ViewportShadingMode::MaterialPreview :
                                                    ViewportShadingMode::Rendered;
        } else if (a == "--profile") {
            profile::Enabled = true;
        } else if (a == "--profile-json" && std::next(it) != args.end()) {
            profile::Enabled = true;
            profile::JsonPath = *++it;
        } else if (a.starts_with('-')) {
            std::println(stderr, "Unknown option '{}'. Run with --help for the option list.", a);
            return std::unexpected(1);
        } else if (initial_file.empty()) {
            initial_file = *it;
        }
    }
    if (capture.Fps <= 0) capture.Fps = 60;
    if (capture.RecordPath.extension() == ".wav") capture.RecordAudio = true;
    if (!capture.RenderBasename.empty() && (!capture.RecordPath.empty() || !capture.ScreenshotPath.empty())) {
        std::println(stderr, "--render cannot be combined with --record or --screenshot");
        return std::unexpected(1);
    }
    if (!render_queue.empty() && (!initial_file.empty() || !capture.RenderBasename.empty() || !capture.RecordPath.empty() || !capture.ScreenshotPath.empty())) {
        std::println(stderr, "--render-queue cannot be combined with a scene file or capture flags");
        return std::unexpected(1);
    }

    return options;
}

// Numbered .job files contain an output basename, then one command-line argument per line.
struct RenderJob {
    size_t Index;
    fs::path OutBasename;
    std::vector<std::string> Args{};
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
        RenderJob job{.Index = std::stoull(path.stem().string()), .OutBasename = std::move(line)};
        while (std::getline(in, line)) job.Args.push_back(std::move(line));
        return job;
    }
    return std::nullopt;
}

// Render every queued job with one engine and write each scene's console output to its log.
bool RunHeadlessQueue(const fs::path &spool, const CaptureRequest &harness) {
    bool succeeded = true;
    RunHeadlessEngine([&](state::Registry &r) {
        const int launcher_out = ::dup(STDOUT_FILENO), launcher_err = ::dup(STDERR_FILENO);
        while (const auto job = ClaimRenderJob(spool)) {
            const auto begin = SteadyClock::now();
            const auto out = job->OutBasename.string();
            std::fflush(stdout);
            std::fflush(stderr);
            if (const int log_fd = ::open((out + ".log").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644); log_fd >= 0) {
                ::dup2(log_fd, STDOUT_FILENO);
                ::dup2(log_fd, STDERR_FILENO);
                ::close(log_fd);
            }
            auto status = SceneStatus::Failed;
            if (const auto options = ParseLaunchOptions(job->Args, harness)) {
                const auto &file = options->InitialFile;
                status = RunHeadlessScene(r, file.empty() ? nullptr : file.c_str(), options->Empty, options->Capture);
            }
            status = FinishHeadlessScene(r, status);
            succeeded &= status != SceneStatus::Failed;
            std::fflush(stdout);
            std::fflush(stderr);
            ::dup2(launcher_out, STDOUT_FILENO);
            ::dup2(launcher_err, STDERR_FILENO);
            std::println(R"({{"job":{},"status":"{}","seconds":{:.2f}}})", job->Index, std::array{"ok", "unsupported", "failed"}[size_t(status)], ElapsedMs(begin) / 1000.0);
            std::fflush(stdout);
        }
        ::close(launcher_out);
        ::close(launcher_err);
    });
    return succeeded;
}
} // namespace

int main(int argc, char **argv) {
    const auto autorelease_pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

    // Convert ffmpeg pipe closure into EPIPE for VideoRecorder error handling.
    std::signal(SIGPIPE, SIG_IGN);

    std::set_terminate([]() {
        try {
            if (const auto eptr = std::current_exception()) std::rethrow_exception(eptr);
        } catch (const std::exception &e) {
            std::println(stderr, "{}", e.what());
        } catch (...) {
            std::println(stderr, "Unhandled unknown exception");
        }
        std::abort();
    });

    const auto options = ParseLaunchOptions(std::vector<std::string>{argv + 1, argv + argc});
    if (!options) return options.error();
    const auto &[file, empty, headless, render_queue, capture] = *options;
    const char *initial_file = file.empty() ? nullptr : file.c_str();

    bool headless_ok = true;
    if (!render_queue.empty()) headless_ok = RunHeadlessQueue(render_queue, capture);
    else if (headless) headless_ok = RunHeadless(initial_file, empty, capture);
    else run(initial_file, empty, capture);
    // Report failed captures and headless scene loads through the process status.
    return !headless_ok || VideoRecorder::AnyFailed() ? 1 : 0;
}
