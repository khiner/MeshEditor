#include "Paths.h"
#include "Timer.h"
#include "Window.h"
#include "action/ActionApply.h"
#include "action/Emit.h"
#include "action/Errors.h"
#include "action/Io.h"
#include "action/Log.h"
#include "action/Object.h"
#include "action/View.h"
#include "animation/TimelineUi.h"
#include "audio/AudioDevice.h"
#include "audio/AudioSystem.h"
#include "gizmo/TransformGizmoTypes.h"
#include "image/ImageEncode.h"
#include "render/SvgResource.h"
#include "render/SvgUpload.h"
#include "scene/SceneControlsUi.h"
#include "snapshot/ReplayTestFixture.h"
#include "snapshot/SaveState.h"
#include "snapshot/SceneSnapshot.h"
#include "viewport/FrameState.h"
#include "viewport/ViewCamera.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportIcons.h"
#include "viewport/ViewportUi.h"
#include "vulkan/VulkanContext.h"

#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "imgui_internal.h"
#include "implot.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <entt/entity/registry.hpp>
#include <nfd.h>

#include <csignal>
#include <ctime>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <print>
#include <set>

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
        if (!entry.is_regular_file() || (ext != ".glb" && ext != ".gltf")) continue;
        samples.push_back({entry.path().filename().string(), entry.path(), ReadExtensionsUsed(entry.path())});
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
        node->Files.push_back(std::move(s));
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

// Emit an action to load a file into the scene based on its extension. Load errors surface via action::Errors.
void LoadFile(entt::registry &r, const fs::path &path) {
    const auto ext = path.extension().string();
    if (ext == ".gltf" || ext == ".glb") {
        action::Emit(action::io::LoadGltf{.Path = path.string()});
    } else if (ext == ".obj" || ext == ".ply") {
        action::Emit(action::object::ImportMesh{path.string(), std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Name = path.stem().string()})});
    } else {
        r.ctx().get<action::Errors>().Messages.push_back(std::format("Unsupported file format: '{}'", ext));
    }
}

// Reset to the default scene, optionally replaying a `.mea` log on top.
void NewProject(entt::registry &r, entt::entity viewport, const fs::path &replay_path = {}, bool with_default_content = true, [[maybe_unused]] bool is_current_replay = false) {
    // Invoked mid-frame from the menu: the prior frame may still be sampling the viewport resources replay is
    // about to recreate, and replay has no consumer fence to wait on. Let the GPU finish all work first.
    r.ctx().get<const VulkanResources>().Device.waitIdle();
    action::StopPlaybackIfPlaying(r, viewport);
#ifdef DEBUG_BUILD
    // Replay test: replaying the live session's *own* log should reproduce the live scene exactly. Only then is
    // the current scene a valid `expected` — replaying a past session's log has nothing to do with live state,
    // so the check (and fixture write) is gated to the current log. Snapshot now, replay below, and compare.
    const bool replay_check = !replay_path.empty() && is_current_replay;
    std::vector<std::byte> expected;
    if (replay_check) expected = snapshot::SnapshotSceneState(r);
#endif
    action::StopLog();
    const auto live_extent = r.ctx().get<ViewportExtent>().Value; // Restore after replay's SetExtent actions change it.
    // View-camera navigation isn't recorded, so replay must not disturb the live view. Restore it afterward.
    auto live_view_camera = r.get<ViewCamera>(viewport);
    ClearScene(r, viewport);
    // Default content is the replay baseline.
    // New->Empty skips building it live to avoid a one-frame flash before its recorded Clear applies.
    if (with_default_content) AddDefaultSceneContent(r);
    // Start the new session log before replaying so ReplayLog re-records the actions into it (otherwise the
    // replayed log is consumed and replaying "Current" again reloads the default scene).
    action::StartLog();
    if (action::ReplayLog(r, viewport, replay_path, &AdvanceViewport)) {
        // Replay ran headless (resized selection resources but never presented the color image).
        // Restore the live window extent and view camera, then render the final replayed state once on-screen.
        r.ctx().get<ViewportExtent>().Value = live_extent;
        r.emplace_or_replace<ViewCamera>(viewport, std::move(live_view_camera));
        PresentViewport(r, viewport);
#ifdef DEBUG_BUILD
        if (replay_check) {
            const auto actual = snapshot::SnapshotSceneState(r);
            if (const auto diff = snapshot::Compare(expected, actual); !diff.Equal) {
                snapshot::WriteReplayTestFixture(replay_path, expected, actual);
            }
        }
#endif
    }
}

#ifdef DEBUG_BUILD
// Save the full persistent state, clear+restore it, run one update pass, and compare the re-saved state.
// Proves the snapshot mechanism reproduces the saved image byte-for-byte across arenas, serialize/restore, and exact entity-handle recreation.
void ValidateSnapshotRoundTrip(entt::registry &r, entt::entity viewport) {
    r.ctx().get<const VulkanResources>().Device.waitIdle();
    action::StopPlaybackIfPlaying(r, viewport);
    const auto before = snapshot::SaveState(r);
    ClearScene(r, viewport);
    snapshot::LoadState(r, before);
    AdvanceViewport(r, viewport);
    const auto after = snapshot::SaveState(r);
    if (const auto diff = snapshot::Compare(before, after); diff.Equal) {
        std::println(stderr, "[snapshot] round-trip OK ({} bytes)", before.size());
    } else {
        std::println(stderr, "[snapshot] round-trip DIVERGED at byte {} (before {} / after {})", diff.FirstDifferingByte, before.size(), after.size());
    }
    PresentViewport(r, viewport);
}
#endif

// Read back the viewport and write it to `path`, choosing the encoder by extension (defaulting to .png).
// Returns the resolved output path on success.
std::expected<fs::path, std::string> SaveScreenshot(entt::registry &r, const fs::path &path) {
    auto image = ReadbackViewportImage(r);
    if (!image) return std::unexpected{std::move(image.error())};

    auto ext = path.extension().string();
    std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return std::tolower(c); });
    auto out_path = ext.empty() ? fs::path{path}.replace_extension(".png") : path;
    const auto name = out_path.filename().string();
    const auto encoded = ext == ".jpg" || ext == ".jpeg" ? EncodeImageJpegRgba8(image->Pixels, image->Width, image->Height, 95, name) : EncodeImagePngRgba8(image->Pixels, image->Width, image->Height, name);
    if (!encoded) return std::unexpected{std::move(encoded.error())};

    std::ofstream out{out_path, std::ios::binary};
    out.write(reinterpret_cast<const char *>(encoded->data()), std::streamsize(encoded->size()));
    if (!out) return std::unexpected{std::format("failed to write '{}'", out_path.string())};
    return out_path;
}

void run(const char *initial_file, bool quiet, bool play, float play_duration, const fs::path &record_path, const fs::path &screenshot_path, int record_fps) {
    Timer::Enabled = !quiet;

    SDL_SetHint(SDL_HINT_MAC_SCROLL_MOMENTUM, "1");
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) throw std::runtime_error(std::format("SDL_Init error: {}", SDL_GetError()));

    if (const char *base = SDL_GetBasePath()) Paths::Init(base);
    else throw std::runtime_error(std::format("SDL_GetBasePath error: {}", SDL_GetError()));

    auto *window = SDL_CreateWindow(
        "MeshEditor", 1280, 800,
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
        const VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR};
#else
        const VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_FIFO_KHR};
#endif
        wd.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(vc->PhysicalDevice, wd.Surface, &present_modes[0], IM_COUNTOF(present_modes));
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

    NFD_Init();
    entt::registry r;
    auto create_svg = [device = *vc->Device, &r, &wd](std::unique_ptr<SvgResource> &svg, fs::path path) {
        // Wait for previous frame's ImGui render to complete, since it may have sampled the old texture.
        CheckVk(device.waitForFences({wd.Frames[wd.FrameIndex].Fence}, true, UINT64_MAX));
        svg = LoadSvg(r, std::move(path));
    };
    const auto viewport = InitEngine(r, vc->Resources());
    InitViewportMedia(r, std::move(create_svg));
    SetupScene(r, viewport); // Before the first frame reads viewport state.
    AddDefaultSceneContent(r);
    AdvanceViewport(r, viewport); // Prime derived state before the first frame reads it.

    struct AudioContext {
        entt::registry *R;
        entt::entity Viewport;
    };
    AudioContext audio_ctx{&r, viewport};
    AudioDevice audio_device{
        {.Callback = [](auto buffer, void *user_data) {
             auto &ctx = *static_cast<AudioContext *>(user_data);
             ProcessAudio(*ctx.R, ctx.Viewport, std::move(buffer));
         },
         .UserData = &audio_ctx}
    };
    audio_device.Start();

    action::StartLog(); // Record each applied action to the write-behind log.
    if (initial_file) {
        ClearScene(r, viewport);
        LoadFile(r, initial_file);
        action::ApplyEmitted(r, viewport);
        AdvanceViewport(r, viewport);
        if (auto &errors = r.ctx().get<action::Errors>().Messages; !errors.empty()) {
            for (const auto &message : errors) std::cerr << message << std::endl;
            play = false; // Don't auto-play if the initial file failed to load.
            errors.clear();
        }
    }

    const bool recording_mode = !record_path.empty(), screenshot_mode = !screenshot_path.empty();
    // Enter presentation mode so the first rendered frame matches the capture.
    // Playback start and the capture itself wait until the viewport settles.
    if (play || screenshot_mode || recording_mode) {
        action::Emit(action::timeline::EnterPresentation{});
        action::ApplyEmitted(r, viewport);
        AdvanceViewport(r, viewport);
    }

    // Sim always uses the real `io.DeltaTime` from SDL, so sim-clock = wall-clock (1:1 real-time) whether or not we're recording.
    // Recording just samples the already-live viewport: a wall-clock accumulator captures one frame every `1/record_fps` seconds,
    // decimating if the display refreshes faster than the video's fps.
    // This keeps the in-app preview and the recorded video playing at the exact same rate.
    const uint64_t capture_interval_ns{1'000'000'000ULL / uint64_t(record_fps)};
    uint64_t next_capture_ns{0}; // Initialized when recording starts.
    float elapsed_play_time{0};
    bool playback_started{false}, screenshot_saved{false};
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
            if (event.type == SDL_EVENT_DROP_FILE) LoadFile(r, event.drop.data);
            // SDL3 backend invalidates MousePos to -FLT_MAX on leave when no mouse button is held,
            // which flings a keyboard-initiated (G/R/S) transform offscreen when switching focus.
            if (event.type == SDL_EVENT_WINDOW_MOUSE_LEAVE && TransformGizmo::IsUsing(r, viewport)) continue;
            ImGui_ImplSDL3_ProcessEvent(&event);
            done = event.type == SDL_EVENT_QUIT || (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window));
        }
        const float elapsed = recording_mode ? float(CapturedFrameCount(r, viewport)) / float(record_fps) : elapsed_play_time;
        if (play_duration > 0 && elapsed >= play_duration) done = true;

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
        elapsed_play_time += io.DeltaTime;
        {
            auto &frame = r.ctx().get<FrameState>();
            frame.DeltaTime = io.DeltaTime;
            frame.DisplayFramebufferScale = {io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y};
        }
        NewFrame();

        auto dockspace_id = DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_AutoHideTabBar);
        if (GetFrameCount() == 1) {
            auto controls_node_id = DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.3f, nullptr, &dockspace_id);
            auto extra_node_id = DockBuilderSplitNode(controls_node_id, ImGuiDir_Down, 0.4f, nullptr, &controls_node_id);
            auto animation_node_id = DockBuilderSplitNode(dockspace_id, ImGuiDir_Down, 0.1f, nullptr, &dockspace_id);
            DockBuilderDockWindow(windows.Debug.Name, extra_node_id);
            DockBuilderDockWindow(windows.ImGuiDemo.Name, extra_node_id);
            DockBuilderDockWindow(windows.ImPlotDemo.Name, extra_node_id);
            DockBuilderDockWindow(windows.SceneControls.Name, controls_node_id);
            DockBuilderDockWindow(windows.Animation.Name, animation_node_id);
            DockBuilderDockWindow(windows.Viewport.Name, dockspace_id);
        }

        if (BeginMainMenuBar()) {
            if (BeginMenu("File")) {
                if (BeginMenu("New")) {
                    if (MenuItem("Default")) NewProject(r, viewport);
                    if (MenuItem("Empty")) {
                        NewProject(r, viewport, {}, /*with_default_content=*/false);
                        action::Emit(action::io::Clear{}); // Recorded on the fresh log so replay reconstructs the empty scene.
                    }
                    EndMenu();
                }
                if (BeginMenu("Replay")) {
                    const auto logs = action::ListReplayLogs(); // Most-recent first; the newest is the live session's log.
                    for (size_t i = 0; i < logs.size(); ++i) {
                        const std::time_t t = logs[i].UnixSeconds;
                        char date[32];
                        std::strftime(date, sizeof date, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
                        const auto label = i == 0 ? std::format("Current ({})", date) : std::string{date};
                        if (MenuItem(label.c_str())) NewProject(r, viewport, logs[i].Path, /*with_default_content=*/true, /*is_current_replay=*/i == 0);
                    }
                    EndMenu();
                }
#ifdef DEBUG_BUILD
                if (MenuItem("Validate snapshot round-trip")) ValidateSnapshotRoundTrip(r, viewport);
#endif
                const auto import_dialog = [&r](const auto &filters) {
                    nfdchar_t *nfd_path;
                    if (auto result = NFD_OpenDialog(&nfd_path, filters.data(), filters.size(), ""); result == NFD_OKAY) {
                        LoadFile(r, nfd_path);
                        NFD_FreePath(nfd_path);
                    } else if (result != NFD_CANCEL) {
                        std::cerr << "Error opening file dialog: " << NFD_GetError() << std::endl;
                    }
                };
                if (BeginMenu("Import")) {
                    if (MenuItem("glTF 2.0 (.glb/.gltf)")) {
                        static constexpr std::array filters{nfdfilteritem_t{"glTF scene", "gltf,glb"}};
                        import_dialog(filters);
                    }
                    if (MenuItem("Wavefront (.obj)")) {
                        static constexpr std::array filters{nfdfilteritem_t{"Wavefront OBJ", "obj"}};
                        import_dialog(filters);
                    }
                    if (MenuItem("Stanford PLY (.ply)")) {
                        static constexpr std::array filters{nfdfilteritem_t{"Stanford PLY", "ply"}};
                        import_dialog(filters);
                    }
                    if (MenuItem("RealImpact")) {
                        nfdchar_t *path;
                        if (auto result = NFD_PickFolder(&path, ""); result == NFD_OKAY) {
                            action::Emit(action::io::LoadRealImpact{.Directory = std::string{path}});
                            NFD_FreePath(path);
                        } else if (result != NFD_CANCEL) {
                            std::cerr << "Error opening folder dialog: " << NFD_GetError() << std::endl;
                        }
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
                    for (const auto &[name, c] : n.Children) items.push_back({&name, &c, nullptr});
                    for (const auto &f : n.Files) items.push_back({&f.Label, nullptr, &f});
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
                            if (MenuItem(it.File->Label.c_str())) LoadFile(r, it.File->Path);
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
                static const auto Examples = BuildGltfSampleTree(Paths::Res() / "examples");
                render_submenu("Examples", Examples);
#ifdef GLTF_SAMPLE_ASSETS_DIR
                static const auto SampleAssets = BuildGltfSampleTree(fs::path{GLTF_SAMPLE_ASSETS_DIR} / "Models");
                static const auto SampleAssetsExtensions = [] {
                    std::set<std::string> exts;
                    [&](this auto &&self, const GltfSampleTree &n) -> void {
                        for (const auto &f : n.Files) exts.insert_range(f.Extensions);
                        for (const auto &[_, c] : n.Children) self(c);
                    }(SampleAssets);
                    return exts;
                }();
                static std::set<std::string> SampleAssetsFilter;
                if (!SampleAssets.Files.empty() || !SampleAssets.Children.empty()) {
                    if (BeginMenu("glTF Samples")) {
                        if (BeginMenu("Filter extensions")) {
                            PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
                            for (const auto &ext : SampleAssetsExtensions) {
                                const bool checked = SampleAssetsFilter.contains(ext);
                                if (MenuItem(ext.c_str(), nullptr, checked)) {
                                    if (checked) SampleAssetsFilter.erase(ext);
                                    else SampleAssetsFilter.insert(ext);
                                }
                            }
                            PopItemFlag();
                            EndMenu();
                        }
                        render_tree(SampleAssets, [](const GltfSample &f) { return all_of(SampleAssetsFilter, [&](const auto &e) { return f.Extensions.contains(e); }); });
                        EndMenu();
                    }
                }
#endif
#ifdef GLTF_PHYSICS_DIR
                static const auto PhysicsTree = BuildGltfSampleTree(GLTF_PHYSICS_DIR);
                render_submenu("glTF_Physics Samples", PhysicsTree);
#endif
                if (MenuItem("Save glTF", nullptr)) {
                    static const std::array filters{nfdfilteritem_t{"glTF scene", "gltf,glb"}};
                    nfdchar_t *nfd_path;
                    if (auto result = NFD_SaveDialog(&nfd_path, filters.data(), filters.size(), nullptr, "scene.gltf"); result == NFD_OKAY) {
                        action::Emit(action::io::SaveGltf{.Path = std::string{nfd_path}});
                        NFD_FreePath(nfd_path);
                    } else if (result != NFD_CANCEL) {
                        std::cerr << "Error opening save dialog: " << NFD_GetError() << std::endl;
                    }
                }
                EndMenu();
            }
            if (BeginMenu("Windows")) {
                MenuItem(windows.Debug.Name, nullptr, &windows.Debug.Visible);
                MenuItem(windows.ImGuiDemo.Name, nullptr, &windows.ImGuiDemo.Visible);
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
        if (windows.ImPlotDemo.Visible) ImPlot::ShowDemoWindow(&windows.ImPlotDemo.Visible);

        if (windows.SceneControls.Visible) {
            if (Begin(windows.SceneControls.Name, &windows.SceneControls.Visible) && BeginTabBar("Controls")) {
                if (BeginTabItem("Scene")) {
                    RenderControls(r, viewport);
                    EndTabItem();
                }
                if (BeginTabItem("Audio device")) {
                    audio_device.RenderControls();
                    EndTabItem();
                }
                EndTabBar();
            }
            End();
        }

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
                if (auto a = RenderAnimationTimeline(r.get<const TimelineRange>(scene_e), r.get<const TimelinePlayback>(scene_e), r.get<const AnimationTimelineView>(scene_e), r.ctx().get<const ViewportIcons>().Anim)) {
                    std::visit([](auto leaf) { action::Emit(leaf); }, std::move(*a));
                }
            }
            End();
            PopStyleVar();
        }

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
                const auto content_region = ImGui::GetContentRegionAvail();
                new_logical_extent = {uint32_t(std::max(content_region.x, 0.f)), uint32_t(std::max(content_region.y, 0.f))};
            }
        }
        const bool viewport_settled = new_logical_extent != uvec2{} && new_logical_extent == r.ctx().get<const ViewportExtent>().Value;

        // Remaining emits go after Interact so it wins the single-action buffer.
        // Start playback once settled, for play or video (the screenshot stays on the held frame).
        if (!playback_started && viewport_settled && (play || recording_mode)) {
            action::Emit(action::timeline::StartPresentation{});
            playback_started = true;
        }
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

        // Surface and clear any failures action handlers reported this frame.
        if (auto &errors = r.ctx().get<action::Errors>().Messages; !errors.empty()) {
            for (const auto &message : errors) std::cerr << message << std::endl;
            errors.clear();
        }

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
            // Save the image, then exit unless recording or a play duration is still running.
            if (screenshot_mode && !screenshot_saved && viewport_settled) {
                if (auto saved = SaveScreenshot(r, screenshot_path); saved) std::println("Saved screenshot: {}", saved->string());
                else std::println(stderr, "Screenshot: {}", saved.error());
                screenshot_saved = true;
                if (!recording_mode && play_duration <= 0) done = true;
            }
            if (recording_mode && !IsRecording(r, viewport) && viewport_settled) {
                StartRecording(r, viewport, record_path, record_fps);
                if (IsRecording(r, viewport)) next_capture_ns = SDL_GetTicksNS();
                else done = true;
            }
            if (IsRecording(r, viewport) && SDL_GetTicksNS() >= next_capture_ns) {
                CaptureRecordFrame(r, viewport);
                next_capture_ns += capture_interval_ns;
            }
            RenderFrame(*vc->Device, vc->Queue, wd, draw_data);
            PresentFrame(vc->Queue, wd);
        }
    }

    action::StopLog(); // Flush and join the log writer before teardown.
    vc->Device->waitIdle();

    audio_device.Uninit();
    NFD_Quit();

    // Tear down the viewport and its ctx-resident GPU stores in order before clearing the registry,
    // so GpuBuffers (and its VMA allocator) outlives the MeshStore allocations that retire into it.
    DeinitViewportMedia(r); // App-only media (icons/Faust/ImGui texture), while the device + GpuBuffers are alive.
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
} // namespace

int main(int argc, char **argv) {
    // VideoRecorder pipes frames to ffmpeg via popen; ignore SIGPIPE so writes return EPIPE instead of killing us.
    std::signal(SIGPIPE, SIG_IGN);

    std::set_terminate([]() {
        try {
            if (auto eptr = std::current_exception()) {
                std::rethrow_exception(eptr);
            }
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
    bool play = false;
    float play_duration = 0;
    fs::path record_path, screenshot_path;
    int record_fps = 60;
#ifdef QUIET
    bool quiet = true;
#else
    bool quiet = false;
#endif
    for (auto it = args.begin(); it != args.end(); ++it) {
        const std::string_view a{*it};
        if (a == "--quiet" || a == "-q") quiet = true;
        else if (a == "--play") {
            play = true;
            if (std::next(it) != args.end() && looks_numeric(*std::next(it))) play_duration = std::atof(*++it);
        } else if (a == "--record" && std::next(it) != args.end()) record_path = *++it;
        else if (a == "--screenshot" && std::next(it) != args.end()) screenshot_path = *++it;
        else if (a == "--fps" && std::next(it) != args.end()) record_fps = std::atoi(*++it);
        else if (!a.starts_with('-') && !initial_file) initial_file = *it;
    }
    if (record_fps <= 0) record_fps = 60;

    run(initial_file, quiet, play, play_duration, std::move(record_path), std::move(screenshot_path), record_fps);
    return 0;
}
