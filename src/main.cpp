#include "Paths.h"
#include "ProcessEvents.h"
#include "Timer.h"
#include "TransformMath.h"
#include "Window.h"
#include "action/ActionApply.h"
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
#include "gizmo/TransformGizmoTypes.h"
#include "image/ImageEncode.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "object/ExtrasComponents.h"
#include "physics/PhysicsTypes.h"
#include "render/MaterialComponents.h"
#include "render/SvgResource.h"
#include "render/SvgUpload.h"
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
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <entt/entity/registry.hpp>
#include <nfd.h>

#include <csignal>
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

// Apply `action` now and settle the scene's derived state, for actions that must take effect outside the main loop.
template<typename ActionType> void Perform(entt::registry &r, entt::entity viewport, ActionType action) {
    action::Emit(std::move(action));
    action::ApplyEmitted(r, viewport);
    ProcessComponentEvents(r, viewport);
}

// Reset to the default scene, optionally replaying an action log on top.
void NewProject(entt::registry &r, entt::entity viewport, const fs::path &replay_path = {}, bool with_default_content = true) {
    // Called mid-frame: the GPU may still be using viewport resources we're about to recreate, with no fence to wait on.
    r.ctx().get<const VulkanResources>().Device.waitIdle();
    action::StopPlaybackIfPlaying(r, viewport);
    action::StopLog();
    const auto live_extent = r.ctx().get<ViewportExtent>().Value; // Restore after replay's SetExtent actions change it.
    // View-camera navigation isn't recorded, so save and restore afterward to not disturb the live view.
    auto live_view_cameras = GetViewCameraState(r, viewport);
    ClearScene(r, viewport);
    // An action-less log replays as an empty scene. Start the log before seeding/replaying so those actions re-record into it.
    action::StartLog();
    if (with_default_content) {
        action::Emit(action::io::LoadDefaultScene{});
    } else if (action::ReplayLog(r, viewport, replay_path, [](entt::registry &r, entt::entity viewport) { ProcessComponentEvents(r, viewport); })) {
        // Replay ran headless: restore the live extent and view camera, then present the final state.
        r.ctx().get<ViewportExtent>().Value = live_extent;
        SetViewCameraState(r, viewport, std::move(live_view_cameras));
        PresentViewport(r, viewport);
    }
}

#ifdef DEBUG_BUILD
// Validate replay then snapshot correctness, aborting on the first divergence:
// - Replay: replaying the current log onto a fresh scene must reproduce the saved image (writes a replay-test fixture on failure).
// - Round-trip: save, clear, restore must reproduce the saved image.
void ValidateRoundTrip(entt::registry &r, entt::entity viewport) {
    r.ctx().get<const VulkanResources>().Device.waitIdle();
    action::StopPlaybackIfPlaying(r, viewport);

    const auto logs = action::ListReplayLogs(); // Most-recent first; the newest is the live session's log.
    if (logs.empty()) {
        std::println(stderr, "[snapshot] replay SKIPPED (no log)");
    } else {
        const auto &current_log = logs.front().Path;
        const auto expected = snapshot::SnapshotSceneState(r);
        NewProject(r, viewport, current_log, /*with_default_content=*/false);
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

// Headless capture options from the CLI. `--render` is a preset for the full scene corpus; `--screenshot`/`--record` target one output.
struct CaptureRequest {
    bool Play{false};
    float PlayDuration{0}; // 0 = run until playback completes one loop.
    int Fps{60};
    fs::path RecordPath, ScreenshotPath;
    fs::path RenderBasename; // Output basename, no extension.
};

void run(const char *initial_file, bool quiet, bool empty, const CaptureRequest &capture) {
    Timer::Enabled = !quiet;

    bool play = capture.Play;
    const float play_duration = capture.PlayDuration;
    const bool render_mode = !capture.RenderBasename.empty();

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
    // Capture the DPI scale (only set during NewFrame) before priming DPI-scaled GPU state like edge-line width.
    ImGui_ImplSDL3_NewFrame();
    r.ctx().get<FrameState>().DisplayFramebufferScale = {io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y};
    ProcessComponentEvents(r, viewport); // Prime derived state before the first frame reads it.

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

    // Seed the scene's first recorded action: the initial file, else the default scene.
    if (initial_file) {
        if (const fs::path path = initial_file; path.extension() == ".actions") {
            NewProject(r, viewport, path, /*with_default_content=*/false);
        } else {
            Perform(r, viewport, action::io::Load{.Path = initial_file});
            if (auto &errors = r.ctx().get<action::Errors>().Messages; !errors.empty()) {
                for (const auto &message : errors) std::cerr << message << std::endl;
                errors.clear();
                play = false; // Don't auto-play if the initial file failed to load.
            }
        }
    } else if (!empty) {
        Perform(r, viewport, action::io::LoadDefaultScene{});
    }

    fs::path record_path = capture.RecordPath, screenshot_path = capture.ScreenshotPath, corpus_actions_path;
    if (render_mode) {
        const auto with = [&](const char *ext) { return fs::path{capture.RenderBasename.string() + ext}; };
        corpus_actions_path = with(".actions");
        const bool dynamic = r.view<const PhysicsMotion>().size() > 0 ||
            r.view<const ArmatureAnimation>().size() > 0 ||
            r.view<const NodeTransformAnimation>().size() > 0 ||
            r.view<const MorphWeightAnimation>().size() > 0;
        if (dynamic) record_path = with(".mp4");
        else screenshot_path = with(".png");
    }

    const bool recording_mode = !record_path.empty(), screenshot_mode = !screenshot_path.empty();
    const bool presenting = play || screenshot_mode || recording_mode;
    // Enter presentation mode so the first rendered frame matches the capture.
    // Playback start and the capture itself wait until the viewport settles.
    if (presenting) Perform(r, viewport, action::timeline::EnterPresentation{});

    // Interactively (including `--record`), sim-clock = wall-clock.
    // Recording samples the live viewport every `1/record_fps` seconds, so the video plays at the same rate as the in-app preview.
    // Render mode instead runs a fixed-step clock: one timeline frame per tick, and every tick is captured.
    r.ctx().get<FrameState>().FixedFrameStep = render_mode;
    const float timeline_fps = r.get<const TimelineRange>(viewport).Fps;
    const float render_dt = 1.f / timeline_fps;
    const int record_fps = render_mode ? int(std::lround(timeline_fps)) : capture.Fps;
    const uint64_t capture_interval_ns{1'000'000'000ULL / uint64_t(capture.Fps)};
    uint64_t next_capture_ns{0}; // Initialized when recording starts.
    float elapsed_play_time{0};
    uint32_t next_render_clip{1}; // Next clip to capture once the current loop finishes.
    uint32_t next_render_variant{0}; // Next material variant to capture once the default image saves.
    bool playback_started{false}, screenshot_saved{false}, view_framed{false};
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
            // Replay a dropped `.actions` log through NewProject; load any other file as a single action.
            if (event.type == SDL_EVENT_DROP_FILE) {
                if (const fs::path dropped = event.drop.data; dropped.extension() == ".actions") NewProject(r, viewport, dropped, /*with_default_content=*/false);
                else action::Emit(action::io::Load{.Path = dropped});
            }
            // SDL3 backend invalidates MousePos to -FLT_MAX on leave when no mouse button is held,
            // which flings a keyboard-initiated (G/R/S) transform offscreen when switching focus.
            if (event.type != SDL_EVENT_WINDOW_MOUSE_LEAVE || !TransformGizmo::IsUsing(r, viewport)) {
                ImGui_ImplSDL3_ProcessEvent(&event);
                done = event.type == SDL_EVENT_QUIT || (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window));
            }
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
        // Scene-affecting code reads FrameState::DeltaTime. `io.DeltaTime` is wall-clock, UI-only.
        r.ctx().get<FrameState>().DeltaTime = render_mode ? render_dt : io.DeltaTime;
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
                // The `.state` document Save writes to without prompting. Empty until the scene is opened from or saved to a file.
                static fs::path current_state_path;
                if (BeginMenu("New")) {
                    if (MenuItem("Default")) {
                        current_state_path.clear();
                        NewProject(r, viewport);
                    }
                    if (MenuItem("Empty")) {
                        current_state_path.clear();
                        NewProject(r, viewport, {}, /*with_default_content=*/false);
                    }
                    EndMenu();
                }
                if (MenuItem("Open")) {
                    static constexpr std::array filters{nfdfilteritem_t{"Scene state", "state"}, nfdfilteritem_t{"Action log", "actions"}};
                    nfdchar_t *nfd_path;
                    if (auto result = NFD_OpenDialog(&nfd_path, filters.data(), filters.size(), ""); result == NFD_OKAY) {
                        if (const fs::path path = nfd_path; path.extension() == ".actions") {
                            current_state_path.clear();
                            NewProject(r, viewport, path, /*with_default_content=*/false);
                        } else {
                            current_state_path = path;
                            action::Emit(action::io::Load{.Path = path});
                        }
                        NFD_FreePath(nfd_path);
                    } else if (result != NFD_CANCEL) {
                        std::cerr << "Error opening file dialog: " << NFD_GetError() << std::endl;
                    }
                }
                const auto save_state_as = [&] {
                    static const std::array filters{nfdfilteritem_t{"Scene state", "state"}};
                    nfdchar_t *nfd_path;
                    if (auto result = NFD_SaveDialog(&nfd_path, filters.data(), filters.size(), nullptr, "scene.state"); result == NFD_OKAY) {
                        current_state_path = nfd_path;
                        if (current_state_path.extension() != ".state") current_state_path += ".state"; // NFD doesn't force the filter's extension.
                        action::Emit(action::io::SaveState{.Path = current_state_path});
                        NFD_FreePath(nfd_path);
                    } else if (result != NFD_CANCEL) {
                        std::cerr << "Error opening save dialog: " << NFD_GetError() << std::endl;
                    }
                };
                if (MenuItem("Save")) {
                    if (current_state_path.empty()) save_state_as();
                    else action::Emit(action::io::SaveState{.Path = current_state_path});
                }
                if (MenuItem("Save as...")) save_state_as();
                if (BeginMenu("Replay")) {
                    const auto logs = action::ListReplayLogs(); // Most-recent first; the newest is the live session's log.
                    for (size_t i = 0; i < logs.size(); ++i) {
                        const std::time_t t = logs[i].UnixSeconds;
                        char date[32];
                        std::strftime(date, sizeof date, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
                        const auto label = i == 0 ? std::format("Current ({})", date) : std::string{date};
                        if (MenuItem(label.c_str())) NewProject(r, viewport, logs[i].Path, /*with_default_content=*/false);
                    }
                    EndMenu();
                }
                const auto import_dialog = [](const auto &filters) {
                    nfdchar_t *nfd_path;
                    if (auto result = NFD_OpenDialog(&nfd_path, filters.data(), filters.size(), ""); result == NFD_OKAY) {
                        action::Emit(action::io::Load{.Path = nfd_path});
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
                            action::Emit(action::io::LoadRealImpact{.Directory = path});
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
                        action::Emit(action::io::SaveGltf{.Path = nfd_path});
                        NFD_FreePath(nfd_path);
                    } else if (result != NFD_CANCEL) {
                        std::cerr << "Error opening save dialog: " << NFD_GetError() << std::endl;
                    }
                }
#ifdef DEBUG_BUILD
                if (MenuItem("[Debug] Roundtrip")) ValidateRoundTrip(r, viewport);
#endif
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
        if (!view_framed && presenting && r.view<const Camera>().empty() && new_logical_extent != uvec2{}) {
            FrameScene(r, viewport, float(new_logical_extent.x) / float(new_logical_extent.y));
            view_framed = viewport_settled;
        }

        // Remaining emits go after Interact so it wins the single-action buffer.
        // Start playback once settled, for play or video (the screenshot stays on the held frame).
        // Render mode waits one more tick, until recording has begun, so the start frame is captured.
        if (!playback_started && viewport_settled && (play || (render_mode ? IsRecording(r, viewport) : recording_mode))) {
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
                // After the default, save one image per material variant.
                const auto *mv = render_mode ? r.try_get<const MaterialVariants>(viewport) : nullptr;
                if (mv && next_render_variant < mv->Names.size()) {
                    auto name = mv->Names[next_render_variant].empty() ? std::format("Variant {}", next_render_variant) : mv->Names[next_render_variant];
                    std::ranges::replace(name, '/', '-');
                    screenshot_path = fs::path{capture.RenderBasename.string() + "." + name + ".png"};
                    action::Emit(action::UpdateOf<&MaterialVariants::Active>(viewport, std::optional{next_render_variant}));
                    ++next_render_variant;
                } else {
                    screenshot_saved = true;
                    if (!recording_mode && play_duration <= 0) done = true;
                }
            }
            if (recording_mode && !IsRecording(r, viewport) && viewport_settled) {
                StartRecording(r, viewport, record_path, record_fps);
                if (IsRecording(r, viewport)) next_capture_ns = SDL_GetTicksNS();
                else done = true;
            }
            if (IsRecording(r, viewport)) {
                if (render_mode) {
                    // Fixed step: every tick is one timeline frame; capture each. At a loop's last frame,
                    // switch each multi-clip component to its next clip, or stop after the last.
                    // Emitted, not Performed: a mid-loop Perform would advance playback an extra tick.
                    CaptureRecordFrame(r, viewport);
                    if (r.get<const TimelinePlayback>(viewport).CurrentFrame == r.get<const TimelineRange>(viewport).EndFrame) {
                        bool switched = false;
                        const auto switch_clips = [&]<typename Anim>() {
                            for (const auto [entity, anim] : r.view<const Anim>().each()) {
                                if (next_render_clip < anim.Clips.size()) {
                                    action::Emit(action::UpdateOf<&Anim::ActiveClipIndex>(entity, next_render_clip));
                                    switched = true;
                                }
                            }
                        };
                        switch_clips.template operator()<ArmatureAnimation>();
                        switch_clips.template operator()<MorphWeightAnimation>();
                        switch_clips.template operator()<NodeTransformAnimation>();
                        if (switched) ++next_render_clip;
                        else done = true;
                    }
                } else if (SDL_GetTicksNS() >= next_capture_ns) {
                    CaptureRecordFrame(r, viewport);
                    next_capture_ns += capture_interval_ns;
                }
            }
            RenderFrame(*vc->Device, vc->Queue, wd, draw_data);
            PresentFrame(vc->Queue, wd);
        }
    }

    if (const fs::path session_log = action::StopLog(); !corpus_actions_path.empty() && !session_log.empty()) {
        std::error_code ec;
        fs::copy_file(session_log, corpus_actions_path, fs::copy_options::overwrite_existing, ec);
    }
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
    bool empty = false;
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
        else if (a == "--empty") empty = true;
        else if (a == "--fps" && std::next(it) != args.end()) capture.Fps = std::atoi(*++it);
        else if (!a.starts_with('-') && !initial_file) initial_file = *it;
    }
    if (capture.Fps <= 0) capture.Fps = 60;

    run(initial_file, quiet, empty, capture);
    return 0;
}
