#include "Paths.h"
#include "Scene.h"
#include "Timer.h"
#include "Window.h"
#include "audio/AudioDevice.h"
#include "audio/AudioSystem.h"
#include "audio/FaustDSP.h"
#include "vulkan/VulkanContext.h"

#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "imgui_internal.h"
#include "implot.h"
#include "mesh/Primitives.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <entt/entity/registry.hpp>
#include <nfd.h>
#include <vulkan/vulkan_to_string.hpp>

#include <array>
#include <csignal>
#include <exception>
#include <format>
#include <iostream>

using std::ranges::any_of, std::ranges::distance, std::ranges::find_if;
namespace fs = std::filesystem;

// #define IMGUI_UNLIMITED_FRAME_RATE

namespace {
void CheckVk(vk::Result err) {
    if (err != vk::Result::eSuccess) throw std::runtime_error(std::format("Vulkan error: {}", vk::to_string(err)));
}

bool RebuildSwapchain = false;
void RenderFrame(vk::Device device, vk::Queue queue, ImGui_ImplVulkanH_Window &wd, ImDrawData *draw_data) {
    auto image_acquired_semaphore = wd.FrameSemaphores[wd.SemaphoreIndex].ImageAcquiredSemaphore;
    const auto err = device.acquireNextImageKHR(wd.Swapchain, UINT64_MAX, image_acquired_semaphore, nullptr, &wd.FrameIndex);
    if (err == vk::Result::eErrorOutOfDateKHR || err == vk::Result::eSuboptimalKHR) {
        RebuildSwapchain = true;
        return;
    }
    CheckVk(err);

    const auto &fd = wd.Frames[wd.FrameIndex];
    vk::Fence fd_fence{fd.Fence};
    CheckVk(device.waitForFences(fd_fence, true, UINT64_MAX));
    device.resetFences(fd_fence);
    device.resetCommandPool(fd.CommandPool);
    vk::CommandBuffer command_buffer{fd.CommandBuffer};
    command_buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    constexpr static vk::ClearValue clear_color{{0.45f, 0.55f, 0.60f, 1.f}};
    command_buffer.beginRenderPass({wd.RenderPass, fd.Framebuffer, {{0, 0}, {uint32_t(wd.Width), uint32_t(wd.Height)}}, 1, &clear_color}, vk::SubpassContents::eInline);
    // Record dear imgui primitives into command buffer
    ImGui_ImplVulkan_RenderDrawData(draw_data, fd.CommandBuffer);
    // Submit command buffer
    command_buffer.endRenderPass();
    command_buffer.end();

    vk::Semaphore wait_semaphores[]{image_acquired_semaphore};
    vk::PipelineStageFlags wait_stage{vk::PipelineStageFlagBits::eColorAttachmentOutput};
    vk::CommandBuffer command_buffers[]{command_buffer};
    vk::Semaphore signal_semaphores[]{wd.FrameSemaphores[wd.SemaphoreIndex].RenderCompleteSemaphore};
    queue.submit(vk::SubmitInfo{wait_semaphores, wait_stage, command_buffers, signal_semaphores}, fd_fence);
}
void PresentFrame(vk::Queue queue, ImGui_ImplVulkanH_Window &wd) {
    if (RebuildSwapchain) return;

    vk::Semaphore wait_semaphores[]{wd.FrameSemaphores[wd.SemaphoreIndex].RenderCompleteSemaphore};
    vk::SwapchainKHR swapchains[]{wd.Swapchain};
    uint32_t image_indices[]{wd.FrameIndex};
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

// Load a file into the scene based on its extension.
std::expected<void, std::string> LoadFile(Scene &scene, const fs::path &path) {
    const auto ext = path.extension().string();
    if (ext == ".gltf" || ext == ".glb") {
        if (auto result = scene.AddGltfScene(path); !result) {
            return std::unexpected(std::format("Error loading glTF file '{}': {}", path.string(), result.error()));
        }
    } else if (ext == ".obj" || ext == ".ply") {
        scene.AddMesh(path, MeshInstanceCreateInfo{.Name = path.stem().string()});
    } else {
        return std::unexpected(std::format("Unsupported file format: '{}'", ext));
    }
    return {};
}

void run(const char *initial_file, bool quiet, bool play, float play_duration, fs::path record_path, int record_fps) {
    Timer::Enabled = !quiet;

    SDL_SetHint(SDL_HINT_MAC_SCROLL_MOMENTUM, "1");

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        throw std::runtime_error(std::format("SDL_Init error: {}", SDL_GetError()));
    }
    if (const char *base = SDL_GetBasePath()) Paths::Init(base);
    else throw std::runtime_error(std::format("SDL_GetBasePath error: {}", SDL_GetError()));

    // Create window with Vulkan graphics context.
    auto *window = SDL_CreateWindow(
        "MeshEditor", 1280, 800,
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED | SDL_WINDOW_HIGH_PIXEL_DENSITY
    );

    uint extensions_count = 0;
    const char *const *instance_extensions_raw = SDL_Vulkan_GetInstanceExtensions(&extensions_count);
    auto vc = std::make_unique<VulkanContext>(std::vector<const char *>{instance_extensions_raw, instance_extensions_raw + extensions_count});

    constexpr static uint MinImageCount = 2;
    ImGui_ImplVulkanH_Window wd;
    // Set up Vulkan window.
    {
        VkSurfaceKHR surface;
        // Create window surface.
        if (!SDL_Vulkan_CreateSurface(window, *vc->Instance, nullptr, &surface)) throw std::runtime_error("Failed to create Vulkan surface.\n");
        wd.Surface = surface;

        int w, h;
        SDL_GetWindowSize(window, &w, &h);

        // Check for WSI support
        if (auto res = vc->PhysicalDevice.getSurfaceSupportKHR(vc->QueueFamily, wd.Surface); res != VK_TRUE) {
            throw std::runtime_error("Error no WSI support on physical device 0\n");
        }

        // Select surface format.
        const VkFormat requestSurfaceImageFormat[]{VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM};
        const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
        wd.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(vc->PhysicalDevice, wd.Surface, requestSurfaceImageFormat, (size_t)IM_COUNTOF(requestSurfaceImageFormat), requestSurfaceColorSpace);

        // Select present mode.
#ifdef IMGUI_UNLIMITED_FRAME_RATE
        VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR};
#else
        VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_FIFO_KHR};
#endif
        wd.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(vc->PhysicalDevice, wd.Surface, &present_modes[0], IM_COUNTOF(present_modes));
        ImGui_ImplVulkanH_CreateOrResizeWindow(*vc->Instance, vc->PhysicalDevice, *vc->Device, &wd, vc->QueueFamily, nullptr, w, h, MinImageCount, 0);
    }

    // Setup ImGui context.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    auto &io = GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad | ImGuiConfigFlags_DockingEnable;
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows
    io.IniFilename = nullptr; // Disable ImGui's .ini file saving

    StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplSDL3_InitForVulkan(window);
    ImGui_ImplVulkan_InitInfo init_info{
        .ApiVersion = VkApiVersion,
        .Instance = *vc->Instance,
        .PhysicalDevice = vc->PhysicalDevice,
        .Device = *vc->Device,
        .QueueFamily = vc->QueueFamily,
        .Queue = vc->Queue,
        .DescriptorPool = *vc->DescriptorPool,
        .DescriptorPoolSize = 0,
        .MinImageCount = MinImageCount,
        .ImageCount = wd.ImageCount,
        .PipelineCache = *vc->PipelineCache,
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
    auto scene = std::make_unique<Scene>(SceneVulkanResources{*vc->Instance, vc->PhysicalDevice, *vc->Device, vc->QueueFamily, vc->Queue}, r);

    // Load transform mode icons
    scene->LoadIcons();

    const auto CreateSvg = [device = *vc->Device, &scene, &wd](std::unique_ptr<SvgResource> &svg, fs::path path) {
        // Wait for previous frame's ImGui render to complete, since it may have sampled the old texture.
        CheckVk(device.waitForFences({wd.Frames[wd.FrameIndex].Fence}, true, UINT64_MAX));
        scene->CreateSvgResource(svg, std::move(path));
    };
    auto &faust_dsp = r.emplace<FaustDSP>(scene->GetSceneEntity(), CreateSvg);
    RegisterAudioComponentHandlers(r, scene->GetSceneEntity());

    struct AudioContext {
        FaustDSP *Dsp;
        entt::registry *R;
        entt::entity SceneEntity;
    };
    AudioContext audio_ctx{&faust_dsp, &r, scene->GetSceneEntity()};
    AudioDevice audio_device{
        {.Callback = [](auto buffer, void *user_data) {
             auto &ctx = *static_cast<AudioContext *>(user_data);
             ProcessAudio(*ctx.Dsp, *ctx.R, ctx.SceneEntity, std::move(buffer));
         },
         .UserData = &audio_ctx}
    };
    audio_device.Start();

    WindowsState windows;

    // Main loop.
    // Sim always uses the real `io.DeltaTime` from SDL, so sim-clock = wall-clock (1:1 real-time) whether
    // or not we're recording. Recording just samples the already-live viewport: a wall-clock accumulator
    // captures one frame every `1/record_fps` seconds, decimating if the display refreshes faster than the
    // video's fps. This keeps the in-app preview and the recorded video playing at the exact same rate —
    // the file is a faithful sample of what you saw on screen.
    const bool recording_mode = !record_path.empty();
    float elapsed_play_time = 0;
    const Uint64 capture_interval_ns = 1'000'000'000ULL / Uint64(record_fps);
    Uint64 next_capture_ns = 0; // Initialized when recording starts.
    bool done = false;
    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_MOUSE_WHEEL) {
                scene->PreciseWheelDelta += vec2{-event.wheel.x, event.wheel.y};
                // SDL's pixel-derived deltas overscroll ImGui panels.
                constexpr float ImGuiWheelScale = 0.3f;
                event.wheel.x *= ImGuiWheelScale;
                event.wheel.y *= ImGuiWheelScale;
            }
            ImGui_ImplSDL3_ProcessEvent(&event);
            done = event.type == SDL_EVENT_QUIT ||
                (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window));
        }
        const float elapsed = recording_mode ? float(scene->CapturedFrameCount()) / float(record_fps) : elapsed_play_time;
        if (play_duration > 0 && elapsed >= play_duration) done = true;

        // Resize swap chain?
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

        // Start the ImGui frame.
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        elapsed_play_time += io.DeltaTime;
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
            DockBuilderDockWindow(windows.Scene.Name, dockspace_id);
        }

        if (BeginMainMenuBar()) {
            if (BeginMenu("File")) {
                if (MenuItem("Load glTF", nullptr)) {
                    static const std::array filters{nfdfilteritem_t{"glTF scene", "gltf,glb"}};
                    nfdchar_t *nfd_path;
                    if (auto result = NFD_OpenDialog(&nfd_path, filters.data(), filters.size(), ""); result == NFD_OKAY) {
                        if (auto load = LoadFile(*scene, fs::path(nfd_path)); !load) std::cerr << load.error() << std::endl;
                        NFD_FreePath(nfd_path);
                    } else if (result != NFD_CANCEL) {
                        std::cerr << "Error opening file dialog: " << NFD_GetError() << std::endl;
                    }
                }
                if (MenuItem("Load OBJ/PLY", nullptr)) {
                    static const std::array filters{nfdfilteritem_t{"Mesh object", "obj,ply"}};
                    nfdchar_t *nfd_path;
                    if (auto result = NFD_OpenDialog(&nfd_path, filters.data(), filters.size(), ""); result == NFD_OKAY) {
                        if (auto load = LoadFile(*scene, fs::path(nfd_path)); !load) std::cerr << load.error() << std::endl;
                        NFD_FreePath(nfd_path);
                    } else if (result != NFD_CANCEL) {
                        std::cerr << "Error opening file dialog: " << NFD_GetError() << std::endl;
                    }
                }
                // if (MenuItem("Export mesh", nullptr, false, MainMesh != nullptr)) {
                //     nfdchar_t *path;
                //     if (auto result = NFD_SaveDialog(&path, filtes.data(), filters.size(), nullptr); result == NFD_OKAY) {
                //         scene->SaveMesh(fs::path(path));
                //         NFD_FreePath(path);
                //     } else if (result != NFD_CANCEL) {
                //         throw std::runtime_error(std::format("Error saving mesh file: {}", NFD_GetError()));
                //     }
                // }
                if (MenuItem("Load RealImpact", nullptr)) {
                    static const std::vector<nfdfilteritem_t> filters{};
                    nfdchar_t *path;
                    if (auto result = NFD_PickFolder(&path, ""); result == NFD_OKAY) {
                        scene->LoadRealImpact(fs::path{path});
                        NFD_FreePath(path);
                    } else if (result != NFD_CANCEL) {
                        throw std::runtime_error(std::format("Error loading RealImpact file: {}", NFD_GetError()));
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
                MenuItem(windows.Scene.Name, nullptr, &windows.Scene.Visible);
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
                    if (BeginTabItem("Scene")) {
                        SeparatorText("Buffer memory");
                        TextUnformatted(scene->DebugBufferHeapUsage().c_str());
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
                    scene->RenderControls();
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
                if (auto action = RenderAnimationTimeline(scene->GetTimeline(), scene->GetTimelineView(), scene->GetAnimationIcons())) {
                    scene->ApplyTimelineAction(*action);
                }
            }
            End();
            PopStyleVar();
        }

        if (windows.Scene.Visible) {
            PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
            if (Begin(windows.Scene.Name, &windows.Scene.Visible)) {
                scene->Interact();
                // Submit GPU render (nonblocking). WaitForRender() is called later, before RenderFrame() samples the final image.
                scene->Render(GetFrameCount() > 1 ? vk::Fence{wd.Frames[wd.FrameIndex].Fence} : vk::Fence{});
            }
            End();
            PopStyleVar();

            if (GetFrameCount() == 1) {
                // Initialize scene now that it has an extent.
                // static const auto DefaultRealImpactPath = fs::path{"../../"} / "RealImpact" / "dataset" / "22_Cup" / "preprocessed";
                // if (fs::exists(DefaultRealImpactPath)) scene->LoadRealImpact(DefaultRealImpactPath);
                if (initial_file) {
                    if (auto result = LoadFile(*scene, fs::path(initial_file)); !result) {
                        std::cerr << result.error() << std::endl;
                        play = false;
                    }
                } else {
                    constexpr PrimitiveShape default_shape{primitive::Cuboid{}};
                    const auto [mesh_entity, _] = scene->AddMesh(primitive::CreateMesh(default_shape), MeshInstanceCreateInfo{.Name = ToString(default_shape)});
                    r.emplace<PrimitiveShape>(mesh_entity, default_shape);
                }
            } else if (GetFrameCount() == 2 && play) {
                // Wait to play until after app load to avoid a long initial dt.
                scene->Play();
                play = false;
            }
        }

        ImGui::Render();
        auto *draw_data = GetDrawData();
        if (bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f); !is_minimized) {
            scene->WaitForRender(); // ImGui samples final image
            // Lazy-start recording once the viewport has rendered at least once (so FinalColorImage has a valid extent).
            if (recording_mode && !scene->IsRecording() && GetFrameCount() > 1) {
                scene->StartRecording(record_path, record_fps);
                if (scene->IsRecording()) next_capture_ns = SDL_GetTicksNS();
                else done = true;
            }
            if (scene->IsRecording() && SDL_GetTicksNS() >= next_capture_ns) {
                scene->CaptureRecordFrame();
                next_capture_ns += capture_interval_ns;
            }
            RenderFrame(*vc->Device, vc->Queue, wd, draw_data);
            PresentFrame(vc->Queue, wd);
        }
    }

    // Cleanup
    vc->Device->waitIdle();
    r.clear();

    audio_device.Uninit();
    NFD_Quit();

    scene.reset();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    ImGui_ImplVulkanH_DestroyWindow(*vc->Instance, *vc->Device, &wd, nullptr);
    vkDestroySurfaceKHR(*vc->Instance, wd.Surface, nullptr);
    wd.Surface = {};
    vc.reset();

    SDL_DestroyWindow(window);
    SDL_Quit();
}

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
    fs::path record_path;
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
        else if (a == "--fps" && std::next(it) != args.end()) record_fps = std::atoi(*++it);
        else if (!a.starts_with('-') && !initial_file) initial_file = *it;
    }
    if (record_fps <= 0) record_fps = 60;

    run(initial_file, quiet, play, play_duration, std::move(record_path), record_fps);
    return 0;
}
