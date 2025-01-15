#include "Widgets.h" // imgui

#include "Scene.h"
#include "SvgResource.h"
#include "Window.h"
#include "audio/AcousticScene.h"
#include "audio/AudioDevice.h"
#include "audio/SoundObject.h"
#include "numeric/vec4.h"
#include "vulkan/VulkanContext.h"

#include "imgui_internal.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "implot.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <entt/entity/registry.hpp>
#include <nfd.h>

#include <format>
#include <ranges>
#include <stack>
#include <stdexcept>

using std::ranges::to;

// #define IMGUI_UNLIMITED_FRAME_RATE

namespace {
constexpr uint MinImageCount = 2;
bool SwapChainRebuild = false;

std::unique_ptr<VulkanContext> VC;

void SetupVulkanWindow(ImGui_ImplVulkanH_Window &wd, vk::SurfaceKHR surface, int width, int height) {
    wd.Surface = surface;

    // Check for WSI support
    if (auto res = VC->PhysicalDevice.getSurfaceSupportKHR(VC->QueueFamily, wd.Surface); res != VK_TRUE) {
        throw std::runtime_error("Error no WSI support on physical device 0\n");
    }

    // Select surface format.
    const VkFormat requestSurfaceImageFormat[]{VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM};
    const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    wd.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(VC->PhysicalDevice, wd.Surface, requestSurfaceImageFormat, (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat), requestSurfaceColorSpace);

    // Select present mode.
#ifdef IMGUI_UNLIMITED_FRAME_RATE
    VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR};
#else
    VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_FIFO_KHR};
#endif
    wd.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(VC->PhysicalDevice, wd.Surface, &present_modes[0], IM_ARRAYSIZE(present_modes));

    ImGui_ImplVulkanH_CreateOrResizeWindow(*VC->Instance, VC->PhysicalDevice, *VC->Device, &wd, VC->QueueFamily, nullptr, width, height, MinImageCount);
}

void CheckVk(VkResult err) {
    if (err != 0) throw std::runtime_error(std::format("Vulkan error: {}", int(err)));
}

void FrameRender(ImGui_ImplVulkanH_Window &wd, ImDrawData *draw_data) {
    auto image_acquired_semaphore = wd.FrameSemaphores[wd.SemaphoreIndex].ImageAcquiredSemaphore;
    auto render_complete_semaphore = wd.FrameSemaphores[wd.SemaphoreIndex].RenderCompleteSemaphore;
    const auto err = vkAcquireNextImageKHR(*VC->Device, wd.Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &wd.FrameIndex);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        SwapChainRebuild = true;
        return;
    }
    CheckVk(err);

    auto *fd = &wd.Frames[wd.FrameIndex];
    {
        CheckVk(vkWaitForFences(*VC->Device, 1, &fd->Fence, VK_TRUE, UINT64_MAX)); // wait indefinitely instead of periodically checking
        CheckVk(vkResetFences(*VC->Device, 1, &fd->Fence));
    }
    {
        CheckVk(vkResetCommandPool(*VC->Device, fd->CommandPool, 0));
        VkCommandBufferBeginInfo info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        };
        CheckVk(vkBeginCommandBuffer(fd->CommandBuffer, &info));
    }
    {
        VkRenderPassBeginInfo info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = wd.RenderPass,
            .framebuffer = fd->Framebuffer,
            .renderArea = {{0, 0}, {uint32_t(wd.Width), uint32_t(wd.Height)}},
            .clearValueCount = 1,
            .pClearValues = &wd.ClearValue,
        };
        vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
    }
    // Record dear imgui primitives into command buffer
    ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);
    // Submit command buffer
    vkCmdEndRenderPass(fd->CommandBuffer);
    {
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo info{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &image_acquired_semaphore,
            .pWaitDstStageMask = &wait_stage,
            .commandBufferCount = 1,
            .pCommandBuffers = &fd->CommandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &render_complete_semaphore,
        };
        CheckVk(vkEndCommandBuffer(fd->CommandBuffer));
        CheckVk(vkQueueSubmit(VC->Queue, 1, &info, fd->Fence));
    }
}

void FramePresent(ImGui_ImplVulkanH_Window &wd) {
    if (SwapChainRebuild) return;

    VkPresentInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wd.FrameSemaphores[wd.SemaphoreIndex].RenderCompleteSemaphore,
        .swapchainCount = 1,
        .pSwapchains = &wd.Swapchain,
        .pImageIndices = &wd.FrameIndex,
        .pResults = nullptr,
    };
    auto err = vkQueuePresentKHR(VC->Queue, &info);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        SwapChainRebuild = true;
        return;
    }
    CheckVk(err);
    wd.SemaphoreIndex = (wd.SemaphoreIndex + 1) % wd.SemaphoreCount; // Now we can use the next set of semaphores.
}

void CreateSvg(std::unique_ptr<SvgResource> &svg, fs::path path) {
    VC->Device->waitIdle();
    svg.reset(); // Ensure destruction before creation.
    svg = std::make_unique<SvgResource>(*VC, std::move(path));
};

using namespace ImGui;

enum class FontFamily {
    Main,
    Monospace,
};

constexpr float FontAtlasScale = 2; // Rasterize to a scaled-up texture and scale down the font size globally, for sharper text.
ImFont *MainFont{nullptr}, *MonospaceFont{nullptr};
ImFont *AddFont(FontFamily family, const std::string_view font_file) {
    static const auto FontsPath = fs::path("./") / "res" / "fonts";
    // These are eyeballed.
    static const std::unordered_map<FontFamily, uint> PixelsForFamily{
        {FontFamily::Main, 15},
        {FontFamily::Monospace, 17},
    };

    return GetIO().Fonts->AddFontFromFileTTF((FontsPath / font_file).c_str(), PixelsForFamily.at(family) * FontAtlasScale);
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

entt::registry R;
void AudioCallback(AudioBuffer buffer) {
    for (const auto &audio_source : R.storage<SoundObject>()) {
        audio_source.ProduceAudio(buffer);
    }
}

int main(int, char **) {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        throw std::runtime_error(std::format("SDL_Init error: {}", SDL_GetError()));
    }

    // Create window with Vulkan graphics context.
    auto *window = SDL_CreateWindow(
        "MeshEditor", 1280, 720,
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED | SDL_WINDOW_HIGH_PIXEL_DENSITY
    );

    uint extensions_count = 0;
    const char *const *instance_extensions_raw = SDL_Vulkan_GetInstanceExtensions(&extensions_count);
    VC = std::make_unique<VulkanContext>(std::vector<const char *>{instance_extensions_raw, instance_extensions_raw + extensions_count});

    // Create window surface.
    VkSurfaceKHR surface;
    if (!SDL_Vulkan_CreateSurface(window, *VC->Instance, nullptr, &surface)) throw std::runtime_error("Failed to create Vulkan surface.\n");

    // Create framebuffers.
    int w, h;
    SDL_GetWindowSize(window, &w, &h);

    ImGui_ImplVulkanH_Window wd;
    SetupVulkanWindow(wd, surface, w, h);

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
        .Instance = *VC->Instance,
        .PhysicalDevice = VC->PhysicalDevice,
        .Device = *VC->Device,
        .QueueFamily = VC->QueueFamily,
        .Queue = VC->Queue,
        .DescriptorPool = *VC->DescriptorPool,
        .RenderPass = wd.RenderPass,
        .MinImageCount = MinImageCount,
        .ImageCount = wd.ImageCount,
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        .PipelineCache = *VC->PipelineCache,
        .Subpass = 0,
        .DescriptorPoolSize = 0,
        .UseDynamicRendering = false,
        .PipelineRenderingCreateInfo = {},
        .Allocator = nullptr,
        .CheckVkResultFn = CheckVk,
        .MinAllocationSize = {},
    };
    ImGui_ImplVulkan_Init(&init_info);

    InitFonts();

    NFD_Init();

    // EnTT listeners
    R.on_construct<ExcitedVertex>().connect<[](entt::registry &r, entt::entity entity) {
        const auto &excited_vertex = r.get<ExcitedVertex>(entity);
        if (auto *sound_object = r.try_get<SoundObject>(entity)) {
            sound_object->Apply(SoundObjectAction::Excite{excited_vertex.Vertex, excited_vertex.Force});
        }
    }>();
    R.on_destroy<ExcitedVertex>().connect<[](entt::registry &r, entt::entity entity) {
        if (auto *sound_object = r.try_get<SoundObject>(entity)) {
            sound_object->Apply(SoundObjectAction::SetExciteForce{0.f});
        }
    }>();

    std::unique_ptr<Scene> scene = std::make_unique<Scene>(*VC, R);
    std::unique_ptr<AcousticScene> acoustic_scene = std::make_unique<AcousticScene>(R);
    std::unique_ptr<ImGuiTexture> scene_texture;

    AudioDevice audio_device{AudioCallback};
    audio_device.Start();

    WindowsState windows;

    // Main loop
    bool done = false;
    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            done = event.type == SDL_EVENT_QUIT ||
                (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window));
        }

        // Resize swap chain?
        if (SwapChainRebuild) {
            int width, height;
            SDL_GetWindowSize(window, &width, &height);
            if (width > 0 && height > 0) {
                ImGui_ImplVulkan_SetMinImageCount(MinImageCount);
                ImGui_ImplVulkanH_CreateOrResizeWindow(*VC->Instance, VC->PhysicalDevice, *VC->Device, &wd, VC->QueueFamily, nullptr, width, height, MinImageCount);
                wd.FrameIndex = 0;
                SwapChainRebuild = false;
            }
        }

        // Start the ImGui frame.
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        NewFrame();

        auto dockspace_id = DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_AutoHideTabBar);
        if (GetFrameCount() == 1) {
            auto controls_node_id = DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.35f, nullptr, &dockspace_id);
            auto demo_node_id = DockBuilderSplitNode(controls_node_id, ImGuiDir_Down, 0.4f, nullptr, &controls_node_id);
            DockBuilderDockWindow(windows.ImGuiDemo.Name, demo_node_id);
            DockBuilderDockWindow(windows.ImPlotDemo.Name, demo_node_id);
            DockBuilderDockWindow(windows.SceneControls.Name, controls_node_id);
            DockBuilderDockWindow(windows.Scene.Name, dockspace_id);
        }

        if (BeginMainMenuBar()) {
            if (BeginMenu("File")) {
                if (MenuItem("Load mesh", nullptr)) {
                    static const std::vector<nfdfilteritem_t> filters{{"Mesh object", "obj,off,ply,stl,om"}};
                    nfdchar_t *nfd_path;
                    if (auto result = NFD_OpenDialog(&nfd_path, filters.data(), filters.size(), ""); result == NFD_OKAY) {
                        const auto path = fs::path(nfd_path);
                        scene->AddMesh(path, {.Name = path.filename().string()});
                        NFD_FreePath(nfd_path);
                    } else if (result != NFD_CANCEL) {
                        throw std::runtime_error(std::format("Error loading mesh file: {}", NFD_GetError()));
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
                        AcousticScene::LoadRealImpact(fs::path{path}, R, *scene, CreateSvg);
                        NFD_FreePath(path);
                    } else if (result != NFD_CANCEL) {
                        throw std::runtime_error(std::format("Error loading RealImpact file: {}", NFD_GetError()));
                    }
                }
                EndMenu();
            }
            if (BeginMenu("Windows")) {
                MenuItem(windows.ImGuiDemo.Name, nullptr, &windows.ImGuiDemo.Visible);
                MenuItem(windows.ImPlotDemo.Name, nullptr, &windows.ImPlotDemo.Visible);
                MenuItem(windows.SceneControls.Name, nullptr, &windows.SceneControls.Visible);
                MenuItem(windows.Scene.Name, nullptr, &windows.Scene.Visible);
                EndMenu();
            }
            EndMainMenuBar();
        }

        if (windows.ImGuiDemo.Visible) ImGui::ShowDemoWindow(&windows.ImGuiDemo.Visible);
        if (windows.ImPlotDemo.Visible) ImPlot::ShowDemoWindow(&windows.ImPlotDemo.Visible);

        if (windows.SceneControls.Visible) {
            Begin(windows.SceneControls.Name, &windows.SceneControls.Visible);
            if (BeginTabBar("Controls")) {
                if (BeginTabItem("Scene")) {
                    scene->RenderControls();
                    EndTabItem();
                }
                if (BeginTabItem("Acoustic scene")) {
                    acoustic_scene->RenderControls(*scene);
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

        if (windows.Scene.Visible) {
            PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
            Begin(windows.Scene.Name, &windows.Scene.Visible);
            if (scene->Render()) {
                // Extent changed. Update the scene texture.
                scene_texture.reset(); // Ensure destruction before creation.
                scene_texture = std::make_unique<ImGuiTexture>(*VC->Device, scene->GetResolveImageView(), vec2{0, 1}, vec2{1, 0});
            }

            const auto &cursor = GetCursorPos();
            if (scene_texture) {
                const auto &scene_extent = scene->GetExtent();
                scene_texture->Render({float(scene_extent.width), float(scene_extent.height)});
            }
            SetCursorPos(cursor);
            scene->RenderGizmo();
            End();
            PopStyleVar();

            if (GetFrameCount() == 1) {
                // Initialize scene now that it has an extent.
                static const auto DefaultRealImpactPath = fs::path("../../") / "RealImpact" / "dataset" / "22_Cup" / "preprocessed";
                if (fs::exists(DefaultRealImpactPath)) AcousticScene::LoadRealImpact(DefaultRealImpactPath, R, *scene, CreateSvg);
            }
        }

        // Render
        ImGui::Render();
        auto *draw_data = GetDrawData();
        if (bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f); !is_minimized) {
            static constexpr vec4 ClearColor{0.45f, 0.55f, 0.60f, 1.f};
            wd.ClearValue.color.float32[0] = ClearColor.r * ClearColor.a;
            wd.ClearValue.color.float32[1] = ClearColor.g * ClearColor.a;
            wd.ClearValue.color.float32[2] = ClearColor.b * ClearColor.a;
            wd.ClearValue.color.float32[3] = ClearColor.a;
            FrameRender(wd, draw_data);
            FramePresent(wd);
        }
    }

    audio_device.Uninit();

    // Cleanup
    NFD_Quit();

    VC->Device->waitIdle();

    R.clear();
    scene_texture.reset();
    scene.reset();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();

    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    ImGui_ImplVulkanH_DestroyWindow(*VC->Instance, *VC->Device, &wd, nullptr);
    VC.reset();

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
