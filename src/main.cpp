#include "Scene.h"
#include "SvgResource.h"
#include "Widgets.h" // imgui
#include "Window.h"
#include "audio/AcousticScene.h"
#include "audio/AudioDevice.h"
#include "numeric/vec4.h"

#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "imgui_internal.h"
#include "implot.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <entt/entity/registry.hpp>
#include <nfd.h>
#include <vulkan/vulkan_to_string.hpp>

#include <array>
#include <format>
#include <iostream>
#include <numeric>
#include <ranges>
#include <stack>
#include <stdexcept>

using std::ranges::any_of, std::ranges::distance, std::ranges::find_if, std::ranges::to;

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
    CheckVk(device.waitForFences(fd_fence, true, UINT64_MAX)); // wait indefinitely instead of periodically checking
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
    static const auto FontsPath = fs::path("./") / "res" / "fonts";
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

// Find a discrete GPU, or the first available (integrated) GPU.
vk::PhysicalDevice FindPhysicalDevice(const vk::UniqueInstance &instance) {
    const auto physical_devices = instance->enumeratePhysicalDevices();
    if (physical_devices.empty()) throw std::runtime_error("No Vulkan devices found.");

    for (const auto &device : physical_devices) {
        if (device.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu) return device;
    }
    return physical_devices[0];
}

constexpr auto VkApiVersion = VK_API_VERSION_1_4;

struct VulkanContext {
    VulkanContext(std::vector<const char *> enabled_extensions) {
        const auto IsLayerAvailable = [&](std::string_view layer) {
            static const auto available_layers = vk::enumerateInstanceLayerProperties();
            return any_of(available_layers, [layer](const auto &prop) { return strcmp(prop.layerName, layer.data()) == 0; });
        };
        std::vector<const char *> enabled_layers;
        const auto AddLayerIfAvailable = [&](std::string_view layer) {
            if (IsLayerAvailable(layer)) {
                enabled_layers.push_back(layer.data());
            } else {
                std::cerr << "Warning: Validation layer " << layer << " not available." << std::endl;
            }
        };
        AddLayerIfAvailable("VK_LAYER_KHRONOS_validation");

        const auto instance_props = vk::enumerateInstanceExtensionProperties();
        const auto IsExtensionAvailable = [&](std::string_view extension) {
            return any_of(instance_props, [extension](const auto &prop) { return strcmp(prop.extensionName, extension.data()) == 0; });
        };

        vk::InstanceCreateFlags flags;
        const auto AddExtensionIfAvailable = [&](std::string_view extension, vk::InstanceCreateFlags flag = {}) {
            if (IsExtensionAvailable(extension)) {
                enabled_extensions.push_back(extension.data());
                flags |= flag;
                return true;
            }
            std::cerr << "Warning: Extension " << extension << " not available." << std::endl;
            return false;
        };
        AddExtensionIfAvailable(vk::KHRPortabilityEnumerationExtensionName, vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR);
        AddExtensionIfAvailable(vk::EXTDebugUtilsExtensionName);

        const vk::ApplicationInfo app{"", {}, "", {}, VkApiVersion};
        Instance = vk::createInstanceUnique({flags, &app, enabled_layers, enabled_extensions});
        PhysicalDevice = FindPhysicalDevice(Instance);

        const auto qfp = PhysicalDevice.getQueueFamilyProperties();
        const auto qfp_find_graphics_it = find_if(qfp, [](const auto &qfp) { return bool(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });
        if (qfp_find_graphics_it == qfp.end()) throw std::runtime_error("No graphics queue family found.");

        QueueFamily = distance(qfp.begin(), qfp_find_graphics_it);
        if (!PhysicalDevice.getFeatures().fillModeNonSolid) {
            throw std::runtime_error("`fillModeNonSolid` is not supported, but is needed for line rendering.");
        }

        vk::PhysicalDeviceFeatures device_features{};
        device_features.fillModeNonSolid = VK_TRUE;

        // Create logical device (with one queue).
        static const std::vector<const char *> device_extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME, "VK_KHR_portability_subset"};
        static constexpr std::array queue_priority{1.0f};
        const vk::DeviceQueueCreateInfo queue_info{{}, QueueFamily, 1, queue_priority.data()};
        Device = PhysicalDevice.createDeviceUnique({{}, queue_info, {}, device_extensions, &device_features});
        Queue = Device->getQueue(QueueFamily, 0);

        // Create descriptor pool.
        const std::vector pool_sizes{
            // Image samplers:
            // 1) (2D filled shape) silhouette of the active mesh
            // 2) (2D line) silhouette of the active mesh, after edge detection
            // 3) Final scene texture sampler
            // 4) ImGui fonts
            // 5) SVG image texture
            // 6-9) SVG icons (move/rotate/scale/universal)
            vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 9},
            // All uniform buffer descriptors used across all shaders.
            {vk::DescriptorType::eUniformBuffer, 7},
        };
        const uint max_sets = std::accumulate(
            pool_sizes.begin(), pool_sizes.end(), 0u,
            [](uint sum, const vk::DescriptorPoolSize &pool_size) { return sum + pool_size.descriptorCount; }
        );
        DescriptorPool = Device->createDescriptorPoolUnique({vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, max_sets, pool_sizes});
    }
    ~VulkanContext() = default; // Using unique handles, so no need to manually destroy anything.

    vk::UniqueInstance Instance;
    vk::PhysicalDevice PhysicalDevice;
    vk::UniqueDevice Device;

    uint QueueFamily{uint(-1)};
    vk::Queue Queue;
    vk::UniquePipelineCache PipelineCache;
    vk::UniqueDescriptorPool DescriptorPool;
};

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

int main(int, char **) {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        throw std::runtime_error(std::format("SDL_Init error: {}", SDL_GetError()));
    }

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
        wd.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(vc->PhysicalDevice, wd.Surface, requestSurfaceImageFormat, (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat), requestSurfaceColorSpace);

        // Select present mode.
#ifdef IMGUI_UNLIMITED_FRAME_RATE
        VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR};
#else
        VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_FIFO_KHR};
#endif
        wd.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(vc->PhysicalDevice, wd.Surface, &present_modes[0], IM_ARRAYSIZE(present_modes));
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
            .PipelineRenderingCreateInfo = {},
            .SwapChainImageUsage = {},
        },
        .PipelineInfoForViewports = {},
        .UseDynamicRendering = false,
        .Allocator = nullptr,
        .CheckVkResultFn = [](VkResult err) {
            if (err != 0) throw std::runtime_error(std::format("Vulkan error: {}", int(err)));
        },
        .MinAllocationSize = {},
        .CustomShaderVertCreateInfo = {},
        .CustomShaderFragCreateInfo = {},
    };
    ImGui_ImplVulkan_Init(&init_info);

    InitFonts();

    NFD_Init();
    entt::registry r;
    std::unique_ptr<Scene> scene = std::make_unique<Scene>(
        SceneVulkanResources{
            *vc->Instance,
            vc->PhysicalDevice,
            *vc->Device,
            vc->QueueFamily,
            vc->Queue,
            *vc->DescriptorPool,
        },
        r
    );
    auto device = *vc->Device;

    // Load transform mode icons
    scene->LoadIcons(device);

    const auto CreateSvg = [device, &scene](std::unique_ptr<SvgResource> &svg, fs::path path) {
        const auto RenderBitmap = [&scene](std::span<const std::byte> data, uint32_t width, uint32_t height) {
            return scene->RenderBitmapToImage(data, width, height);
        };
        device.waitIdle();
        svg.reset(); // Ensure destruction before creation.
        svg = std::make_unique<SvgResource>(device, RenderBitmap, std::move(path));
    };

    auto acoustic_scene = std::make_unique<AcousticScene>(r, CreateSvg);
    AudioDevice audio_device{
        {.Callback = [](auto buffer, void *user_data) {
             const auto *acoustic_scene = static_cast<const AcousticScene *>(user_data);
             acoustic_scene->Process(std::move(buffer));
         },
         .UserData = acoustic_scene.get()}
    };
    audio_device.Start();

    std::unique_ptr<mvk::ImGuiTexture> scene_viewport_texture;
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
        NewFrame();

        auto dockspace_id = DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_AutoHideTabBar);
        if (GetFrameCount() == 1) {
            auto controls_node_id = DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.35f, nullptr, &dockspace_id);
            auto extra_node_id = DockBuilderSplitNode(controls_node_id, ImGuiDir_Down, 0.4f, nullptr, &controls_node_id);
            DockBuilderDockWindow(windows.Debug.Name, extra_node_id);
            DockBuilderDockWindow(windows.ImGuiDemo.Name, extra_node_id);
            DockBuilderDockWindow(windows.ImPlotDemo.Name, extra_node_id);
            DockBuilderDockWindow(windows.SceneControls.Name, controls_node_id);
            DockBuilderDockWindow(windows.Scene.Name, dockspace_id);
        }

        if (BeginMainMenuBar()) {
            if (BeginMenu("File")) {
                if (MenuItem("Load mesh", nullptr)) {
                    static const std::array filters{nfdfilteritem_t{"Mesh object", "obj,ply"}};
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
                        acoustic_scene->LoadRealImpact(fs::path{path}, *scene);
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
            if (Begin(windows.Scene.Name, &windows.Scene.Visible)) {
                scene->Interact();
                if (scene->RenderViewport()) {
                    // Extent changed. Update the scene texture.
                    scene_viewport_texture.reset(); // Ensure destruction before creation.
                    scene_viewport_texture = std::make_unique<mvk::ImGuiTexture>(*vc->Device, scene->GetViewportImageView(), vec2{0, 1}, vec2{1, 0});
                }
                if (scene_viewport_texture) {
                    const auto cursor = GetCursorPos();
                    const auto &scene_extent = scene->GetExtent();
                    scene_viewport_texture->Render({float(scene_extent.width), float(scene_extent.height)});
                    SetCursorPos(cursor);
                }
                scene->RenderOverlay();
            }
            End();
            PopStyleVar();

            if (GetFrameCount() == 1) {
                // Initialize scene now that it has an extent.
                // static const auto DefaultRealImpactPath = fs::path("../../") / "RealImpact" / "dataset" / "22_Cup" / "preprocessed";
                // if (fs::exists(DefaultRealImpactPath)) acoustic_scene->LoadRealImpact(DefaultRealImpactPath, *scene);
            }
        }

        ImGui::Render();
        auto *draw_data = GetDrawData();
        if (bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f); !is_minimized) {
            RenderFrame(*vc->Device, vc->Queue, wd, draw_data);
            PresentFrame(vc->Queue, wd);
        }
    }

    // Cleanup
    vc->Device->waitIdle();
    r.clear();

    audio_device.Uninit();
    acoustic_scene.reset();
    NFD_Quit();

    scene_viewport_texture.reset();
    scene.reset();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    ImGui_ImplVulkanH_DestroyWindow(*vc->Instance, *vc->Device, &wd, nullptr);
    vc.reset();

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
