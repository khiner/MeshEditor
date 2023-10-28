#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "imgui_internal.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <format>
#include <stdexcept>

#include "Scene.h"
#include "VulkanContext.h"
#include "Window.h"

// #define IMGUI_UNLIMITED_FRAME_RATE

static WindowsState Windows;

// Vulkan data.
static std::unique_ptr<VulkanContext> VC;
static std::unique_ptr<Scene> MainScene;
static ImGui_ImplVulkanH_Window g_MainWindowData;
static uint g_MinImageCount = 2;
static bool g_SwapChainRebuild = false;

// All the ImGui_ImplVulkanH_XXX structures/functions are optional helpers used by the demo.
// Your real engine/app may not use them.
static void SetupVulkanWindow(ImGui_ImplVulkanH_Window *wd, vk::SurfaceKHR surface, int width, int height) {
    wd->Surface = surface;

    // Check for WSI support
    auto res = VC->PhysicalDevice.getSurfaceSupportKHR(VC->QueueFamily, wd->Surface);
    if (res != VK_TRUE) throw std::runtime_error("Error no WSI support on physical device 0\n");

    // Select surface format.
    const VkFormat requestSurfaceImageFormat[] = {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM};
    const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(VC->PhysicalDevice, wd->Surface, requestSurfaceImageFormat, (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat), requestSurfaceColorSpace);

    // Select present mode.
#ifdef IMGUI_UNLIMITED_FRAME_RATE
    VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR};
#else
    VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_FIFO_KHR};
#endif
    wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(VC->PhysicalDevice, wd->Surface, &present_modes[0], IM_ARRAYSIZE(present_modes));
    // printf("[vulkan] Selected PresentMode = %d\n", wd->PresentMode);

    // Create SwapChain, RenderPass, Framebuffer, etc.
    IM_ASSERT(g_MinImageCount >= 2);
    ImGui_ImplVulkanH_CreateOrResizeWindow(VC->Instance.get(), VC->PhysicalDevice, VC->Device.get(), wd, VC->QueueFamily, nullptr, width, height, g_MinImageCount);
}

static void CleanupVulkanWindow() {
    ImGui_ImplVulkanH_DestroyWindow(VC->Instance.get(), VC->Device.get(), &g_MainWindowData, nullptr);
}

static void FrameRender(ImGui_ImplVulkanH_Window *wd, ImDrawData *draw_data) {
    VkSemaphore image_acquired_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    const VkResult err = vkAcquireNextImageKHR(VC->Device.get(), wd->Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &wd->FrameIndex);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        g_SwapChainRebuild = true;
        return;
    }
    CheckVk(err);

    ImGui_ImplVulkanH_Frame *fd = &wd->Frames[wd->FrameIndex];
    {
        CheckVk(vkWaitForFences(VC->Device.get(), 1, &fd->Fence, VK_TRUE, UINT64_MAX)); // wait indefinitely instead of periodically checking
        CheckVk(vkResetFences(VC->Device.get(), 1, &fd->Fence));
    }
    {
        CheckVk(vkResetCommandPool(VC->Device.get(), fd->CommandPool, 0));
        VkCommandBufferBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CheckVk(vkBeginCommandBuffer(fd->CommandBuffer, &info));
    }
    {
        VkRenderPassBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass = wd->RenderPass;
        info.framebuffer = fd->Framebuffer;
        info.renderArea.extent.width = wd->Width;
        info.renderArea.extent.height = wd->Height;
        info.clearValueCount = 1;
        info.pClearValues = &wd->ClearValue;
        vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
    }

    // Record dear imgui primitives into command buffer
    ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);

    // Submit command buffer
    vkCmdEndRenderPass(fd->CommandBuffer);
    {
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.waitSemaphoreCount = 1;
        info.pWaitSemaphores = &image_acquired_semaphore;
        info.pWaitDstStageMask = &wait_stage;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &fd->CommandBuffer;
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &render_complete_semaphore;

        CheckVk(vkEndCommandBuffer(fd->CommandBuffer));
        CheckVk(vkQueueSubmit(VC->Queue, 1, &info, fd->Fence));
    }
}

static void FramePresent(ImGui_ImplVulkanH_Window *wd) {
    if (g_SwapChainRebuild) return;

    VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    VkPresentInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &render_complete_semaphore;
    info.swapchainCount = 1;
    info.pSwapchains = &wd->Swapchain;
    info.pImageIndices = &wd->FrameIndex;
    VkResult err = vkQueuePresentKHR(VC->Queue, &info);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        g_SwapChainRebuild = true;
        return;
    }
    CheckVk(err);
    wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->ImageCount; // Now we can use the next set of semaphores.
}

using namespace ImGui;

int main(int, char **) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMEPAD) != 0) {
        throw std::runtime_error(std::format("SDL_Init error: {}", SDL_GetError()));
    }

    SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");

    // Create window with Vulkan graphics context.
    const auto window_flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    auto *Window = SDL_CreateWindowWithPosition("Mesh2Audio-Vulkan", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);

    uint extensions_count = 0;
    SDL_Vulkan_GetInstanceExtensions(&extensions_count, nullptr);
    std::vector<const char *> extensions(extensions_count);
    SDL_Vulkan_GetInstanceExtensions(&extensions_count, extensions.data());
    VC = std::make_unique<VulkanContext>(extensions);

    // Create window surface.
    VkSurfaceKHR surface;
    if (SDL_Vulkan_CreateSurface(Window, VC->Instance.get(), &surface) == 0) throw std::runtime_error("Failed to create Vulkan surface.\n");

    // Create framebuffers.
    int w, h;
    SDL_GetWindowSize(Window, &w, &h);
    ImGui_ImplVulkanH_Window *wd = &g_MainWindowData;
    SetupVulkanWindow(wd, surface, w, h);

    // Setup ImGui context.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO &io = GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows

    io.IniFilename = nullptr; // Disable ImGui's .ini file saving

    StyleColorsDark();
    // StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplSDL3_InitForVulkan(Window);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = VC->Instance.get();
    init_info.PhysicalDevice = VC->PhysicalDevice;
    init_info.Device = VC->Device.get();
    init_info.QueueFamily = VC->QueueFamily;
    init_info.Queue = VC->Queue;
    init_info.PipelineCache = VC->PipelineCache.get();
    init_info.DescriptorPool = VC->DescriptorPool.get();
    init_info.Subpass = 0;
    init_info.MinImageCount = g_MinImageCount;
    init_info.ImageCount = wd->ImageCount;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator = nullptr;
    init_info.CheckVkResultFn = CheckVk;
    ImGui_ImplVulkan_Init(&init_info, wd->RenderPass);

    // Load fonts.
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return a nullptr. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Use '#define IMGUI_ENABLE_FREETYPE' in your imconfig file to use Freetype for higher quality font rendering.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    // io.Fonts->AddFontDefault();
    // io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\segoeui.ttf", 18.0f);
    // io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    // io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    // io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    // ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, nullptr, io.Fonts->GetGlyphRangesJapanese());
    // IM_ASSERT(font != nullptr);

    // Upload fonts.
    {
        // Use any command queue
        vk::CommandPool command_pool = wd->Frames[wd->FrameIndex].CommandPool;
        vk::CommandBuffer command_buffer = wd->Frames[wd->FrameIndex].CommandBuffer;
        VC->Device->resetCommandPool(command_pool, vk::CommandPoolResetFlags());
        command_buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
        command_buffer.end();

        vk::SubmitInfo submit;
        submit.setCommandBuffers(command_buffer);
        VC->Queue.submit(submit);
        VC->Device->waitIdle();
        ImGui_ImplVulkan_DestroyFontUploadObjects();
    }

    MainScene = std::make_unique<Scene>(*VC, w, h);
    MainScene->TC.DescriptorSet = ImGui_ImplVulkan_AddTexture(MainScene->TC.TextureSampler.get(), MainScene->TC.ResolveImageView.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    ImVec4 clear_color = {0.45f, 0.55f, 0.60f, 1.f};

    // Main loop
    bool done = false;
    while (!done) {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT)
                done = true;
            if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(Window))
                done = true;
        }

        // Resize swap chain?
        if (g_SwapChainRebuild) {
            int width, height;
            SDL_GetWindowSize(Window, &width, &height);
            if (width > 0 && height > 0) {
                ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
                ImGui_ImplVulkanH_CreateOrResizeWindow(VC->Instance.get(), VC->PhysicalDevice, VC->Device.get(), &g_MainWindowData, VC->QueueFamily, nullptr, width, height, g_MinImageCount);
                g_MainWindowData.FrameIndex = 0;
                g_SwapChainRebuild = false;
            }
        }

        // Start the ImGui frame.
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        NewFrame();

        auto dockspace_id = DockSpaceOverViewport(nullptr, ImGuiDockNodeFlags_PassthruCentralNode);
        if (GetFrameCount() == 1) {
            auto demo_node_id = DockBuilderSplitNode(dockspace_id, ImGuiDir_Right, 0.3f, nullptr, &dockspace_id);
            DockBuilderDockWindow(Windows.ImGuiDemo.Name, demo_node_id);
            auto mesh_node_id = dockspace_id;
            auto controls_node_id = DockBuilderSplitNode(mesh_node_id, ImGuiDir_Left, 0.4f, nullptr, &mesh_node_id);
            DockBuilderDockWindow(Windows.SceneControls.Name, controls_node_id);
            DockBuilderDockWindow(Windows.Scene.Name, mesh_node_id);
        }

        if (Windows.ImGuiDemo.Visible) ShowDemoWindow(&Windows.ImGuiDemo.Visible);

        if (Windows.Scene.Visible) {
            PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
            Begin(Windows.Scene.Name, &Windows.Scene.Visible);
            Image((ImTextureID)MainScene->TC.DescriptorSet, ImGui::GetContentRegionAvail());
            End();
            PopStyleVar();
        }

        // Rendering
        ImGui::Render();
        ImDrawData *draw_data = GetDrawData();
        const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
        if (!is_minimized) {
            wd->ClearValue.color.float32[0] = clear_color.x * clear_color.w;
            wd->ClearValue.color.float32[1] = clear_color.y * clear_color.w;
            wd->ClearValue.color.float32[2] = clear_color.z * clear_color.w;
            wd->ClearValue.color.float32[3] = clear_color.w;
            FrameRender(wd, draw_data);
            FramePresent(wd);
        }
    }

    // Cleanup
    VC->Device->waitIdle();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    DestroyContext();

    CleanupVulkanWindow();
    MainScene.reset();
    VC.reset();

    SDL_DestroyWindow(Window);
    SDL_Quit();

    return 0;
}
