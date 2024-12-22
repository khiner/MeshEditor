#include <format>
#include <stdexcept>

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "imgui_internal.h"
#include "implot.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <nfd.h>

#include "numeric/vec4.h"

#include "Scene.h"
#include "Tets.h"
#include "Window.h"
#include "Worker.h"
#include "mesh/Arrow.h"
#include "mesh/Primitives.h"
#include "vulkan/VulkanContext.h"

#include "audio/AudioSourcesPlayer.h"
#include "audio/RealImpact.h"
#include "audio/SoundObject.h"

// #define IMGUI_UNLIMITED_FRAME_RATE

ImGui_ImplVulkanH_Window MainWindowData;
uint MinImageCount = 2;
bool SwapChainRebuild = false;

WindowsState Windows;
std::unique_ptr<VulkanContext> VC;
std::unique_ptr<Scene> MainScene;
vk::DescriptorSet MainSceneDescriptorSet;
vk::UniqueSampler MainSceneTextureSampler;

entt::registry R;
AudioSourcesPlayer AudioSources{R};

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
    IM_ASSERT(MinImageCount >= 2);
    ImGui_ImplVulkanH_CreateOrResizeWindow(*VC->Instance, VC->PhysicalDevice, *VC->Device, wd, VC->QueueFamily, nullptr, width, height, MinImageCount);
}

static void CleanupVulkanWindow() {
    ImGui_ImplVulkanH_DestroyWindow(*VC->Instance, *VC->Device, &MainWindowData, nullptr);
}

static void CheckVk(VkResult err) {
    if (err != 0) throw std::runtime_error(std::format("Vulkan error: {}", int(err)));
}

static void FrameRender(ImGui_ImplVulkanH_Window *wd, ImDrawData *draw_data) {
    VkSemaphore image_acquired_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    const VkResult err = vkAcquireNextImageKHR(*VC->Device, wd->Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &wd->FrameIndex);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        SwapChainRebuild = true;
        return;
    }
    CheckVk(err);

    ImGui_ImplVulkanH_Frame *fd = &wd->Frames[wd->FrameIndex];
    {
        CheckVk(vkWaitForFences(*VC->Device, 1, &fd->Fence, VK_TRUE, UINT64_MAX)); // wait indefinitely instead of periodically checking
        CheckVk(vkResetFences(*VC->Device, 1, &fd->Fence));
    }
    {
        CheckVk(vkResetCommandPool(*VC->Device, fd->CommandPool, 0));
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
    if (SwapChainRebuild) return;

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
        SwapChainRebuild = true;
        return;
    }
    CheckVk(err);
    wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->SemaphoreCount; // Now we can use the next set of semaphores.
}

static entt::entity FindListenerEntityWithIndex(uint index) {
    for (const auto entity : R.view<RealImpactListenerPoint>()) {
        if (R.get<RealImpactListenerPoint>(entity).Index == index) return entity;
    }
    return entt::null;
}

static void LoadRealImpact(const fs::path &path, entt::registry &R) {
    if (!fs::exists(path)) throw std::runtime_error(std::format("RealImpact path does not exist: {}", path.string()));

    MainScene->ClearMeshes();
    RealImpact real_impact{fs::path(path)};
    // RealImpact meshes are oriented with Z up, but MeshEditor uses Y up.
    mat4 swap{1};
    swap[1][1] = 0;
    swap[1][2] = 1;
    swap[2][1] = 1;
    swap[2][2] = 0;

    const auto object_name = real_impact.ObjectName;
    const auto object_entity = MainScene->AddMesh(
        real_impact.ObjPath,
        {.Name = std::format("RealImpact Object: {}", object_name), .Transform = std::move(swap), .Submit = false}
    );
    // Vertex indices may have changed due to deduplication.
    auto &mesh = R.get<Mesh>(object_entity);
    for (uint i = 0; i < RealImpact::NumImpactVertices; ++i) {
        const auto &pos = real_impact.ImpactPositions[i];
        const auto vh = mesh.FindNearestVertex(pos);
        real_impact.VertexIndices[i] = uint(vh.idx());
        mesh.HighlightVertex(vh);
    }
    MainScene->UpdateRenderBuffers(object_entity);

    static const mat4 I{1};
    auto listener_point_mesh = Cylinder(0.5 * RealImpact::MicWidthMm / 1000.f, RealImpact::MicLengthMm / 1000.f);
    auto listener_points_name = std::format("RealImpact Listeners: {}", object_name);
    const auto listener_point_entity = MainScene->AddMesh(std::move(listener_point_mesh), {std::move(listener_points_name), I, false, false, false});
    static const auto rot_z = glm::rotate(I, float(M_PI / 2), {0, 0, 1}); // Cylinder is oriended with center along the Y axis.
    // todo: `Scene::AddInstances` to add multiple instances at once (mainly to avoid updating model buffer for every instance)
    for (const auto &p : real_impact.LoadListenerPoints()) {
        const auto pos = p.GetPosition(MainScene->World.Up, true);
        const auto rot = glm::rotate(I, glm::radians(float(p.AngleDeg)), MainScene->World.Up) * rot_z;
        const auto listener_point_name = std::format("RealImpact Listener: {}", p.Index);
        R.emplace<RealImpactListenerPoint>(
            MainScene->AddInstance(
                listener_point_entity,
                {.Name = std::move(listener_point_name), .Transform = glm::translate(I, pos) * rot, .Select = false, .Submit = false}
            ),
            p
        );
    }
    R.emplace<RealImpact>(object_entity, std::move(real_impact));
    MainScene->RecordAndSubmitCommandBuffer();
}

using namespace ImGui;

std::unique_ptr<Worker<Tets>> TetGenerator;

static void HelpMarker(const char *desc) {
    SameLine();
    TextDisabled("(?)");
    if (BeginItemTooltip()) {
        PushTextWrapPos(GetFontSize() * 35.0f);
        TextUnformatted(desc);
        PopTextWrapPos();
        EndTooltip();
    }
}

void AudioModelControls() {
    const static float CharWidth = CalcTextSize("A").x;

    const auto selected_entity = MainScene->GetSelectedEntity();
    const auto sound_objects = R.view<SoundObject>();
    if (sound_objects.begin() != sound_objects.end() && CollapsingHeader("Sound objects")) {
        if (BeginTable("Sound objects", 3)) {
            TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, CharWidth * 10);
            TableSetupColumn("Name");
            TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, CharWidth * 20);
            TableHeadersRow();
            entt::entity entity_to_select = entt::null, entity_to_delete = entt::null;
            for (const auto &[entity, sound_object] : sound_objects.each()) {
                PushID(uint(entity));
                TableNextColumn();
                AlignTextToFramePadding();
                if (entity == selected_entity) TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(ImGuiCol_TextSelectedBg));
                TextUnformatted(IdString(entity).c_str());
                TableNextColumn();
                TextUnformatted(R.get<std::string>(entity).c_str());
                TableNextColumn();
                if (Button("Select")) entity_to_select = entity;
                SameLine();
                if (Button("Delete")) entity_to_delete = entity;
                if (Button("Select listener point")) entity_to_select = entt::entity(sound_object.ListenerEntityId);
                PopID();
            }
            if (entity_to_select != entt::null) MainScene->SelectEntity(entity_to_select);
            if (entity_to_delete != entt::null) MainScene->DestroyEntity(entity_to_delete);
            EndTable();
        }
    }
    const auto listener_points = R.view<RealImpactListenerPoint>();
    if (listener_points.begin() != listener_points.end() && CollapsingHeader("Listener points")) {
        if (BeginTable("Listener points", 3)) {
            TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, CharWidth * 10);
            TableSetupColumn("Name");
            TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, CharWidth * 16);
            TableHeadersRow();
            entt::entity entity_to_select = entt::null, entity_to_delete = entt::null;
            for (const auto entity : listener_points) {
                PushID(uint(entity));
                TableNextColumn();
                AlignTextToFramePadding();
                if (entity == selected_entity) TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(ImGuiCol_TextSelectedBg));
                TextUnformatted(IdString(entity).c_str());
                TableNextColumn();
                TextUnformatted(R.get<std::string>(entity).c_str());
                TableNextColumn();
                if (Button("Select")) entity_to_select = entity;
                SameLine();
                if (Button("Delete")) entity_to_delete = entity;
                PopID();
            }
            if (entity_to_select != entt::null) MainScene->SelectEntity(entity_to_select);
            if (entity_to_delete != entt::null) MainScene->DestroyEntity(entity_to_delete);
            EndTable();
        }
    }

    if (selected_entity == entt::null) return;

    if (R.all_of<Mesh>(selected_entity) && !R.all_of<Tets>(selected_entity) && !R.all_of<RealImpactListenerPoint>(selected_entity)) {
        if (TetGenerator) {
            if (auto tets = TetGenerator->Render()) {
                // Add an invisible tet mesh to the scene, to support toggling between surface/volumetric tet mesh views.
                MainScene->AddMesh(tets->CreateMesh(), {"Tet Mesh", MainScene->GetModel(selected_entity), false, false, false});
                TetGenerator.reset();

                const auto *real_impact = R.try_get<RealImpact>(selected_entity);
                const auto material_name = real_impact ? real_impact->MaterialName : DefaultMaterialPresetName;
                const auto listener_point_entity = FindListenerEntityWithIndex(263); // This listener point is roughly centered.
                const auto *listener_point = R.try_get<RealImpactListenerPoint>(listener_point_entity);
                const auto &registry_tets = R.emplace<Tets>(selected_entity, std::move(*tets));
                auto &sound_object = R.emplace<SoundObject>(
                    selected_entity, MainScene->GetName(selected_entity), registry_tets, material_name,
                    listener_point ? listener_point->GetPosition() : vec3{0}, uint(listener_point_entity)
                );
                if (real_impact && listener_point) sound_object.SetImpactFrames(listener_point->LoadImpactSamples(*real_impact));
            }
        } else { // todo conditionally show "Regenerate tet mesh"
            SeparatorText("Tet mesh generation");

            static bool preserve_surface = true;
            static bool quality = false;
            Checkbox("Preserve surface", &preserve_surface);
            HelpMarker("Input boundary edges and faces of the mesh are preserved in the generated tetrahedral mesh.\nSteiner points appear only in the interior space of the mesh.");
            SameLine();
            Checkbox("Quality", &quality);
            HelpMarker("Adds new points to improve the mesh quality.");
            if (Button("Create sound object")) {
                // If RealImpact data is present, ensure impact points on the tet mesh are the exact same as the surface mesh.
                // todo quality UI toggle, and also a toggle for `PreserveSurface` for non-RealImpact meshes
                // todo display tet mesh in UI and select vertices for debugging (just like other meshes but restrict to edge view)
                const auto &surface_mesh = R.get<Mesh>(selected_entity);
                TetGenerator = std::make_unique<Worker<Tets>>("Generating tetrahedral mesh...", [&] {
                    return Tets::CreateTets(surface_mesh, {.PreserveSurface = preserve_surface, .Quality = quality});
                });
            }
        }
        return;
    }

    // Display the selected sound object, or the first one if any are present.
    if (sound_objects.begin() == sound_objects.end()) return;

    SeparatorText("Audio model");
    entt::entity sound_object_entity = entt::null;
    for (const auto entity : sound_objects) {
        if (entity == selected_entity || R.get<SoundObject>(entity).ListenerEntityId == uint(selected_entity)) {
            sound_object_entity = entity;
            break;
        }
    }
    if (sound_object_entity == entt::null) sound_object_entity = *sound_objects.begin();

    const auto &mesh = R.get<Mesh>(sound_object_entity);
    auto &sound_object = R.get<SoundObject>(sound_object_entity);
    if (R.all_of<RealImpactListenerPoint>(selected_entity)) {
        if (sound_object.ListenerEntityId != uint(selected_entity)) {
            if (Button("Set listener point")) {
                const auto &real_impact = R.get<RealImpact>(sound_object_entity);
                const auto &listener_point = R.get<RealImpactListenerPoint>(selected_entity);
                sound_object.ListenerPosition = listener_point.GetPosition(MainScene->World.Up, true);
                sound_object.SetImpactFrames(listener_point.LoadImpactSamples(real_impact));
                sound_object.ListenerEntityId = uint(selected_entity);
            }
            if (Button("Select listener point")) {
                MainScene->SelectEntity(entt::entity(sound_object.ListenerEntityId));
            }
        } else {
            if (Button("Select sound object")) {
                MainScene->SelectEntity(sound_object_entity);
            }
        }
    } else {
        if (Button("Select listener point")) {
            MainScene->SelectEntity(entt::entity(sound_object.ListenerEntityId));
        }
    }
    if (Button("Remove audio model")) {
        R.remove<SoundObject>(sound_object_entity);
        R.remove<Tets>(sound_object_entity);
    }

    const auto before_current_vertex = sound_object.CurrentVertex;
    sound_object.RenderControls(); // May change the current vertex.
    if (const auto *selected_vertex = R.try_get<SelectedVertex>(sound_object_entity)) {
        if (const auto nearest_excite_vertex = sound_object.FindNearestExcitableVertex(selected_vertex->Position)) {
            sound_object.SetVertex(*nearest_excite_vertex);
            sound_object.SetVertexForce(1);
        }
    } else {
        sound_object.SetVertexForce(0);
    }
    if (!sound_object.CurrentVertexIndicatorEntityId || sound_object.CurrentVertex != before_current_vertex) {
        // Vertex indicator arrow mesh needs to be created or moved to point at the current excitable vertex.
        const auto &model = MainScene->GetModel(sound_object_entity);
        const auto vh = Mesh::VH(sound_object.CurrentVertex);
        const vec3 vertex_pos = {model * vec4{mesh.GetPosition(vh), 1}};
        const vec3 normal = {model * vec4{mesh.GetVertexNormal(vh), 0}};

        const float scale_factor = 0.1f * mesh.BoundingBox.DiagonalLength();
        const mat4 scale = glm::scale({1}, vec3{scale_factor});
        const mat4 translate = glm::translate({1}, vertex_pos + 0.05f * scale_factor * normal);
        const mat4 rotate = glm::mat4_cast(glm::rotation(MainScene->World.Up, normal));
        mat4 indicator_model{translate * rotate * scale};
        if (!sound_object.CurrentVertexIndicatorEntityId) {
            auto vertex_indicator_mesh = Arrow();
            vertex_indicator_mesh.SetFaceColor({1, 0, 0, 1});
            sound_object.CurrentVertexIndicatorEntityId = uint(MainScene->AddMesh(
                std::move(vertex_indicator_mesh),
                {.Name = "Excite vertex indicator", .Transform = std::move(indicator_model), .Select = false, .Submit = true}
            ));
        } else {
            MainScene->SetModel(entt::entity(sound_object.CurrentVertexIndicatorEntityId), std::move(indicator_model), true);
        }
    }
}

int main(int, char **) {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        throw std::runtime_error(std::format("SDL_Init error: {}", SDL_GetError()));
    }

    // Create window with Vulkan graphics context.
    const auto window_flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    auto *Window = SDL_CreateWindow("MeshEditor", 1280, 720, window_flags);

    uint extensions_count = 0;
    const char *const *instance_extensions_raw = SDL_Vulkan_GetInstanceExtensions(&extensions_count);
    const std::vector<const char *> instance_extensions(instance_extensions_raw, instance_extensions_raw + extensions_count);
    VC = std::make_unique<VulkanContext>(instance_extensions);

    // Create window surface.
    VkSurfaceKHR surface;
    if (!SDL_Vulkan_CreateSurface(Window, *VC->Instance, nullptr, &surface)) throw std::runtime_error("Failed to create Vulkan surface.\n");

    // Create framebuffers.
    int w, h;
    SDL_GetWindowSize(Window, &w, &h);
    ImGui_ImplVulkanH_Window *wd = &MainWindowData;
    SetupVulkanWindow(wd, surface, w, h);
    MainSceneTextureSampler = VC->Device->createSamplerUnique({{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear});

    // Setup ImGui context.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

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
    init_info.Instance = *VC->Instance;
    init_info.PhysicalDevice = VC->PhysicalDevice;
    init_info.Device = *VC->Device;
    init_info.QueueFamily = VC->QueueFamily;
    init_info.Queue = VC->Queue;
    init_info.PipelineCache = *VC->PipelineCache;
    init_info.DescriptorPool = *VC->DescriptorPool;
    init_info.RenderPass = wd->RenderPass;
    init_info.Subpass = 0;
    init_info.MinImageCount = MinImageCount;
    init_info.ImageCount = wd->ImageCount;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator = nullptr;
    init_info.CheckVkResultFn = CheckVk;
    ImGui_ImplVulkan_Init(&init_info);

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

    NFD_Init();

    MainScene = std::make_unique<Scene>(*VC, R);
    AudioSources.Start();

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
        if (SwapChainRebuild) {
            int width, height;
            SDL_GetWindowSize(Window, &width, &height);
            if (width > 0 && height > 0) {
                ImGui_ImplVulkan_SetMinImageCount(MinImageCount);
                ImGui_ImplVulkanH_CreateOrResizeWindow(*VC->Instance, VC->PhysicalDevice, *VC->Device, &MainWindowData, VC->QueueFamily, nullptr, width, height, MinImageCount);
                MainWindowData.FrameIndex = 0;
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
            DockBuilderDockWindow(Windows.ImGuiDemo.Name, demo_node_id);
            DockBuilderDockWindow(Windows.ImPlotDemo.Name, demo_node_id);
            DockBuilderDockWindow(Windows.SceneControls.Name, controls_node_id);
            DockBuilderDockWindow(Windows.Scene.Name, dockspace_id);
        }

        if (BeginMainMenuBar()) {
            if (BeginMenu("File")) {
                if (MenuItem("Load mesh", nullptr)) {
                    static const std::vector<nfdfilteritem_t> filters{{"Mesh object", "obj,off,ply,stl,om"}};
                    nfdchar_t *nfd_path;
                    nfdresult_t result = NFD_OpenDialog(&nfd_path, filters.data(), filters.size(), "");
                    if (result == NFD_OKAY) {
                        const auto path = fs::path(nfd_path);
                        MainScene->AddMesh(path, {.Name = path.filename().string()});
                        NFD_FreePath(nfd_path);
                    } else if (result != NFD_CANCEL) {
                        throw std::runtime_error(std::format("Error loading mesh file: {}", NFD_GetError()));
                    }
                }
                // if (MenuItem("Export mesh", nullptr, false, MainMesh != nullptr)) {
                //     nfdchar_t *path;
                //     nfdresult_t result = NFD_SaveDialog(&path, filtes.data(), filters.size(), nullptr);
                //     if (result == NFD_OKAY) {
                //         MainScene->SaveMesh(fs::path(path));
                //         NFD_FreePath(path);
                //     } else if (result != NFD_CANCEL) {
                //         throw std::runtime_error(std::format("Error saving mesh file: {}", NFD_GetError()));
                //     }
                // }
                if (MenuItem("Load RealImpact", nullptr)) {
                    static const std::vector<nfdfilteritem_t> filters{};
                    nfdchar_t *path;
                    nfdresult_t result = NFD_PickFolder(&path, "");
                    if (result == NFD_OKAY) {
                        LoadRealImpact(fs::path(path), R);
                        NFD_FreePath(path);
                    } else if (result != NFD_CANCEL) {
                        throw std::runtime_error(std::format("Error loading RealImpact file: {}", NFD_GetError()));
                    }
                }
                EndMenu();
            }
            if (BeginMenu("Windows")) {
                MenuItem(Windows.ImGuiDemo.Name, nullptr, &Windows.ImGuiDemo.Visible);
                MenuItem(Windows.ImPlotDemo.Name, nullptr, &Windows.ImPlotDemo.Visible);
                MenuItem(Windows.SceneControls.Name, nullptr, &Windows.SceneControls.Visible);
                MenuItem(Windows.Scene.Name, nullptr, &Windows.Scene.Visible);
                EndMenu();
            }
            EndMainMenuBar();
        }

        if (Windows.ImGuiDemo.Visible) ImGui::ShowDemoWindow(&Windows.ImGuiDemo.Visible);
        if (Windows.ImPlotDemo.Visible) ImPlot::ShowDemoWindow(&Windows.ImPlotDemo.Visible);

        if (Windows.SceneControls.Visible) {
            Begin(Windows.SceneControls.Name, &Windows.SceneControls.Visible);
            if (BeginTabBar("Controls")) {
                if (BeginTabItem("Scene")) {
                    MainScene->RenderConfig();
                    EndTabItem();
                }
                if (BeginTabItem("Audio")) {
                    if (BeginTabBar("Audio")) {
                        if (BeginTabItem("Device")) {
                            AudioSources.RenderControls();
                            EndTabItem();
                        }
                        if (BeginTabItem("Model")) {
                            AudioModelControls();
                            EndTabItem();
                        }
                        EndTabBar();
                    }
                    EndTabItem();
                }
                EndTabBar();
            }
            End();
        }

        if (Windows.Scene.Visible) {
            PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
            Begin(Windows.Scene.Name, &Windows.Scene.Visible);
            if (MainScene->Render()) {
                ImGui_ImplVulkan_RemoveTexture(MainSceneDescriptorSet);
                MainSceneDescriptorSet = ImGui_ImplVulkan_AddTexture(*MainSceneTextureSampler, MainScene->GetResolveImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }

            const auto &cursor = GetCursorPos();
            const auto &scene_extent = MainScene->GetExtent();
            Image((ImTextureID)(void *)MainSceneDescriptorSet, {float(scene_extent.width), float(scene_extent.height)}, {0, 1}, {1, 0});
            SetCursorPos(cursor);
            MainScene->RenderGizmo();
            End();
            PopStyleVar();

            if (ImGui::GetFrameCount() == 1) {
                // Initialize scene now that it has an extent.
                static const auto DefaultRealImpactPath = fs::path("../../") / "RealImpact" / "dataset" / "22_Cup" / "preprocessed";
                if (fs::exists(DefaultRealImpactPath)) LoadRealImpact(DefaultRealImpactPath, R);
            }
        }

        // Render
        ImGui::Render();
        auto *draw_data = GetDrawData();
        if (bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f); !is_minimized) {
            static constexpr vec4 ClearColor{0.45f, 0.55f, 0.60f, 1.f};
            wd->ClearValue.color.float32[0] = ClearColor.r * ClearColor.a;
            wd->ClearValue.color.float32[1] = ClearColor.g * ClearColor.a;
            wd->ClearValue.color.float32[2] = ClearColor.b * ClearColor.a;
            wd->ClearValue.color.float32[3] = ClearColor.a;
            FrameRender(wd, draw_data);
            FramePresent(wd);
        }
    }

    // Cleanup
    NFD_Quit();

    VC->Device->waitIdle();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();

    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    CleanupVulkanWindow();
    MainSceneTextureSampler.reset();
    MainScene.reset();
    VC.reset();

    R.clear();

    SDL_DestroyWindow(Window);
    SDL_Quit();

    return 0;
}
