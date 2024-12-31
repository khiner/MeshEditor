#include <format>
#include <stack>
#include <stdexcept>

#include "Widgets.h" // imgui

#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "imgui_internal.h"
#include "implot.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <entt/entity/registry.hpp>
#include <glm/gtx/quaternion.hpp>
#include <nfd.h>

#include "Scene.h"
#include "SvgResource.h"
#include "Tets.h"
#include "Window.h"
#include "Worker.h"
#include "audio/AudioDevice.h"
#include "audio/RealImpact.h"
#include "audio/SoundObject.h"
#include "mesh/Arrow.h"
#include "mesh/Primitives.h"
#include "numeric/vec4.h"
#include "vulkan/VulkanContext.h"

// #define IMGUI_UNLIMITED_FRAME_RATE

// If an entity has this component, it is being listened to by `Listener`.
struct SoundObjectListener {
    entt::entity Listener;
};
// If an entity has this component, it is being excited at this vertex/force.
struct SoundObjectExcitation {
    uint Vertex;
    float Force;
};
struct SoundObjectExcitationIndicator {
    entt::entity Entity;
};

namespace {
std::unique_ptr<VulkanContext> VC;

uint MinImageCount = 2;
bool SwapChainRebuild = false;

WindowsState Windows;
std::unique_ptr<Scene> MainScene;
std::unique_ptr<ImGuiTexture> MainSceneTexture;

entt::registry R;

void AudioCallback(FrameInfo device_data, float *output, const float *input, uint frame_count) {
    for (const auto &audio_source : R.storage<SoundObject>()) {
        audio_source.ProduceAudio(device_data, (float *)input, (float *)output, frame_count);
    }
}

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

    IM_ASSERT(MinImageCount >= 2);
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

entt::entity FindListenerEntityWithIndex(uint index) {
    for (const auto &[entity, listener_point] : R.view<const RealImpactListenerPoint>().each()) {
        if (listener_point.Index == index) return entity;
    }
    return entt::null;
}

void LoadRealImpact(const fs::path &path, entt::registry &R) {
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
        {.Name = std::format("RealImpact Object: {}", object_name), .Transform = std::move(swap)}
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

    static constexpr mat4 I{1};
    auto listener_point_mesh = Cylinder(0.5f * RealImpact::MicWidthMm / 1000.f, RealImpact::MicLengthMm / 1000.f);
    auto listener_points_name = std::format("RealImpact Listeners: {}", object_name);
    const auto listener_point_entity = MainScene->AddMesh(std::move(listener_point_mesh), {std::move(listener_points_name), I, false, false});
    static const auto rot_z = glm::rotate(I, float(M_PI_2), {0, 0, 1}); // Cylinder is oriended with center along the Y axis.
    // todo: `Scene::AddInstances` to add multiple instances at once (mainly to avoid updating model buffer for every instance)
    for (const auto &p : real_impact.LoadListenerPoints()) {
        const auto pos = p.GetPosition(MainScene->World.Up, true);
        const auto rot = glm::rotate(I, glm::radians(float(p.AngleDeg)), MainScene->World.Up) * rot_z;
        const auto listener_point_name = std::format("RealImpact Listener: {}", p.Index);
        const auto listener_point_instance_entity = MainScene->AddInstance(
            listener_point_entity,
            {.Name = std::move(listener_point_name), .Transform = glm::translate(I, pos) * rot, .Select = false}
        );
        R.emplace<RealImpactListenerPoint>(listener_point_instance_entity, p);
    }
    R.emplace<RealImpact>(object_entity, std::move(real_impact));
}

using namespace ImGui;

void HelpMarker(const char *desc) {
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
    static const CreateSvgResource CreateSvg = [](std::unique_ptr<SvgResource> &svg, fs::path path) {
        VC->Device->waitIdle();
        svg.reset(); // Ensure destruction before creation.
        svg = std::make_unique<SvgResource>(*VC, std::move(path));
    };
    static const float CharWidth = CalcTextSize("A").x;

    const auto selected_entity = MainScene->GetSelectedEntity();
    if (!R.storage<SoundObject>().empty() && CollapsingHeader("Sound objects")) {
        if (MeshEditor::BeginTable("Sound objects", 3)) {
            TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, CharWidth * 10);
            TableSetupColumn("Name");
            TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, CharWidth * 20);
            TableHeadersRow();
            entt::entity entity_to_select = entt::null, entity_to_delete = entt::null;
            for (const auto &[entity, sound_object] : R.view<const SoundObject>().each()) {
                const bool is_selected = entity == selected_entity;
                PushID(uint(entity));
                TableNextColumn();
                AlignTextToFramePadding();
                if (is_selected) TableSetBgColor(ImGuiTableBgTarget_RowBg0, GetColorU32(ImGuiCol_TextSelectedBg));
                TextUnformatted(IdString(entity).c_str());
                TableNextColumn();
                TextUnformatted(R.get<Name>(entity).Value.c_str());
                TableNextColumn();
                if (is_selected) BeginDisabled();
                if (Button("Select")) entity_to_select = entity;
                if (is_selected) EndDisabled();
                SameLine();
                if (Button("Delete")) entity_to_delete = entity;
                if (const auto *sound_listener = R.try_get<SoundObjectListener>(entity); sound_listener) {
                    if (Button("Select listener point")) entity_to_select = sound_listener->Listener;
                }
                PopID();
            }
            if (entity_to_select != entt::null) MainScene->SelectEntity(entity_to_select);
            if (entity_to_delete != entt::null) MainScene->DestroyEntity(entity_to_delete);
            EndTable();
        }
    }
    if (!R.storage<RealImpactListenerPoint>().empty() && CollapsingHeader("Listener points")) {
        if (MeshEditor::BeginTable("Listener points", 3)) {
            TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, CharWidth * 10);
            TableSetupColumn("Name");
            TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, CharWidth * 16);
            TableHeadersRow();
            entt::entity entity_to_select = entt::null, entity_to_delete = entt::null;
            for (const auto entity : R.view<RealImpactListenerPoint>()) {
                const bool is_selected = entity == selected_entity;
                PushID(uint(entity));
                TableNextColumn();
                AlignTextToFramePadding();
                if (is_selected) TableSetBgColor(ImGuiTableBgTarget_RowBg0, GetColorU32(ImGuiCol_TextSelectedBg));
                TextUnformatted(IdString(entity).c_str());
                TableNextColumn();
                TextUnformatted(R.get<Name>(entity).Value.c_str());
                TableNextColumn();
                if (is_selected) BeginDisabled();
                if (Button("Select")) entity_to_select = entity;
                if (is_selected) EndDisabled();
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

    static std::unique_ptr<Worker<Tets>> TetGenerator;
    if (R.all_of<Mesh>(selected_entity) && !R.all_of<Tets>(selected_entity) && !R.all_of<RealImpactListenerPoint>(selected_entity)) {
        if (TetGenerator) {
            if (auto tets = TetGenerator->Render()) {
                // Add an invisible tet mesh to the scene, to support toggling between surface/volumetric tet mesh views.
                MainScene->AddMesh(tets->CreateMesh(), {"Tet Mesh", MainScene->GetModel(selected_entity), false, false});
                TetGenerator.reset();

                const auto *real_impact = R.try_get<RealImpact>(selected_entity);
                const auto material_name = real_impact ? real_impact->MaterialName : DefaultMaterialPresetName;
                const auto listener_point_entity = FindListenerEntityWithIndex(263); // This listener point is roughly centered.
                R.emplace<Tets>(selected_entity, std::move(*tets));
                const auto *listener_point = R.try_get<RealImpactListenerPoint>(listener_point_entity);
                if (listener_point) R.emplace<SoundObjectListener>(selected_entity, listener_point_entity);

                auto &sound_object = R.emplace<SoundObject>(selected_entity, material_name, CreateSvg);
                if (real_impact && listener_point) sound_object.SetImpactFrames(listener_point->LoadImpactSamples(*real_impact));

                R.emplace<Excitable>(selected_entity); // Let the scene know this object is excitable.
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
    if (R.storage<SoundObject>().empty()) return;

    SeparatorText("Audio model");

    const auto FindSelectedSoundEntity = [&]() {
        for (const auto entity : R.view<SoundObject>()) {
            if (entity == selected_entity) return entity;
            if (const auto *listener = R.try_get<SoundObjectListener>(entity);
                listener && listener->Listener == selected_entity) return entity;
        }
        return *R.view<const SoundObject>().begin();
    };
    const auto sound_entity = FindSelectedSoundEntity();
    auto &sound_object = R.get<SoundObject>(sound_entity);
    if (sound_entity != selected_entity && Button("Select sound object")) {
        MainScene->SelectEntity(sound_entity);
    }

    const auto *listener = R.try_get<SoundObjectListener>(sound_entity);
    if (listener && listener->Listener != selected_entity) {
        if (Button("Select listener point")) {
            MainScene->SelectEntity(listener->Listener);
        }
    }
    if (const auto *listener_point = R.try_get<RealImpactListenerPoint>(selected_entity);
        listener_point && (!listener || selected_entity != listener->Listener)) {
        if (Button("Set listener point")) {
            sound_object.SetImpactFrames(listener_point->LoadImpactSamples(R.get<RealImpact>(sound_entity)));
            R.emplace_or_replace<SoundObjectListener>(sound_entity, selected_entity);
        }
    }

    const auto &tets = R.get<Tets>(selected_entity);
    if (auto sound_object_action = sound_object.RenderControls(GetName(R, sound_entity), tets)) {
        // We introduce this component indirection since excitations have multiple scene effects
        // (vertex indicator arrow, applying the excitation), and can be triggered in multiple ways
        // (from the control UI or clicking on the mesh).
        std::visit(
            Match{
                [&](SoundObjectAction::SelectVertex action) {
                    sound_object.Apply(action);
                    R.remove<SoundObjectExcitation>(sound_entity);
                },
                [&](SoundObjectAction::SetExciteForce action) {
                    if (action.Force == 0) {
                        R.remove<SoundObjectExcitation>(sound_entity);
                    } else {
                        R.emplace<SoundObjectExcitation>(sound_entity, sound_object.SelectedVertex, action.Force);
                    }
                },
                [&](auto action) {
                    sound_object.Apply(action);
                }
            },
            *sound_object_action
        );
    }

    if (Button("Remove audio model")) {
        R.remove<Excitable, SoundObjectListener, SoundObject, Tets>(sound_entity);
    }
}

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
    R.on_construct<SoundObjectExcitation>().connect<[](entt::registry &r, entt::entity entity) {
        if (auto *sound_object = r.try_get<SoundObject>(entity)) {
            const auto &excitation = r.get<SoundObjectExcitation>(entity);
            sound_object->Apply(SoundObjectAction::Excite{excitation.Vertex, excitation.Force});

            // Orient the camera towards the excited vertex.
            const auto &mesh = r.get<Mesh>(entity);
            const auto &model = MainScene->GetModel(entity);
            const auto vh = Mesh::VH(sound_object->SelectedVertex);
            const vec3 vertex_pos{model * vec4{mesh.GetPosition(vh), 1}};
            MainScene->Camera.SetTargetDirection(glm::normalize(vertex_pos - MainScene->Camera.Target));

            // Create vertex indicator arrow pointing at the excited vertex.
            const vec3 normal{model * vec4{mesh.GetVertexNormal(vh), 0}};
            const float scale_factor = 0.1f * mesh.BoundingBox.DiagonalLength();
            const mat4 scale = glm::scale({1}, vec3{scale_factor});
            const mat4 translate = glm::translate({1}, vertex_pos + 0.05f * scale_factor * normal);
            const mat4 rotate = glm::mat4_cast(glm::rotation(MainScene->World.Up, normal));
            auto vertex_indicator_mesh = Arrow();
            vertex_indicator_mesh.SetFaceColor({1, 0, 0, 1});
            const auto indicator_entity = MainScene->AddMesh(
                std::move(vertex_indicator_mesh),
                {.Name = "Excite vertex indicator", .Transform = mat4{translate * rotate * scale}, .Select = false}
            );
            r.emplace<SoundObjectExcitationIndicator>(entity, indicator_entity);
        }
    }>();
    R.on_destroy<SoundObjectExcitation>().connect<[](entt::registry &r, entt::entity entity) {
        if (auto *sound_object = r.try_get<SoundObject>(entity)) {
            sound_object->Apply(SoundObjectAction::SetExciteForce{0.f});
        }
        if (const auto *excitation_indicator = r.try_get<SoundObjectExcitationIndicator>(entity)) {
            MainScene->DestroyEntity(excitation_indicator->Entity);
        }
        r.remove<SoundObjectExcitationIndicator>(entity);
    }>();
    R.on_construct<ExcitedVertex>().connect<[](entt::registry &r, entt::entity entity) {
        if (const auto *sound_object = r.try_get<SoundObject>(entity)) {
            const auto &tets = r.get<Tets>(entity);
            const auto &excited_vertex = r.get<ExcitedVertex>(entity);
            const auto &mesh = r.get<Mesh>(entity);
            const auto position = mesh.GetPosition(Mesh::VH(excited_vertex.Vertex));
            if (const auto nearest_excite_vertex = sound_object->FindNearestExcitableVertex(tets, position)) {
                r.emplace<SoundObjectExcitation>(entity, *nearest_excite_vertex, 1.f);
            }
        }
    }>();
    R.on_destroy<ExcitedVertex>().connect<[](entt::registry &r, entt::entity entity) {
        r.remove<SoundObjectExcitation>(entity);
    }>();

    MainScene = std::make_unique<Scene>(*VC, R);

    AudioDevice audio_device{AudioCallback};
    audio_device.Start();

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
                    if (auto result = NFD_OpenDialog(&nfd_path, filters.data(), filters.size(), ""); result == NFD_OKAY) {
                        const auto path = fs::path(nfd_path);
                        MainScene->AddMesh(path, {.Name = path.filename().string()});
                        NFD_FreePath(nfd_path);
                    } else if (result != NFD_CANCEL) {
                        throw std::runtime_error(std::format("Error loading mesh file: {}", NFD_GetError()));
                    }
                }
                // if (MenuItem("Export mesh", nullptr, false, MainMesh != nullptr)) {
                //     nfdchar_t *path;
                //     if (auto result = NFD_SaveDialog(&path, filtes.data(), filters.size(), nullptr); result == NFD_OKAY) {
                //         MainScene->SaveMesh(fs::path(path));
                //         NFD_FreePath(path);
                //     } else if (result != NFD_CANCEL) {
                //         throw std::runtime_error(std::format("Error saving mesh file: {}", NFD_GetError()));
                //     }
                // }
                if (MenuItem("Load RealImpact", nullptr)) {
                    static const std::vector<nfdfilteritem_t> filters{};
                    nfdchar_t *path;
                    if (auto result = NFD_PickFolder(&path, ""); result == NFD_OKAY) {
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
                            audio_device.RenderControls();
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
                // Extent changed. Update the scene texture.
                MainSceneTexture.reset(); // Ensure destruction before creation.
                MainSceneTexture = std::make_unique<ImGuiTexture>(*VC->Device, MainScene->GetResolveImageView(), vec2{0, 1}, vec2{1, 0});
            }

            const auto &cursor = GetCursorPos();
            if (MainSceneTexture) {
                const auto &scene_extent = MainScene->GetExtent();
                MainSceneTexture->Render({float(scene_extent.width), float(scene_extent.height)});
            }
            SetCursorPos(cursor);
            MainScene->RenderGizmo();
            End();
            PopStyleVar();

            if (GetFrameCount() == 1) {
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
            wd.ClearValue.color.float32[0] = ClearColor.r * ClearColor.a;
            wd.ClearValue.color.float32[1] = ClearColor.g * ClearColor.a;
            wd.ClearValue.color.float32[2] = ClearColor.b * ClearColor.a;
            wd.ClearValue.color.float32[3] = ClearColor.a;
            FrameRender(wd, draw_data);
            FramePresent(wd);
        }
    }

    // Cleanup
    NFD_Quit();

    VC->Device->waitIdle();

    R.clear();
    MainSceneTexture.reset();
    MainScene.reset();

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
