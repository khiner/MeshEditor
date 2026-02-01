#include "Scene.h"
#include "SceneDefaults.h"
#include "Widgets.h" // imgui

#include "BBox.h"
#include "Bindless.h"
#include "Entity.h"
#include "Excitable.h"
#include "OrientationGizmo.h"
#include "Shader.h"
#include "SvgResource.h"
#include "Timer.h"
#include "gpu/SilhouetteEdgeColorPushConstants.h"
#include "gpu/SilhouetteEdgeDepthObjectPushConstants.h"
#include "mesh/Primitives.h"

#include <entt/entity/registry.hpp>
#include <imgui_internal.h>

#include "Variant.h"
#include <format>
#include <unordered_set>

using std::ranges::any_of, std::ranges::all_of, std::ranges::distance, std::ranges::find, std::ranges::find_if, std::ranges::fold_left, std::ranges::to;
using std::views::filter, std::views::iota, std::views::take, std::views::transform;

namespace {
const SceneDefaults Defaults{};

using namespace he;

struct MeshSelection {
    std::unordered_set<uint32_t> Handles{};
    // Most recently selected element (may not be in Handles - active handle is remembered even when not selected)
    std::optional<uint32_t> ActiveHandle{};
};

// Tag to request overlay + element-state buffer refresh after mesh geometry changes.
struct MeshGeometryDirty {};

void WaitFor(vk::Fence fence, vk::Device device) {
    if (auto wait_result = device.waitForFences(fence, VK_TRUE, UINT64_MAX); wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    device.resetFences(fence);
}
} // namespace

#include "scene_impl/MeshRender.h"
#include "scene_impl/SceneComponents.h"

#include "scene_impl/SceneBuffers.h"
#include "scene_impl/SceneSelection.h"

#include "scene_impl/SceneTree.h"

#include "scene_impl/SceneDrawing.h"
#include "scene_impl/ScenePipelines.h"
#include "scene_impl/SceneTransformUtils.h"
#include "scene_impl/SceneUI.h"

// Tracks state during vertex grab (G key) in Edit mode. Attached to instance entity.
struct VertexGrabState {
    std::vector<std::pair<uint32_t, vec3>> StartPositions{}; // (vertex index, local position)
};

namespace {
vec3 ComputeElementLocalPosition(const Mesh &mesh, Element element, uint32_t handle) {
    if (element == Element::Vertex) return mesh.GetPosition(VH{handle});
    if (element == Element::Edge) {
        const auto heh = mesh.GetHalfedge(EH{handle}, 0);
        return (mesh.GetPosition(mesh.GetFromVertex(heh)) + mesh.GetPosition(mesh.GetToVertex(heh))) * 0.5f;
    }
    return mesh.CalcFaceCentroid(FH{handle});
}

vec3 ComputeElementWorldPosition(entt::registry &r, entt::entity instance_entity, Element element, uint32_t handle) {
    const auto &mesh = r.get<Mesh>(r.get<MeshInstance>(instance_entity).MeshEntity);
    const auto local_pos = ComputeElementLocalPosition(mesh, element, handle);
    return {r.get<WorldMatrix>(instance_entity).M * vec4{local_pos, 1.f}};
}

namespace changes {
using namespace entt::literals;
constexpr auto
    Selected = "selected_changes"_hs,
    Rerecord = "rerecord_changes"_hs,
    MeshSelection = "mesh_selection_changes"_hs,
    MeshGeometry = "mesh_geometry_changes"_hs,
    Excitable = "excitable_changes"_hs,
    ExcitedVertex = "excited_vertex_changes"_hs,
    ModelsBuffer = "models_buffer_changes"_hs,
    SceneSettings = "scene_settings_changes"_hs,
    InteractionMode = "interaction_mode_changes"_hs,
    ViewportTheme = "viewport_theme_changes"_hs,
    SceneView = "scene_view_changes"_hs;
} // namespace changes
} // namespace

struct Scene::SelectionSlotHandles {
    explicit SelectionSlotHandles(DescriptorSlots &slots)
        : Slots(slots),
          HeadImage(slots.Allocate(SlotType::Image)),
          SelectionCounter(slots.Allocate(SlotType::Buffer)),
          ClickResult(slots.Allocate(SlotType::Buffer)),
          ClickElementResult(slots.Allocate(SlotType::Buffer)),
          BoxResult(slots.Allocate(SlotType::Buffer)),
          ObjectIdSampler(slots.Allocate(SlotType::Sampler)),
          DepthSampler(slots.Allocate(SlotType::Sampler)),
          SilhouetteSampler(slots.Allocate(SlotType::Sampler)) {}

    ~SelectionSlotHandles() {
        Slots.Release({SlotType::Image, HeadImage});
        Slots.Release({SlotType::Buffer, SelectionCounter});
        Slots.Release({SlotType::Buffer, ClickResult});
        Slots.Release({SlotType::Buffer, ClickElementResult});
        Slots.Release({SlotType::Buffer, BoxResult});
        Slots.Release({SlotType::Sampler, ObjectIdSampler});
        Slots.Release({SlotType::Sampler, DepthSampler});
        Slots.Release({SlotType::Sampler, SilhouetteSampler});
    }

    DescriptorSlots &Slots;
    uint32_t HeadImage, SelectionCounter, ClickResult, ClickElementResult, BoxResult, ObjectIdSampler, DepthSampler, SilhouetteSampler;
};

// Unmanaged reactive storage to track entity destruction.
// Unlike managed storage (via R.storage<entt::reactive>()), this keeps entities after destruction
// until manually cleared, allowing ProcessComponentEvents to detect that entities were deleted.
struct Scene::EntityDestroyTracker {
    entt::storage_for_t<entt::reactive> Storage;

    void Bind(entt::registry &r) {
        Storage.bind(r);
        Storage.on_destroy<RenderInstance>();
    }
};

Scene::Scene(SceneVulkanResources vc, entt::registry &r)
    : Vk{vc},
      R{r},
      CommandPool{Vk.Device.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, Vk.QueueFamily})},
      RenderCommandBuffer{std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front())},
#ifdef MVK_FORCE_STAGED_TRANSFERS
      TransferCommandBuffer{std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front())},
#endif
      RenderFence{Vk.Device.createFenceUnique({})},
      OneShotFence{Vk.Device.createFenceUnique({})},
      SelectionReadySemaphore{Vk.Device.createSemaphoreUnique({})},
      ClickCommandBuffer{std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front())},
      Slots{std::make_unique<DescriptorSlots>(
          Vk.Device,
          Vk.PhysicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDescriptorIndexingProperties>().get<vk::PhysicalDeviceDescriptorIndexingProperties>()
      )},
      SelectionHandles{std::make_unique<SelectionSlotHandles>(*Slots)},
      DestroyTracker{std::make_unique<EntityDestroyTracker>()},
      Pipelines{std::make_unique<ScenePipelines>(Vk.Device, Vk.PhysicalDevice, Slots->GetSetLayout(), Slots->GetSet())},
      Buffers{std::make_unique<SceneBuffers>(Vk.PhysicalDevice, Vk.Device, Vk.Instance, *Slots)},
      Meshes{Buffers->Ctx} {
    // Reactive storage subscriptions for deferred once-per-frame processing
    using namespace entt::literals;

    R.storage<entt::reactive>(changes::Selected)
        .on_construct<Selected>()
        .on_destroy<Selected>();
    R.storage<entt::reactive>(changes::Rerecord)
        .on_construct<RenderInstance>()
        .on_destroy<RenderInstance>()
        .on_construct<Active>()
        .on_destroy<Active>()
        .on_construct<StartTransform>()
        .on_destroy<StartTransform>()
        .on_construct<SceneEditMode>()
        .on_update<SceneEditMode>();

    R.storage<entt::reactive>(changes::MeshSelection)
        .on_construct<MeshSelection>()
        .on_update<MeshSelection>();
    R.storage<entt::reactive>(changes::MeshGeometry)
        .on_construct<MeshGeometryDirty>()
        .on_update<MeshGeometryDirty>();
    R.storage<entt::reactive>(changes::Excitable)
        .on_construct<Excitable>()
        .on_destroy<Excitable>();
    R.storage<entt::reactive>(changes::ExcitedVertex)
        .on_construct<ExcitedVertex>()
        .on_destroy<ExcitedVertex>();
    R.storage<entt::reactive>(changes::ModelsBuffer)
        .on_update<ModelsBuffer>();
    R.storage<entt::reactive>(changes::SceneSettings)
        .on_construct<SceneSettings>()
        .on_update<SceneSettings>();
    R.storage<entt::reactive>(changes::InteractionMode)
        .on_construct<SceneInteraction>()
        .on_update<SceneInteraction>();
    R.storage<entt::reactive>(changes::ViewportTheme)
        .on_construct<ViewportTheme>()
        .on_update<ViewportTheme>();
    R.storage<entt::reactive>(changes::SceneView)
        .on_construct<Camera>()
        .on_update<Camera>()
        .on_construct<Lights>()
        .on_update<Lights>()
        .on_construct<ViewportExtent>()
        .on_update<ViewportExtent>()
        .on_construct<SceneEditMode>()
        .on_update<SceneEditMode>();

    DestroyTracker->Bind(R);

    SceneEntity = R.create();
    R.emplace<SceneSettings>(SceneEntity);
    R.emplace<SceneInteraction>(SceneEntity);
    R.emplace<SceneEditMode>(SceneEntity);
    R.emplace<ViewportTheme>(SceneEntity, Defaults.ViewportTheme);
    R.emplace<Camera>(SceneEntity, Defaults.Camera);
    R.emplace<Lights>(SceneEntity, Defaults.Lights);
    R.emplace<ViewportExtent>(SceneEntity);

    BoxSelectZeroBits.assign(SceneBuffers::BoxSelectBitsetWords, 0);

    Pipelines->CompileShaders();
}

Scene::~Scene() {
    R.clear<Mesh>();
}

World Scene::GetWorld() const { return Defaults.World; }

entt::entity Scene::GetMeshEntity(entt::entity e) const {
    if (const auto *mesh_instance = R.try_get<MeshInstance>(e)) return mesh_instance->MeshEntity;
    return entt::null;
}
entt::entity Scene::GetActiveMeshEntity() const {
    if (const auto active = FindActiveEntity(R); active != entt::null) return GetMeshEntity(active);
    return entt::null;
}

void Scene::Select(entt::entity e) {
    R.clear<Selected>();
    R.clear<Active>();
    if (e != entt::null) {
        R.emplace<Active>(e);
        R.emplace<Selected>(e);
    }
}
void Scene::ToggleSelected(entt::entity e) {
    if (e == entt::null) return;

    if (R.all_of<Selected>(e)) R.remove<Selected>(e);
    else R.emplace_or_replace<Selected>(e);
}

mvk::ImageResource Scene::RenderBitmapToImage(std::span<const std::byte> data, uint32_t width, uint32_t height) const {
    auto image = mvk::CreateImage(
        Vk.Device, Vk.PhysicalDevice,
        {{},
         vk::ImageType::e2D,
         Format::Color,
         {width, height, 1},
         1,
         1,
         vk::SampleCountFlagBits::e1,
         vk::ImageTiling::eOptimal,
         vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
         vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, Format::Color, {}, ColorSubresourceRange}
    );

    auto cb = std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    {
        // Write the bitmap into a temporary staging buffer.
        mvk::Buffer staging_buffer{Buffers->Ctx, as_bytes(data), mvk::MemoryUsage::CpuOnly};
        // Transition the image layout to be ready for data transfer.
        cb->pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer,
            {}, {}, {}, // Dependency flags, memory barriers, buffer memory barriers
            vk::ImageMemoryBarrier{
                {}, // srcAccessMask
                vk::AccessFlagBits::eTransferWrite, // dstAccessMask
                {}, // oldLayout
                vk::ImageLayout::eTransferDstOptimal, // newLayout
                {}, // srcQueueFamilyIndex
                {}, // dstQueueFamilyIndex
                *image.Image, // image
                ColorSubresourceRange // subresourceRange
            }
        );

        // Copy buffer to image.
        cb->copyBufferToImage(
            *staging_buffer, *image.Image, vk::ImageLayout::eTransferDstOptimal,
            vk::BufferImageCopy{
                0, // bufferOffset
                0, // bufferRowLength (tightly packed)
                0, // bufferImageHeight
                {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, // imageSubresource
                {0, 0, 0}, // imageOffset
                {width, height, 1} // imageExtent
            }
        );

        // Transition the image layout to be ready for shader sampling.
        cb->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader,
            {}, {}, {},
            vk::ImageMemoryBarrier{
                vk::AccessFlagBits::eTransferWrite, // srcAccessMask
                vk::AccessFlagBits::eShaderRead, // dstAccessMask
                vk::ImageLayout::eTransferDstOptimal, // oldLayout
                vk::ImageLayout::eShaderReadOnlyOptimal, // newLayout
                {}, // srcQueueFamilyIndex
                {}, // dstQueueFamilyIndex
                *image.Image, // image
                ColorSubresourceRange // subresourceRange
            }
        );
        cb->end();

        vk::SubmitInfo submit;
        submit.setCommandBuffers(*cb);
        Vk.Queue.submit(submit, *OneShotFence);
        WaitFor(*OneShotFence, Vk.Device);
    } // staging buffer is destroyed here

    Buffers->Ctx.ReclaimRetiredBuffers();

    return image;
}

void Scene::LoadIcons(vk::Device device) {
    const auto RenderBitmap = [this](std::span<const std::byte> data, uint32_t width, uint32_t height) {
        return RenderBitmapToImage(data, width, height);
    };

    static const std::filesystem::path svg_path{"res/svg/"};
    Icons.Select = std::make_unique<SvgResource>(device, RenderBitmap, svg_path / "select.svg");
    Icons.SelectBox = std::make_unique<SvgResource>(device, RenderBitmap, svg_path / "select_box.svg");
    Icons.Move = std::make_unique<SvgResource>(device, RenderBitmap, svg_path / "move.svg");
    Icons.Rotate = std::make_unique<SvgResource>(device, RenderBitmap, svg_path / "rotate.svg");
    Icons.Scale = std::make_unique<SvgResource>(device, RenderBitmap, svg_path / "scale.svg");
    Icons.Universal = std::make_unique<SvgResource>(device, RenderBitmap, svg_path / "transform.svg");
}

Scene::RenderRequest Scene::ProcessComponentEvents() {
    using namespace entt::literals;

    auto render_request = RenderRequest::None;
    auto request = [&render_request](RenderRequest req) { render_request = std::max(render_request, req); };

    if (ShaderRecompileRequested) {
        ShaderRecompileRequested = false;
        Pipelines->CompileShaders();
        request(RenderRequest::ReRecord);
    }

    { // Note: Can mutate InteractionMode, so do this first before `changes::InteractionMode` handling below.
        const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
        if (R.storage<Excitable>().empty()) {
            if (interaction_mode == InteractionMode::Excite) SetInteractionMode(*InteractionModes.begin());
            InteractionModes.erase(InteractionMode::Excite);
        } else if (!R.storage<entt::reactive>(changes::Excitable).empty()) {
            InteractionModes.insert(InteractionMode::Excite);
            if (interaction_mode == InteractionMode::Excite) request(RenderRequest::ReRecord);
            else SetInteractionMode(InteractionMode::Excite); // Switch to excite mode
        }
    }
    std::unordered_set<entt::entity> dirty_overlay_meshes, dirty_element_state_meshes;
    { // Selected changes
        auto &selected_tracker = R.storage<entt::reactive>(changes::Selected);
        if (!selected_tracker.empty()) request(RenderRequest::ReRecord);
        for (auto instance_entity : selected_tracker) {
            if (auto *mi = R.try_get<MeshInstance>(instance_entity)) {
                const auto mesh_entity = mi->MeshEntity;
                if (R.all_of<Selected>(instance_entity)) {
                    dirty_overlay_meshes.insert(mesh_entity);
                } else if (!HasSelectedInstance(R, mesh_entity)) {
                    // Clean up overlays for this mesh
                    if (auto *buffers = R.try_get<MeshBuffers>(mesh_entity)) {
                        for (auto &[_, rb] : buffers->NormalIndicators) Buffers->Release(rb);
                        buffers->NormalIndicators.clear();
                    }
                    if (auto *bbox = R.try_get<BoundingBoxesBuffers>(mesh_entity)) Buffers->Release(bbox->Buffers);
                    R.remove<BoundingBoxesBuffers>(mesh_entity);
                }
            }
        }
    }
    if (!R.storage<entt::reactive>(changes::Rerecord).empty() || !DestroyTracker->Storage.empty()) {
        request(RenderRequest::ReRecord);
    }

    const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
    const auto orbit_to_active = [&](entt::entity instance_entity, Element element, uint32_t handle) {
        if (!OrbitToActive) return;
        const auto world_pos = ComputeElementWorldPosition(R, instance_entity, element, handle);
        R.patch<Camera>(SceneEntity, [&](auto &camera) {
            if (const auto dir = world_pos - camera.Target; glm::dot(dir, dir) >= 1e-6f) {
                camera.SetTargetDirection(glm::normalize(dir));
            }
        });
    };
    if (const auto &mesh_selection_tracker = R.storage<entt::reactive>(changes::MeshSelection); !mesh_selection_tracker.empty()) {
        const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
        const auto active_entity = FindActiveEntity(R);
        for (auto mesh_entity : mesh_selection_tracker) {
            if (const auto *selection = R.try_get<MeshSelection>(mesh_entity)) {
                dirty_element_state_meshes.insert(mesh_entity);
                if (selection->ActiveHandle && edit_mode != Element::None && R.all_of<MeshInstance>(active_entity) &&
                    R.get<MeshInstance>(active_entity).MeshEntity == mesh_entity) {
                    orbit_to_active(active_entity, edit_mode, *selection->ActiveHandle);
                }
            }
        }
    }
    if (auto &mesh_geometry_tracker = R.storage<entt::reactive>(changes::MeshGeometry); !mesh_geometry_tracker.empty()) {
        for (auto mesh_entity : mesh_geometry_tracker) {
            if (R.all_of<Selected>(mesh_entity) || HasSelectedInstance(R, mesh_entity)) {
                dirty_overlay_meshes.insert(mesh_entity);
            }
        }
        R.clear<MeshGeometryDirty>();
        request(RenderRequest::Submit);
    }
    for (auto instance_entity : R.storage<entt::reactive>(changes::ExcitedVertex)) {
        if (const auto *mi = R.try_get<MeshInstance>(instance_entity)) {
            dirty_element_state_meshes.insert(mi->MeshEntity);
        }
        if (const auto *ev = R.try_get<ExcitedVertex>(instance_entity)) {
            orbit_to_active(instance_entity, Element::Vertex, ev->Vertex);
        }
    }
    if (!R.storage<entt::reactive>(changes::ModelsBuffer).empty()) request(RenderRequest::Submit);
    if (!R.storage<entt::reactive>(changes::ViewportTheme).empty()) {
        Buffers->ViewportThemeUBO.Update(as_bytes(R.get<const ViewportTheme>(SceneEntity)));
        request(RenderRequest::Submit);
    }

    bool scene_view_dirty = false;
    if (!R.storage<entt::reactive>(changes::SceneSettings).empty()) {
        request(RenderRequest::ReRecord);
        scene_view_dirty = true;
        for (const auto selected_entity : R.view<Selected>()) dirty_overlay_meshes.insert(R.get<MeshInstance>(selected_entity).MeshEntity);
    }
    if (!R.storage<entt::reactive>(changes::InteractionMode).empty()) {
        request(RenderRequest::ReRecord);
        scene_view_dirty = true;
        for (const auto [mesh_entity, selection] : R.view<MeshSelection>().each()) {
            if (!selection.Handles.empty()) dirty_element_state_meshes.insert(mesh_entity);
        }
        for (const auto [_, mi, __] : R.view<const MeshInstance, const Excitable>().each()) {
            dirty_element_state_meshes.insert(mi.MeshEntity);
        }
    }
    if (!R.storage<entt::reactive>(changes::SceneView).empty()) scene_view_dirty = true;
    if (scene_view_dirty) {
        const auto &camera = R.get<const Camera>(SceneEntity);
        const auto &lights = R.get<const Lights>(SceneEntity);
        const auto extent = R.get<const ViewportExtent>(SceneEntity).Value;
        Buffers->SceneViewUBO.Update(as_bytes(SceneViewUBO{
            .View = camera.View(),
            .Proj = camera.Projection(extent.width == 0 || extent.height == 0 ? 1.f : float(extent.width) / float(extent.height)),
            .CameraPosition = camera.Position(),
            .CameraNear = camera.NearClip,
            .CameraFar = camera.FarClip,
            .ViewColor = lights.ViewColor,
            .AmbientIntensity = lights.AmbientIntensity,
            .DirectionalColor = lights.DirectionalColor,
            .DirectionalIntensity = lights.DirectionalIntensity,
            .LightDirection = lights.Direction,
            .InteractionMode = uint32_t(interaction_mode),
            .EditElement = uint32_t(R.get<const SceneEditMode>(SceneEntity).Value),
        }));
        request(RenderRequest::Submit);
    }

    // Update selection overlays
    const auto &settings = R.get<const SceneSettings>(SceneEntity);
    for (const auto mesh_entity : dirty_overlay_meshes) {
        const auto &mesh = R.get<const Mesh>(mesh_entity);
        R.patch<MeshBuffers>(mesh_entity, [&](auto &mesh_buffers) {
            for (const auto element : NormalElements) {
                if (ElementMaskContains(settings.NormalOverlays, element)) {
                    if (!mesh_buffers.NormalIndicators.contains(element)) {
                        const auto index_kind = element == Element::Face ? IndexKind::Face : IndexKind::Vertex;
                        mesh_buffers.NormalIndicators.emplace(
                            element,
                            Buffers->CreateRenderBuffers(CreateNormalVertices(mesh, element), CreateNormalIndices(mesh, element), index_kind)
                        );
                    } else {
                        Buffers->VertexBuffer.Update(mesh_buffers.NormalIndicators.at(element).Vertices, CreateNormalVertices(mesh, element));
                    }
                } else if (mesh_buffers.NormalIndicators.contains(element)) {
                    Buffers->Release(mesh_buffers.NormalIndicators.at(element));
                    mesh_buffers.NormalIndicators.erase(element);
                }
            }
        });
        if (settings.ShowBoundingBoxes) {
            static const auto create_bbox = [](const Mesh &mesh) {
                BBox bbox;
                for (const auto vh : mesh.vertices()) {
                    const auto p = mesh.GetPosition(vh);
                    bbox.Min = glm::min(bbox.Min, p);
                    bbox.Max = glm::max(bbox.Max, p);
                }
                return bbox;
            };
            const auto box_vertices = create_bbox(mesh).Corners() |
                transform([](const auto &corner) { return Vertex{corner, vec3{}}; }) |
                to<std::vector>();
            if (!R.all_of<BoundingBoxesBuffers>(mesh_entity)) {
                R.emplace<BoundingBoxesBuffers>(mesh_entity, Buffers->CreateRenderBuffers(box_vertices, BBox::EdgeIndices, IndexKind::Edge));
            } else {
                Buffers->VertexBuffer.Update(R.get<BoundingBoxesBuffers>(mesh_entity).Buffers.Vertices, box_vertices);
            }
        } else if (R.all_of<BoundingBoxesBuffers>(mesh_entity)) {
            Buffers->Release(R.get<BoundingBoxesBuffers>(mesh_entity).Buffers);
            R.remove<BoundingBoxesBuffers>(mesh_entity);
        }
    }
    // Update mesh element state buffers
    const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
    for (const auto mesh_entity : dirty_element_state_meshes) {
        const auto &mesh = R.get<Mesh>(mesh_entity);
        std::unordered_set<VH> selected_vertices;
        std::unordered_set<EH> selected_edges, active_edges;
        std::unordered_set<FH> selected_faces;

        auto element{Element::None};
        std::unordered_set<uint32_t> handles;
        std::optional<uint32_t> active_handle;
        if (interaction_mode == InteractionMode::Excite) {
            element = Element::Vertex;
            for (auto [entity, mi, excitable] : R.view<const MeshInstance, const Excitable>().each()) {
                if (mi.MeshEntity != mesh_entity) continue;
                handles = excitable.ExcitableVertices | to<std::unordered_set>();
                selected_vertices.insert(excitable.ExcitableVertices.begin(), excitable.ExcitableVertices.end());
                if (const auto *excited_vertex = R.try_get<ExcitedVertex>(entity)) active_handle = excited_vertex->Vertex;
                break;
            }
        } else if (const auto &selection = R.get<const MeshSelection>(mesh_entity);
                   interaction_mode == InteractionMode::Edit && HasSelectedInstance(R, mesh_entity)) {
            element = edit_mode;
            handles = selection.Handles;
            active_handle = selection.ActiveHandle;
            if (element == Element::Vertex) {
                selected_vertices.insert(selection.Handles.begin(), selection.Handles.end());
            } else if (element == Element::Edge) {
                selected_edges.insert(selection.Handles.begin(), selection.Handles.end());
            } else if (element == Element::Face) {
                for (auto h : selection.Handles) {
                    selected_faces.emplace(h);
                    for (const auto heh : mesh.fh_range(FH{h})) {
                        selected_edges.emplace(mesh.GetEdge(heh));
                        if (active_handle == h) {
                            for (const auto heh : mesh.fh_range(FH{*active_handle})) active_edges.emplace(mesh.GetEdge(heh));
                        }
                    }
                }
            }
        }

        // Update element states in-place
        const auto &state_buffers = R.get<MeshElementStateBuffers>(mesh_entity);
        auto face_states = Buffers->FaceStateBuffer.GetMutable(state_buffers.Faces.Range);
        auto edge_states = Buffers->EdgeStateBuffer.GetMutable(state_buffers.Edges.Range);
        auto vertex_states = Buffers->VertexStateBuffer.GetMutable(state_buffers.Vertices.Range);

        // Clear all states
        std::ranges::fill(face_states, 0u);
        std::ranges::fill(edge_states, 0u);
        std::ranges::fill(vertex_states, 0u);

        // Note: An element must be both active and selected to be displayed as active.
        if (element == Element::Face) {
            for (const auto fh : selected_faces) {
                face_states[*fh] |= ElementStateSelected;
                if (active_handle == *fh) face_states[*active_handle] |= ElementStateActive;
            }
        }

        if (element == Element::Edge || element == Element::Face) {
            for (uint32_t ei = 0; ei < mesh.EdgeCount(); ++ei) {
                uint32_t state = 0;
                if (selected_edges.contains(EH{ei})) {
                    state |= ElementStateSelected;
                    if ((element == Element::Edge && active_handle == ei) || active_edges.contains(EH{ei})) {
                        state |= ElementStateActive;
                    }
                }
                edge_states[2 * ei] = edge_states[2 * ei + 1] = state;
            }
        } else if (element == Element::Vertex) {
            for (uint32_t ei = 0; ei < mesh.EdgeCount(); ++ei) {
                const auto heh = mesh.GetHalfedge(EH{ei}, 0);
                edge_states[2 * ei] = selected_vertices.contains(mesh.GetFromVertex(heh)) ? ElementStateSelected : 0u;
                edge_states[2 * ei + 1] = selected_vertices.contains(mesh.GetToVertex(heh)) ? ElementStateSelected : 0u;
            }
        }

        if (element == Element::Vertex) {
            for (const auto vh : selected_vertices) {
                vertex_states[*vh] |= ElementStateSelected;
                if (active_handle == *vh) vertex_states[*active_handle] |= ElementStateActive;
            }
        }

        SelectionStale = true;
    }
    if (!dirty_element_state_meshes.empty()) request(RenderRequest::Submit);
    for (auto &&[id, storage] : R.storage()) {
        if (storage.info() == entt::type_id<entt::reactive>()) storage.clear();
    }
    DestroyTracker->Storage.clear();

    return render_request;
}

vk::Extent2D Scene::GetExtent() const { return R.get<const ViewportExtent>(SceneEntity).Value; }
vk::ImageView Scene::GetViewportImageView() const { return *Pipelines->Main.Resources->ResolveImage.View; }

void Scene::SetVisible(entt::entity entity, bool visible) {
    const bool already_visible = R.all_of<RenderInstance>(entity);
    if ((visible && already_visible) || (!visible && !already_visible)) return;

    const auto mesh_entity = R.get<MeshInstance>(entity).MeshEntity;
    if (visible) {
        const auto buffer_index = R.get<const ModelsBuffer>(mesh_entity).Buffer.UsedSize / sizeof(WorldMatrix);
        const uint32_t object_id = NextObjectId++;
        R.emplace<RenderInstance>(entity, buffer_index, object_id);
        R.patch<ModelsBuffer>(mesh_entity, [&](auto &mb) {
            mb.Buffer.Insert(as_bytes(R.get<WorldMatrix>(entity)), mb.Buffer.UsedSize);
            mb.ObjectIds.Insert(as_bytes(object_id), mb.ObjectIds.UsedSize);
        });
    } else {
        const uint old_model_index = R.get<const RenderInstance>(entity).BufferIndex;
        R.remove<RenderInstance>(entity);
        R.patch<ModelsBuffer>(mesh_entity, [old_model_index](auto &mb) {
            mb.Buffer.Erase(old_model_index * sizeof(WorldMatrix), sizeof(WorldMatrix));
            mb.ObjectIds.Erase(old_model_index * sizeof(uint32_t), sizeof(uint32_t));
        });
        // Update buffer indices for all instances of this mesh that have higher indices
        for (const auto [other_entity, mesh_instance, ri] : R.view<MeshInstance, const RenderInstance>().each()) {
            if (mesh_instance.MeshEntity == mesh_entity && ri.BufferIndex > old_model_index) {
                R.patch<RenderInstance>(other_entity, [](auto &ri) { --ri.BufferIndex; });
            }
        }
        // Also check if the mesh entity itself is visible (it might not have MeshInstance)
        if (mesh_entity != entity) {
            if (const auto *mesh_ri = R.try_get<const RenderInstance>(mesh_entity)) {
                if (mesh_ri->BufferIndex > old_model_index) {
                    R.patch<RenderInstance>(mesh_entity, [](auto &ri) { --ri.BufferIndex; });
                }
            }
        }
    }
}

std::pair<entt::entity, entt::entity> Scene::AddMesh(Mesh &&mesh, std::optional<MeshInstanceCreateInfo> info) {
    const auto mesh_entity = R.create();
    R.emplace<ModelsBuffer>(
        mesh_entity,
        mvk::Buffer{Buffers->Ctx, sizeof(WorldMatrix), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ModelBuffer},
        mvk::Buffer{Buffers->Ctx, sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ObjectIdBuffer}
    );
    R.emplace<MeshSelection>(mesh_entity);
    R.emplace<MeshBuffers>(
        mesh_entity, SlottedBufferRange{Meshes.GetVerticesRange(mesh.GetStoreId()), Meshes.GetVerticesSlot()},
        Buffers->CreateIndices(mesh.CreateTriangleIndices(), IndexKind::Face),
        Buffers->CreateIndices(mesh.CreateEdgeIndices(), IndexKind::Edge),
        Buffers->CreateIndices(CreateVertexIndices(mesh), IndexKind::Vertex)
    );
    R.emplace<MeshElementStateBuffers>(
        mesh_entity,
        Buffers->AllocateFaceStates(mesh.FaceCount()),
        Buffers->AllocateEdgeStates(mesh.EdgeCount() * 2),
        Buffers->AllocateVertexStates(mesh.VertexCount())
    );
    R.emplace<Mesh>(mesh_entity, std::move(mesh));
    return {mesh_entity, info ? AddMeshInstance(mesh_entity, *info) : entt::null};
}

entt::entity Scene::AddMeshInstance(entt::entity mesh_entity, MeshInstanceCreateInfo info) {
    const auto instance_entity = R.create();
    R.emplace<MeshInstance>(instance_entity, mesh_entity);
    SetTransform(R, instance_entity, info.Transform);
    R.emplace<Name>(instance_entity, CreateName(R, info.Name));

    R.patch<ModelsBuffer>(mesh_entity, [](auto &mb) {
        mb.Buffer.Reserve(mb.Buffer.UsedSize + sizeof(WorldMatrix));
        mb.ObjectIds.Reserve(mb.ObjectIds.UsedSize + sizeof(uint32_t));
    });
    SetVisible(instance_entity, true); // Always set visibility to true first, since this sets up the model buffer/indices.
    if (!info.Visible) SetVisible(instance_entity, false);

    switch (info.Select) {
        case MeshInstanceCreateInfo::SelectBehavior::Exclusive:
            Select(instance_entity);
            break;
        case MeshInstanceCreateInfo::SelectBehavior::Additive:
            R.emplace<Selected>(instance_entity);
            // Fallthrough
        case MeshInstanceCreateInfo::SelectBehavior::None:
            // If no mesh is active yet, activate the new one.
            if (R.storage<Active>().empty()) {
                R.emplace<Active>(instance_entity);
                R.emplace_or_replace<Selected>(instance_entity);
            }
            break;
    }

    return instance_entity;
}

std::pair<entt::entity, entt::entity> Scene::AddMesh(MeshData &&data, std::optional<MeshInstanceCreateInfo> info) {
    return AddMesh(Meshes.CreateMesh(std::move(data)), std::move(info));
}

std::pair<entt::entity, entt::entity> Scene::AddMesh(const std::filesystem::path &path, std::optional<MeshInstanceCreateInfo> info) {
    auto mesh = Meshes.LoadMesh(path);
    if (!mesh) throw std::runtime_error(std::format("Failed to load mesh: {}", path.string()));
    const auto e = AddMesh(std::move(*mesh), std::move(info));
    R.emplace<Path>(e.first, path);
    return e;
}

entt::entity Scene::Duplicate(entt::entity e, std::optional<MeshInstanceCreateInfo> info) {
    const auto mesh_entity = R.get<MeshInstance>(e).MeshEntity;
    const auto e_new = AddMesh(
        Meshes.CloneMesh(R.get<const Mesh>(mesh_entity)),
        info.value_or(MeshInstanceCreateInfo{
            .Name = std::format("{}_copy", GetName(R, e)),
            .Transform = GetTransform(R, e),
            .Select = R.all_of<Selected>(e) ? MeshInstanceCreateInfo::SelectBehavior::Additive : MeshInstanceCreateInfo::SelectBehavior::None,
            .Visible = R.all_of<RenderInstance>(e),
        })
    );
    if (auto primitive_type = R.try_get<PrimitiveType>(mesh_entity)) R.emplace<PrimitiveType>(e_new.first, *primitive_type);
    return e_new.second;
}

entt::entity Scene::DuplicateLinked(entt::entity e, std::optional<MeshInstanceCreateInfo> info) {
    const auto mesh_entity = R.get<MeshInstance>(e).MeshEntity;
    const auto e_new = R.create();
    {
        uint instance_count = 0; // Count instances for naming (first duplicated instance is _1, etc.)
        for (const auto [_, mesh_instance] : R.view<MeshInstance>().each()) {
            if (mesh_instance.MeshEntity == mesh_entity) ++instance_count;
        }
        R.emplace<Name>(e_new, !info || info->Name.empty() ? std::format("{}_{}", GetName(R, e), instance_count) : CreateName(R, info->Name));
    }
    R.emplace<MeshInstance>(e_new, mesh_entity);
    R.patch<ModelsBuffer>(mesh_entity, [](auto &mb) {
        mb.Buffer.Reserve(mb.Buffer.UsedSize + sizeof(WorldMatrix));
        mb.ObjectIds.Reserve(mb.ObjectIds.UsedSize + sizeof(uint32_t));
    });
    SetTransform(R, e_new, info ? info->Transform : GetTransform(R, e));
    SetVisible(e_new, !info || info->Visible);

    if (!info || info->Select == MeshInstanceCreateInfo::SelectBehavior::Additive) R.emplace<Selected>(e_new);
    else if (info->Select == MeshInstanceCreateInfo::SelectBehavior::Exclusive) Select(e_new);

    return e_new;
}

void Scene::Delete() {
    std::vector<entt::entity> entities;
    for (const auto e : R.view<Selected>()) entities.emplace_back(e);
    for (const auto e : entities) Destroy(e);
}

void Scene::Duplicate() {
    const Timer timer{"Duplicate"};
    std::vector<entt::entity> entities;
    for (const auto e : R.view<Selected>()) entities.emplace_back(e);
    for (const auto e : entities) {
        const auto new_e = Duplicate(e);
        if (R.all_of<Active>(e)) {
            R.remove<Active>(e);
            R.emplace<Active>(new_e);
            R.emplace_or_replace<Selected>(new_e);
        }
        R.remove<Selected>(e);
    }
    StartScreenTransform = TransformGizmo::TransformType::Translate;
}
void Scene::DuplicateLinked() {
    const Timer timer{"DuplicateLinked"};
    std::vector<entt::entity> entities;
    for (const auto e : R.view<Selected>()) entities.emplace_back(e);
    for (const auto e : entities) {
        const auto new_e = DuplicateLinked(e);
        if (R.all_of<Active>(e)) {
            R.remove<Active>(e);
            R.emplace<Active>(new_e);
            R.emplace_or_replace<Selected>(new_e);
        }
        R.remove<Selected>(e);
    }
    StartScreenTransform = TransformGizmo::TransformType::Translate;
}

void Scene::ClearMeshes() {
    std::vector<entt::entity> entities;
    for (const auto e : R.view<MeshInstance>()) entities.emplace_back(e);
    for (const auto e : entities) Destroy(e);
}

void Scene::SetMeshPositions(entt::entity e, std::span<const vec3> positions) {
    Meshes.SetPositions(R.get<const Mesh>(e), positions);
    R.emplace_or_replace<MeshGeometryDirty>(e);
}

void Scene::Destroy(entt::entity e) {
    { // Clear relationships
        ClearParent(R, e);
        std::vector<entt::entity> children;
        for (auto child : Children{&R, e}) children.emplace_back(child);
        for (const auto child : children) ClearParent(R, child);
    }

    // Track mesh entity if this is an instance
    entt::entity mesh_entity = entt::null;
    if (R.all_of<MeshInstance>(e)) {
        mesh_entity = R.get<MeshInstance>(e).MeshEntity;
        SetVisible(e, false);
    }

    if (R.valid(e)) R.destroy(e);

    // If this was the last instance, destroy the mesh
    if (R.valid(mesh_entity)) {
        const auto has_instances = any_of(
            R.view<MeshInstance>().each(),
            [mesh_entity](const auto &entry) { return std::get<1>(entry).MeshEntity == mesh_entity; }
        );
        if (!has_instances) {
            if (auto *mesh_buffers = R.try_get<MeshBuffers>(mesh_entity)) Buffers->Release(*mesh_buffers);
            if (auto *state_buffers = R.try_get<MeshElementStateBuffers>(mesh_entity)) Buffers->Release(*state_buffers);
            R.destroy(mesh_entity);
        }
    }
}

void Scene::SetInteractionMode(InteractionMode mode) {
    if (R.get<const SceneInteraction>(SceneEntity).Mode == mode) return;

    R.patch<SceneInteraction>(SceneEntity, [mode](auto &s) { s.Mode = mode; });
    R.patch<ViewportTheme>(SceneEntity, [](auto &) {});
}

void Scene::SetEditMode(Element mode) {
    const auto current_mode = R.get<const SceneEditMode>(SceneEntity).Value;
    if (current_mode == mode) return;

    for (const auto &[e, selection, mesh] : R.view<MeshSelection, Mesh>().each()) {
        R.patch<MeshSelection>(e, [&](auto &selection) {
            selection.Handles = ConvertSelectionElement(selection, mesh, current_mode, mode);
            selection.ActiveHandle = {}; // todo not quite right
        });
    }
    R.patch<SceneEditMode>(SceneEntity, [mode](auto &edit_mode) { edit_mode.Value = mode; });
}

std::string Scene::DebugBufferHeapUsage() const { return Buffers->Ctx.DebugHeapUsage(); }

void Scene::RecordRenderCommandBuffer() {
    SelectionStale = true;
    // In Edit mode, only primary edit instance per selected mesh gets Edit visuals.
    // Other selected instances render normally with silhouettes.
    const auto &settings = R.get<const SceneSettings>(SceneEntity);
    const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
    const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
    const bool is_edit_mode = interaction_mode == InteractionMode::Edit;
    const bool is_excite_mode = interaction_mode == InteractionMode::Excite;
    const bool show_solid = settings.ViewportShading == ViewportShadingMode::Solid;
    const bool show_wireframe = settings.ViewportShading == ViewportShadingMode::Wireframe;
    const SPT fill_pipeline = settings.FaceColorMode == FaceColorMode::Mesh ? SPT::Fill : SPT::DebugNormals;
    const auto primary_edit_instances = is_edit_mode ? ComputePrimaryEditInstances(R) : std::unordered_map<entt::entity, entt::entity>{};
    std::unordered_set<entt::entity> silhouette_instances;
    if (is_edit_mode) {
        for (const auto [e, mi, ri] : R.view<const MeshInstance, const Selected, const RenderInstance>().each()) {
            if (primary_edit_instances.at(mi.MeshEntity) != e) silhouette_instances.insert(e);
        }
    }

    std::unordered_set<entt::entity> excitable_mesh_entities;
    if (is_excite_mode) {
        for (const auto [e, mi, excitable] : R.view<const MeshInstance, const Excitable>().each()) {
            excitable_mesh_entities.emplace(mi.MeshEntity);
        }
    }

    const bool render_silhouette = !R.view<Selected>().empty() &&
        (interaction_mode == InteractionMode::Object || !silhouette_instances.empty());

    DrawListBuilder draw_list;
    DrawBatchInfo fill_batch{}, line_batch{}, point_batch{};
    DrawBatchInfo silhouette_batch{};
    DrawBatchInfo overlay_face_normals_batch{}, overlay_vertex_normals_batch{}, overlay_bbox_batch{};

    if (render_silhouette) {
        silhouette_batch = draw_list.BeginBatch();
        auto append_silhouette = [&](entt::entity e) {
            const auto mesh_entity = R.get<MeshInstance>(e).MeshEntity;
            const auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
            const auto &models = R.get<ModelsBuffer>(mesh_entity);
            auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models);
            draw.ObjectIdSlot = models.ObjectIds.Slot;
            AppendDraw(draw_list, silhouette_batch, mesh_buffers.FaceIndices, models, draw, R.get<RenderInstance>(e).BufferIndex);
        };
        if (is_edit_mode) {
            for (const auto e : silhouette_instances) append_silhouette(e);
        } else {
            for (const auto e : R.view<Selected, RenderInstance>()) append_silhouette(e);
        }
    }

    if (show_solid) {
        fill_batch = draw_list.BeginBatch();
        for (auto [entity, mesh_buffers, models, mesh, state_buffers] :
             R.view<MeshBuffers, ModelsBuffer, Mesh, MeshElementStateBuffers>().each()) {
            auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models);
            const auto face_id_range = Meshes.GetFaceIdRange(mesh.GetStoreId());
            const auto face_normal_range = Meshes.GetFaceNormalRange(mesh.GetStoreId());
            draw.ObjectIdSlot = Meshes.GetFaceIdSlot();
            draw.FaceIdOffset = face_id_range.Offset;
            draw.FaceNormalSlot = settings.SmoothShading ? InvalidSlot : Meshes.GetFaceNormalSlot();
            draw.FaceNormalOffset = face_normal_range.Offset;
            if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                // Draw primary with element state first, then all without (depth LESS won't overwrite)
                draw.ElementStateSlot = state_buffers.Faces.Slot;
                draw.ElementStateOffset = state_buffers.Faces.Range.Offset;
                AppendDraw(draw_list, fill_batch, mesh_buffers.FaceIndices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
                draw.ElementStateSlot = InvalidSlot;
                AppendDraw(draw_list, fill_batch, mesh_buffers.FaceIndices, models, draw);
            } else {
                draw.ElementStateSlot = state_buffers.Faces.Slot;
                draw.ElementStateOffset = state_buffers.Faces.Range.Offset;
                AppendDraw(draw_list, fill_batch, mesh_buffers.FaceIndices, models, draw);
            }
        }
    }

    if (show_wireframe || is_edit_mode || is_excite_mode) {
        line_batch = draw_list.BeginBatch();
        for (auto [entity, mesh_buffers, models, state_buffers] :
             R.view<MeshBuffers, ModelsBuffer, MeshElementStateBuffers>().each()) {
            auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.EdgeIndices, models);
            draw.ElementStateSlot = state_buffers.Edges.Slot;
            draw.ElementStateOffset = state_buffers.Edges.Range.Offset;
            if (show_wireframe) {
                AppendDraw(draw_list, line_batch, mesh_buffers.EdgeIndices, models, draw);
            } else if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                AppendDraw(draw_list, line_batch, mesh_buffers.EdgeIndices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
            } else if (excitable_mesh_entities.contains(entity)) {
                AppendDraw(draw_list, line_batch, mesh_buffers.EdgeIndices, models, draw);
            }
        }
    }

    if ((is_edit_mode && edit_mode == Element::Vertex) || is_excite_mode) {
        point_batch = draw_list.BeginBatch();
        for (auto [entity, mesh_buffers, models, state_buffers] :
             R.view<MeshBuffers, ModelsBuffer, MeshElementStateBuffers>().each()) {
            auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.VertexIndices, models);
            draw.ElementStateSlot = state_buffers.Vertices.Slot;
            draw.ElementStateOffset = state_buffers.Vertices.Range.Offset;
            if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                AppendDraw(draw_list, point_batch, mesh_buffers.VertexIndices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
            } else if (excitable_mesh_entities.contains(entity)) {
                AppendDraw(draw_list, point_batch, mesh_buffers.VertexIndices, models, draw);
            }
        }
    }

    {
        const auto vertex_slot = Buffers->VertexBuffer.Buffer.Slot;
        overlay_face_normals_batch = draw_list.BeginBatch();
        for (auto [_, mesh_buffers, models] : R.view<MeshBuffers, ModelsBuffer>().each()) {
            if (auto it = mesh_buffers.NormalIndicators.find(Element::Face); it != mesh_buffers.NormalIndicators.end()) {
                auto draw = MakeDrawData(it->second, vertex_slot, models);
                AppendDraw(draw_list, overlay_face_normals_batch, it->second.Indices, models, draw);
            }
        }

        overlay_vertex_normals_batch = draw_list.BeginBatch();
        for (auto [_, mesh_buffers, models] : R.view<MeshBuffers, ModelsBuffer>().each()) {
            if (auto it = mesh_buffers.NormalIndicators.find(Element::Vertex); it != mesh_buffers.NormalIndicators.end()) {
                auto draw = MakeDrawData(it->second, vertex_slot, models);
                AppendDraw(draw_list, overlay_vertex_normals_batch, it->second.Indices, models, draw);
            }
        }

        overlay_bbox_batch = draw_list.BeginBatch();
        for (auto [_, bounding_boxes, models] : R.view<BoundingBoxesBuffers, ModelsBuffer>().each()) {
            auto draw = MakeDrawData(bounding_boxes.Buffers, vertex_slot, models);
            AppendDraw(draw_list, overlay_bbox_batch, bounding_boxes.Buffers.Indices, models, draw);
        }
    }

    if (!draw_list.Draws.empty()) Buffers->RenderDrawData.Update(as_bytes(draw_list.Draws));
    if (!draw_list.IndirectCommands.empty()) Buffers->RenderIndirect.Update(as_bytes(draw_list.IndirectCommands));
    Buffers->EnsureIdentityIndexBuffer(draw_list.MaxIndexCount);
    if (auto descriptor_updates = Buffers->Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
        Vk.Device.updateDescriptorSets(std::move(descriptor_updates), {});
        Buffers->Ctx.ClearDeferredDescriptorUpdates();
    }

    const auto extent = R.get<const ViewportExtent>(SceneEntity).Value;
    const auto &cb = *RenderCommandBuffer;
    cb.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(extent.width), float(extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, extent});
    auto record_draw_batch = [&](const PipelineRenderer &renderer, SPT spt, const DrawBatchInfo &batch) {
        if (batch.DrawCount == 0) return;
        const auto &pipeline = renderer.Bind(cb, spt);
        const DrawPassPushConstants pc{Buffers->RenderDrawData.Slot, batch.DrawDataOffset};
        cb.pushConstants(*pipeline.PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        cb.drawIndexedIndirect(*Buffers->RenderIndirect, batch.IndirectOffset, batch.DrawCount, sizeof(vk::DrawIndexedIndirectCommand));
    };

    if (render_silhouette) { // Silhouette depth/object pass
        const auto &silhouette = Pipelines->Silhouette;
        static const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {Transparent}};
        const vk::Rect2D rect{{0, 0}, ToExtent2D(silhouette.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*silhouette.Renderer.RenderPass, *silhouette.Resources->Framebuffer, rect, clear_values}, vk::SubpassContents::eInline);
        cb.bindIndexBuffer(*Buffers->IdentityIndexBuffer, 0, vk::IndexType::eUint32);
        record_draw_batch(silhouette.Renderer, SPT::SilhouetteDepthObject, silhouette_batch);
        cb.endRenderPass();

        const auto &silhouette_edge = Pipelines->SilhouetteEdge;
        static const std::vector<vk::ClearValue> edge_clear_values{{vk::ClearDepthStencilValue{1, 0}}, {Transparent}};
        const vk::Rect2D edge_rect{{0, 0}, ToExtent2D(silhouette_edge.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*silhouette_edge.Renderer.RenderPass, *silhouette_edge.Resources->Framebuffer, edge_rect, edge_clear_values}, vk::SubpassContents::eInline);
        const auto &silhouette_edo = silhouette_edge.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeDepthObject);
        const SilhouetteEdgeDepthObjectPushConstants edge_pc{SelectionHandles->SilhouetteSampler};
        cb.pushConstants(*silhouette_edo.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(edge_pc), &edge_pc);
        silhouette_edo.RenderQuad(cb);
        cb.endRenderPass();
    }

    const auto &main = Pipelines->Main;
    // Main rendering pass
    {
        const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {settings.ClearColor}};
        const vk::Rect2D rect{{0, 0}, ToExtent2D(main.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*main.Renderer.RenderPass, *main.Resources->Framebuffer, rect, clear_values}, vk::SubpassContents::eInline);
    }

    // Silhouette edge depth (not color! we render it before mesh depth to avoid overwriting closer depths with further ones)
    if (render_silhouette) {
        const auto &silhouette_depth = main.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeDepth);
        const uint32_t depth_sampler_index = SelectionHandles->DepthSampler;
        cb.pushConstants(*silhouette_depth.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(depth_sampler_index), &depth_sampler_index);
        silhouette_depth.RenderQuad(cb);
    }

    { // Meshes
        cb.bindIndexBuffer(*Buffers->IdentityIndexBuffer, 0, vk::IndexType::eUint32);
        // Solid faces
        if (show_solid) record_draw_batch(main.Renderer, fill_pipeline, fill_batch);
        // Wireframe edges
        if (show_wireframe || is_edit_mode || is_excite_mode) record_draw_batch(main.Renderer, SPT::Line, line_batch);
        // Vertex points
        if ((is_edit_mode && edit_mode == Element::Vertex) || is_excite_mode) record_draw_batch(main.Renderer, SPT::Point, point_batch);
    }

    // Silhouette edge color (rendered ontop of meshes)
    if (render_silhouette) {
        const auto &silhouette_edc = main.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeColor);
        // In Edit mode, never show active silhouette - only selected (non-active) silhouettes
        uint32_t active_object_id = 0;
        if (!is_edit_mode) {
            const auto active_entity = FindActiveEntity(R);
            active_object_id = active_entity != entt::null && R.all_of<RenderInstance>(active_entity) ?
                R.get<RenderInstance>(active_entity).ObjectId :
                0;
        }
        const SilhouetteEdgeColorPushConstants pc{TransformGizmo::IsUsing(), SelectionHandles->ObjectIdSampler, active_object_id};
        cb.pushConstants(*silhouette_edc.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        silhouette_edc.RenderQuad(cb);
    }

    { // Selection overlays
        cb.bindIndexBuffer(*Buffers->IdentityIndexBuffer, 0, vk::IndexType::eUint32);
        record_draw_batch(main.Renderer, SPT::LineOverlayFaceNormals, overlay_face_normals_batch);
        record_draw_batch(main.Renderer, SPT::LineOverlayVertexNormals, overlay_vertex_normals_batch);
        record_draw_batch(main.Renderer, SPT::LineOverlayBBox, overlay_bbox_batch);
    }

    // Grid lines texture
    if (settings.ShowGrid) main.Renderer.ShaderPipelines.at(SPT::Grid).RenderQuad(cb);

    cb.endRenderPass();
    cb.end();
}

#ifdef MVK_FORCE_STAGED_TRANSFERS
void Scene::RecordTransferCommandBuffer() {
    const Timer timer{"RecordTransferCommandBuffer"};
    TransferCommandBuffer->reset({});
    TransferCommandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    if (auto deferred_copies = Buffers->Ctx.TakeDeferredCopies(); !deferred_copies.empty()) {
        for (const auto &[buffers, ranges] : deferred_copies) {
            auto regions = ranges | transform([](const auto &r) {
                               const auto &[start, end] = r;
                               return vk::BufferCopy{start, start, end - start};
                           }) |
                to<std::vector>();
            TransferCommandBuffer->copyBuffer(buffers.Src, buffers.Dst, regions);
        }
        // Ensure buffer writes (staging copies) are visible to shader reads.
        const vk::MemoryBarrier buffer_barrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead};
        TransferCommandBuffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eComputeShader,
            {}, buffer_barrier, {}, {}
        );
    }
    TransferCommandBuffer->end();
}
#endif

void Scene::RenderSelectionPass(vk::Semaphore signal_semaphore) {
    const Timer timer{"RenderSelectionPass"};
    const auto primary_edit_instances = R.get<const SceneInteraction>(SceneEntity).Mode == InteractionMode::Edit ?
        ComputePrimaryEditInstances(R) :
        std::unordered_map<entt::entity, entt::entity>{};
    // Object selection never uses depth testing - we want all visible pixels regardless of occlusion
    RenderSelectionPassWith(
        false,
        [&](DrawListBuilder &draw_list) {
            auto batch = draw_list.BeginBatch();
            for (auto [mesh_entity, mesh_buffers, models] : R.view<MeshBuffers, ModelsBuffer>().each()) {
                auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models);
                draw.ObjectIdSlot = models.ObjectIds.Slot;
                draw.VertexCountOrHeadImageSlot = 0;
                if (auto it = primary_edit_instances.find(mesh_entity); it != primary_edit_instances.end()) {
                    AppendDraw(draw_list, batch, mesh_buffers.FaceIndices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
                } else { // todo can we guarantee only selected instances are rendered here and thus we can drop this check?
                    AppendDraw(draw_list, batch, mesh_buffers.FaceIndices, models, draw);
                }
            }
            return SelectionDrawInfo{SPT::SelectionFragmentXRay, batch};
        },
        signal_semaphore
    );

    SelectionStale = false;
}

void Scene::RenderSelectionPassWith(bool render_depth, const std::function<SelectionDrawInfo(DrawListBuilder &)> &build_fn, vk::Semaphore signal_semaphore) {
    const Timer timer{"RenderSelectionPassWith"};
    DrawListBuilder draw_list;
    DrawBatchInfo silhouette_batch{};
    if (render_depth) {
        silhouette_batch = draw_list.BeginBatch();
        for (const auto e : R.view<Selected>()) {
            const auto mesh_entity = R.get<MeshInstance>(e).MeshEntity;
            const auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
            const auto &models = R.get<ModelsBuffer>(mesh_entity);
            if (const auto model_index = GetModelBufferIndex(R, e)) {
                auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models);
                draw.ObjectIdSlot = models.ObjectIds.Slot;
                AppendDraw(draw_list, silhouette_batch, mesh_buffers.FaceIndices, models, draw, *model_index);
            }
        }
    }
    const auto selection_draw = build_fn(draw_list);

    if (!draw_list.Draws.empty()) Buffers->SelectionDrawData.Update(as_bytes(draw_list.Draws));
    if (!draw_list.IndirectCommands.empty()) Buffers->SelectionIndirect.Update(as_bytes(draw_list.IndirectCommands));
    Buffers->EnsureIdentityIndexBuffer(draw_list.MaxIndexCount);
    if (auto descriptor_updates = Buffers->Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
        Vk.Device.updateDescriptorSets(std::move(descriptor_updates), {});
        Buffers->Ctx.ClearDeferredDescriptorUpdates();
    }

    auto cb = *ClickCommandBuffer;
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Reset selection counter.
    Buffers->SelectionCounterBuffer.Write(as_bytes(SelectionCounters{}));

    // Transition head image to general layout and clear.
    const auto &head_image = Pipelines->SelectionFragment.Resources->HeadImage;
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
        vk::ImageMemoryBarrier{{}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {}, {}, *head_image.Image, ColorSubresourceRange}
    );
    cb.clearColorImage(*head_image.Image, vk::ImageLayout::eGeneral, vk::ClearColorValue{std::array<uint32_t, 4>{InvalidSlot, 0, 0, 0}}, ColorSubresourceRange);
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {},
        vk::ImageMemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral, {}, {}, *head_image.Image, ColorSubresourceRange}
    );

    const auto extent = R.get<const ViewportExtent>(SceneEntity).Value;
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(extent.width), float(extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, extent});

    auto record_draw_batch = [&](const PipelineRenderer &renderer, SPT spt, const DrawBatchInfo &batch) {
        if (batch.DrawCount == 0) return;
        const auto &pipeline = renderer.Bind(cb, spt);
        const DrawPassPushConstants pc{
            Buffers->SelectionDrawData.Slot,
            batch.DrawDataOffset,
            SelectionHandles->HeadImage,
            Buffers->SelectionNodeBuffer.Slot,
            SelectionHandles->SelectionCounter
        };
        cb.pushConstants(*pipeline.PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        cb.drawIndexedIndirect(*Buffers->SelectionIndirect, batch.IndirectOffset, batch.DrawCount, sizeof(vk::DrawIndexedIndirectCommand));
    };

    if (render_depth) {
        // Render selected meshes to silhouette depth buffer for element occlusion
        const auto &silhouette = Pipelines->Silhouette;
        static const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {Transparent}};
        const vk::Rect2D rect{{0, 0}, ToExtent2D(silhouette.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*silhouette.Renderer.RenderPass, *silhouette.Resources->Framebuffer, rect, clear_values}, vk::SubpassContents::eInline);
        cb.bindIndexBuffer(*Buffers->IdentityIndexBuffer, 0, vk::IndexType::eUint32);
        record_draw_batch(silhouette.Renderer, SPT::SilhouetteDepthObject, silhouette_batch);
        cb.endRenderPass();
    }

    const auto &selection = Pipelines->SelectionFragment;
    const vk::Rect2D rect{{0, 0}, ToExtent2D(Pipelines->Silhouette.Resources->DepthImage.Extent)};
    cb.beginRenderPass({*selection.Renderer.RenderPass, *selection.Resources->Framebuffer, rect, {}}, vk::SubpassContents::eInline);
    cb.bindIndexBuffer(*Buffers->IdentityIndexBuffer, 0, vk::IndexType::eUint32);
    record_draw_batch(selection.Renderer, selection_draw.Pipeline, selection_draw.Batch);
    cb.endRenderPass();

    cb.end();
    vk::SubmitInfo submit;
    submit.setCommandBuffers(cb);
    if (signal_semaphore) submit.setSignalSemaphores(signal_semaphore);
    Vk.Queue.submit(submit, *OneShotFence);
    WaitFor(*OneShotFence, Vk.Device);
}

void Scene::RenderEditSelectionPass(std::span<const ElementRange> ranges, Element element, vk::Semaphore signal_semaphore) {
    if (ranges.empty() || element == Element::None) return;

    const auto primary_edit_instances = ComputePrimaryEditInstances(R);
    const Timer timer{"RenderEditSelectionPass"};
    const bool xray_selection = SelectionXRay || R.get<SceneSettings>(SceneEntity).ViewportShading == ViewportShadingMode::Wireframe;
    const auto selection_pipeline = [xray_selection](Element el) -> SPT {
        if (el == Element::Vertex) return xray_selection ? SPT::SelectionElementVertexXRay : SPT::SelectionElementVertex;
        if (el == Element::Edge) return xray_selection ? SPT::SelectionElementEdgeXRay : SPT::SelectionElementEdge;
        return xray_selection ? SPT::SelectionElementFaceXRay : SPT::SelectionElementFace;
    };
    RenderSelectionPassWith(!xray_selection, [&](DrawListBuilder &draw_list) {
        auto batch = draw_list.BeginBatch();
        for (const auto &r : ranges) {
            const auto &mesh_buffers = R.get<MeshBuffers>(r.MeshEntity);
            const auto &models = R.get<ModelsBuffer>(r.MeshEntity);
            const auto &mesh = R.get<Mesh>(r.MeshEntity);
            const auto &indices = element == Element::Vertex ? mesh_buffers.VertexIndices :
                element == Element::Edge                     ? mesh_buffers.EdgeIndices :
                                                               mesh_buffers.FaceIndices;
            auto draw = MakeDrawData(mesh_buffers.Vertices, indices, models);
            draw.ObjectIdSlot = element == Element::Face ? Meshes.GetFaceIdSlot() : InvalidSlot;
            draw.FaceIdOffset = element == Element::Face ? Meshes.GetFaceIdRange(mesh.GetStoreId()).Offset : 0;
            draw.VertexCountOrHeadImageSlot = 0;
            draw.ElementIdOffset = r.Offset;
            if (auto it = primary_edit_instances.find(r.MeshEntity); it != primary_edit_instances.end()) {
                AppendDraw(draw_list, batch, indices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
            } else { // todo can we guarantee only selected instances are rendered here and thus we can drop this check?
                AppendDraw(draw_list, batch, indices, models, draw);
            }
        }
        return SelectionDrawInfo{selection_pipeline(element), batch}; }, signal_semaphore);

    // Edit selection pass overwrites the shared head image used for object selection.
    SelectionStale = true;
}

std::vector<std::vector<uint32_t>> Scene::RunBoxSelectElements(std::span<const ElementRange> ranges, Element element, std::pair<glm::uvec2, glm::uvec2> box_px) {
    if (ranges.empty()) return {};

    std::vector<std::vector<uint32_t>> results(ranges.size());
    const auto [box_min, box_max] = box_px;
    if (box_min.x >= box_max.x || box_min.y >= box_max.y) return results;

    const Timer timer{"RunBoxSelectElements"};
    const auto element_count = fold_left(
        ranges, uint32_t{0},
        [](uint32_t total, const auto &range) { return std::max(total, range.Offset + range.Count); }
    );
    if (element_count == 0) return results;

    const uint32_t bitset_words = (element_count + 31) / 32;
    if (bitset_words > BoxSelectZeroBits.size()) return results;

    RenderEditSelectionPass(ranges, element, *SelectionReadySemaphore);

    const std::span<const uint32_t> zero_bits{BoxSelectZeroBits.data(), bitset_words};
    Buffers->BoxSelectBitsetBuffer.Write(std::as_bytes(zero_bits));

    auto cb = *ClickCommandBuffer;
    const uint32_t group_count_x = (box_max.x - box_min.x + 15) / 16;
    const uint32_t group_count_y = (box_max.y - box_min.y + 15) / 16;
    RunSelectionCompute(
        cb, Vk.Queue, *OneShotFence, Vk.Device, Pipelines->BoxSelect,
        BoxSelectPushConstants{
            .BoxMin = box_min,
            .BoxMax = box_max,
            .ObjectCount = element_count,
            .HeadImageIndex = SelectionHandles->HeadImage,
            .SelectionNodesIndex = Buffers->SelectionNodeBuffer.Slot,
            .BoxResultIndex = SelectionHandles->BoxResult,
        },
        [group_count_x, group_count_y](vk::CommandBuffer dispatch_cb) {
            dispatch_cb.dispatch(group_count_x, group_count_y, 1);
        },
        *SelectionReadySemaphore
    );

    const auto *bits = reinterpret_cast<const uint32_t *>(Buffers->BoxSelectBitsetBuffer.GetData().data());
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto &range = ranges[i];
        results[i] = iota(range.Offset, range.Offset + range.Count) //
            | filter([&](uint32_t idx) {
                         const uint32_t mask = 1u << (idx % 32);
                         return (bits[idx / 32] & mask) != 0;
                     }) //
            | transform([offset = range.Offset](uint32_t idx) { return idx - offset; }) //
            | to<std::vector>();
    }
    return results;
}

std::optional<uint32_t> Scene::RunClickSelectElement(entt::entity mesh_entity, Element element, glm::uvec2 mouse_px) {
    const auto &mesh = R.get<Mesh>(mesh_entity);
    const uint32_t element_count = GetElementCount(mesh, element);
    if (element_count == 0 || element == Element::None) return {};

    const Timer timer{"RunClickSelectElement"};
    const ElementRange range{mesh_entity, 0, element_count};
    RenderEditSelectionPass(std::span{&range, 1}, element, *SelectionReadySemaphore);
    if (const auto index = FindNearestSelectionElement(
            *Buffers, Pipelines->ClickSelectElement, *ClickCommandBuffer,
            Vk.Queue, *OneShotFence, Vk.Device,
            SelectionHandles->HeadImage, Buffers->SelectionNodeBuffer.Slot, SelectionHandles->ClickElementResult,
            mouse_px, element_count, element,
            *SelectionReadySemaphore
        )) {
        return *index;
    }
    return {};
}

std::optional<uint32_t> Scene::RunClickSelectExcitableVertex(entt::entity instance_entity, glm::uvec2 mouse_px) {
    if (!R.all_of<Excitable>(instance_entity)) return {};

    const Timer timer{"RunClickSelectExcitableVertex"};
    const auto mesh_entity = R.get<MeshInstance>(instance_entity).MeshEntity;
    const auto &mesh = R.get<Mesh>(mesh_entity);
    const uint32_t vertex_count = mesh.VertexCount();
    if (vertex_count == 0) return {};

    const auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
    const auto &models = R.get<ModelsBuffer>(mesh_entity);
    const auto &state_buffers = R.get<MeshElementStateBuffers>(mesh_entity);
    const auto model_index = R.get<RenderInstance>(instance_entity).BufferIndex;
    RenderSelectionPassWith(true, [&](DrawListBuilder &draw_list) {
        auto batch = draw_list.BeginBatch();
        auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.VertexIndices, models);
        draw.VertexCountOrHeadImageSlot = 0;
        draw.ElementStateSlot = state_buffers.Vertices.Slot;
        draw.ElementStateOffset = state_buffers.Vertices.Range.Offset;
        AppendDraw(draw_list, batch, mesh_buffers.VertexIndices, models, draw, model_index);
        return SelectionDrawInfo{SPT::SelectionElementVertex, batch}; }, *SelectionReadySemaphore);
    SelectionStale = true;

    return FindNearestSelectionElement(
        *Buffers, Pipelines->ClickSelectElement, *ClickCommandBuffer,
        Vk.Queue, *OneShotFence, Vk.Device,
        SelectionHandles->HeadImage, Buffers->SelectionNodeBuffer.Slot, SelectionHandles->ClickElementResult,
        mouse_px, vertex_count, Element::Vertex,
        *SelectionReadySemaphore
    );
}

// Returns entities hit at mouse_px, sorted by depth (near-to-far), with duplicates removed.
std::vector<entt::entity> Scene::RunClickSelect(glm::uvec2 mouse_px) {
    const bool selection_rendered = SelectionStale;
    if (selection_rendered) RenderSelectionPass(*SelectionReadySemaphore);

    const Timer timer{"RunClickSelect"};
    Buffers->ClickResultBuffer.Write(as_bytes(ClickResult{}));
    auto cb = *ClickCommandBuffer;
    const auto &compute = Pipelines->ClickSelect;
    RunSelectionCompute(
        cb, Vk.Queue, *OneShotFence, Vk.Device, compute,
        ClickSelectPushConstants{
            .TargetPx = mouse_px,
            .HeadImageIndex = SelectionHandles->HeadImage,
            .SelectionNodesIndex = Buffers->SelectionNodeBuffer.Slot,
            .ClickResultIndex = SelectionHandles->ClickResult,
        },
        [](vk::CommandBuffer dispatch_cb) { dispatch_cb.dispatch(1, 1, 1); },
        selection_rendered ? *SelectionReadySemaphore : vk::Semaphore{}
    );

    // Convert click hits to entities.
    const auto &result = Buffers->GetClickResult();
    std::unordered_map<uint32_t, entt::entity> object_id_to_entity;
    for (const auto [e, ri] : R.view<RenderInstance>().each()) object_id_to_entity[ri.ObjectId] = e;
    auto hits = result.Hits //
        | take(std::min<uint32_t>(result.Count, result.Hits.size())) //
        | filter([&](const auto &hit) { return object_id_to_entity.contains(hit.ObjectId); }) //
        | transform([](const auto &hit) { return std::pair{hit.Depth, hit.ObjectId}; }) //
        | to<std::vector>();
    std::ranges::sort(hits);

    std::vector<entt::entity> entities;
    entities.reserve(hits.size());
    std::unordered_set<uint32_t> seen_object_ids;
    for (const auto &[_, object_id] : hits) {
        if (seen_object_ids.insert(object_id).second) {
            entities.emplace_back(object_id_to_entity.at(object_id));
        }
    }
    return entities;
}

std::vector<entt::entity> Scene::RunBoxSelect(std::pair<glm::uvec2, glm::uvec2> box_px) {
    const auto [box_min, box_max] = box_px;
    if (box_min.x >= box_max.x || box_min.y >= box_max.y) return {};
    if (NextObjectId <= 1) return {}; // No objects have been assigned IDs yet

    // ObjectCount is the max ObjectId that could appear (for shader bounds check)
    const uint32_t max_object_id = std::min(NextObjectId - 1, SceneBuffers::MaxSelectableObjects);
    const uint32_t bitset_words = (max_object_id + 31) / 32;

    const Timer timer{"RunBoxSelect"};
    const bool selection_rendered = SelectionStale;
    if (selection_rendered) RenderSelectionPass(*SelectionReadySemaphore);

    const std::span<const uint32_t> zero_bits{BoxSelectZeroBits.data(), bitset_words};
    Buffers->BoxSelectBitsetBuffer.Write(std::as_bytes(zero_bits));

    auto cb = *ClickCommandBuffer;
    const auto &compute = Pipelines->BoxSelect;
    const uint32_t group_count_x = (box_max.x - box_min.x + 15) / 16;
    const uint32_t group_count_y = (box_max.y - box_min.y + 15) / 16;
    RunSelectionCompute(
        cb, Vk.Queue, *OneShotFence, Vk.Device, compute,
        BoxSelectPushConstants{
            .BoxMin = box_min,
            .BoxMax = box_max,
            .ObjectCount = max_object_id,
            .HeadImageIndex = SelectionHandles->HeadImage,
            .SelectionNodesIndex = Buffers->SelectionNodeBuffer.Slot,
            .BoxResultIndex = SelectionHandles->BoxResult,
        },
        [group_count_x, group_count_y](auto dispatch_cb) { dispatch_cb.dispatch(group_count_x, group_count_y, 1); },
        selection_rendered ? *SelectionReadySemaphore : vk::Semaphore{}
    );

    // Build ObjectId -> entity map for lookup
    std::unordered_map<uint32_t, entt::entity> object_id_to_entity;
    for (const auto [e, ri] : R.view<RenderInstance>().each()) {
        object_id_to_entity[ri.ObjectId] = e;
    }

    const auto *bits = reinterpret_cast<const uint32_t *>(Buffers->BoxSelectBitsetBuffer.GetData().data());
    std::vector<entt::entity> entities;
    for (uint32_t object_id = 1; object_id <= max_object_id; ++object_id) {
        const uint32_t bit_index = object_id - 1;
        const uint32_t mask = 1u << (bit_index % 32);
        if ((bits[bit_index / 32] & mask) != 0) {
            if (auto it = object_id_to_entity.find(object_id); it != object_id_to_entity.end()) {
                entities.emplace_back(it->second);
            }
        }
    }
    return entities;
}

void Scene::Interact() {
    const auto extent = R.get<const ViewportExtent>(SceneEntity).Value;
    if (extent.width == 0 || extent.height == 0) return;

    const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
    const auto active_entity = FindActiveEntity(R);
    // Handle keyboard input.
    if (IsWindowFocused()) {
        if (IsKeyPressed(ImGuiKey_Tab)) {
            // Cycle to the next interaction mode, wrapping around to the first.
            auto it = find(InteractionModes, interaction_mode);
            SetInteractionMode(++it != InteractionModes.end() ? *it : *InteractionModes.begin());
        }
        if (interaction_mode == InteractionMode::Edit) {
            if (IsKeyPressed(ImGuiKey_1, false)) SetEditMode(Element::Vertex);
            else if (IsKeyPressed(ImGuiKey_2, false)) SetEditMode(Element::Edge);
            else if (IsKeyPressed(ImGuiKey_3, false)) SetEditMode(Element::Face);
        }
        if (!R.storage<Selected>().empty()) {
            if (IsKeyPressed(ImGuiKey_D, false) && GetIO().KeyShift) Duplicate();
            else if (IsKeyPressed(ImGuiKey_D, false) && GetIO().KeyAlt) DuplicateLinked();
            else if (IsKeyPressed(ImGuiKey_Delete, false) || IsKeyPressed(ImGuiKey_Backspace, false)) Delete();
            else if (IsKeyPressed(ImGuiKey_G, false)) {
                // In Edit mode with vertex selection, start vertex grab; otherwise start object transform
                const auto edit_mode_value = R.get<const SceneEditMode>(SceneEntity).Value;
                if (interaction_mode == InteractionMode::Edit && edit_mode_value == Element::Vertex) {
                    // Find selected instance with selected vertices
                    for (const auto [instance_entity, mi] : R.view<const MeshInstance, const Selected>().each()) {
                        const auto &selection = R.get<const MeshSelection>(mi.MeshEntity);
                        if (!selection.Handles.empty()) {
                            const auto &mesh = R.get<const Mesh>(mi.MeshEntity);
                            const auto &camera = R.get<const Camera>(SceneEntity);
                            const auto window_pos = ToGlm(GetWindowPos());
                            const auto window_size = ToGlm(GetContentRegionAvail());
                            StartVertexGrabMouseRay = camera.PixelToWorldRay(ToGlm(GetIO().MousePos), window_pos, window_size);

                            auto &grab = R.emplace<VertexGrabState>(instance_entity);
                            for (const auto vi : selection.Handles) {
                                grab.StartPositions.emplace_back(vi, mesh.GetPosition(VH{vi}));
                            }
                            break;
                        }
                    }
                } else {
                    StartScreenTransform = TransformGizmo::TransformType::Translate;
                }
            } else if (IsKeyPressed(ImGuiKey_R, false)) StartScreenTransform = TransformGizmo::TransformType::Rotate;
            else if (IsKeyPressed(ImGuiKey_S, false) && !R.all_of<Frozen>(active_entity)) StartScreenTransform = TransformGizmo::TransformType::Scale;
            else if (IsKeyPressed(ImGuiKey_H, false)) {
                for (const auto e : R.view<Selected>()) SetVisible(e, !R.all_of<RenderInstance>(e));
            } else if (IsKeyPressed(ImGuiKey_P, false) && GetIO().KeyCtrl) {
                if (active_entity != entt::null) {
                    for (const auto e : R.view<Selected>()) {
                        if (e != active_entity) SetParent(R, e, active_entity);
                    }
                }
            } else if (IsKeyPressed(ImGuiKey_P, false) && GetIO().KeyAlt) {
                for (const auto e : R.view<Selected>()) ClearParent(R, e);
            }
        }
    }

    // Handle vertex grab interaction (G key in Edit mode with vertex selection)
    if (StartVertexGrabMouseRay) {
        // Cancel on Escape or right click - restore original positions
        if (IsKeyPressed(ImGuiKey_Escape, false) || IsMouseClicked(ImGuiMouseButton_Right)) {
            for (const auto &[instance_entity, grab] : R.view<VertexGrabState>().each()) {
                const auto mesh_entity = R.get<const MeshInstance>(instance_entity).MeshEntity;
                const auto &mesh = R.get<const Mesh>(mesh_entity);
                for (const auto &[vi, pos] : grab.StartPositions) Meshes.SetPosition(mesh, vi, pos);
                Meshes.UpdateNormals(mesh);
                R.emplace_or_replace<MeshGeometryDirty>(mesh_entity);
            }
            R.clear<VertexGrabState>();
            StartVertexGrabMouseRay.reset();
        } else {
            // Update positions
            const auto &camera = R.get<const Camera>(SceneEntity);
            const auto n = -camera.Forward(); // Plane normal (faces camera)
            const auto window_pos = ToGlm(GetWindowPos());
            const auto window_size = ToGlm(GetContentRegionAvail());
            const auto mouse_ray = camera.PixelToWorldRay(ToGlm(GetIO().MousePos) + AccumulatedWrapMouseDelta, window_pos, window_size);
            const auto start_ray = *StartVertexGrabMouseRay;
            const auto start_scaled = start_ray.d / glm::dot(n, start_ray.d);
            const auto current_scaled = mouse_ray.d / glm::dot(n, mouse_ray.d);
            const auto delta_base = (mouse_ray.o - current_scaled * glm::dot(n, mouse_ray.o)) - (start_ray.o - start_scaled * glm::dot(n, start_ray.o));
            const auto delta_scale = current_scaled - start_scaled;
            for (const auto &[instance_entity, grab] : R.view<VertexGrabState>().each()) {
                const auto mesh_entity = R.get<const MeshInstance>(instance_entity).MeshEntity;
                const auto &mesh = R.get<const Mesh>(mesh_entity);
                const auto &wm = R.get<const WorldMatrix>(instance_entity);
                const auto world_to_local = glm::transpose(mat3{wm.MInv});
                for (const auto &[vi, start_local] : grab.StartPositions) {
                    const auto depth = glm::dot(n, vec3{wm.M * vec4{start_local, 1.f}});
                    Meshes.SetPosition(mesh, vi, start_local + world_to_local * (delta_base + delta_scale * depth));
                }
                Meshes.UpdateNormals(mesh);
                R.emplace_or_replace<MeshGeometryDirty>(mesh_entity);
            }

            // Confirm on left click or enter
            if (IsMouseClicked(ImGuiMouseButton_Left) || IsKeyPressed(ImGuiKey_Enter, false)) {
                R.clear<VertexGrabState>();
                StartVertexGrabMouseRay.reset();
            } else {
                SetMouseCursor(ImGuiMouseCursor_ResizeAll);
                WrapMousePos(GetCurrentWindowRead()->InnerClipRect, AccumulatedWrapMouseDelta);
                return;
            }
        }
    }

    // Handle mouse input.
    if (!IsMouseDown(ImGuiMouseButton_Left)) R.clear<ExcitedVertex>();

    if (TransformGizmo::IsUsing()) {
        // TransformGizmo overrides this mouse cursor during some actions - this is a default.
        SetMouseCursor(ImGuiMouseCursor_ResizeAll);
        WrapMousePos(GetCurrentWindowRead()->InnerClipRect, AccumulatedWrapMouseDelta);
    } else {
        AccumulatedWrapMouseDelta = {0, 0};
    }
    if (!IsWindowHovered()) return;

    // Mouse wheel for camera rotation, Cmd+wheel to zoom.
    const auto &io = GetIO();
    if (const vec2 wheel{io.MouseWheelH, io.MouseWheel}; wheel != vec2{0, 0}) {
        if (io.KeyCtrl || io.KeySuper) {
            R.patch<Camera>(SceneEntity, [&](auto &camera) { camera.SetTargetDistance(std::max(camera.Distance * (1 - wheel.y / 16.f), 0.01f)); });
        } else {
            R.patch<Camera>(SceneEntity, [&](auto &camera) { camera.SetTargetYawPitch(camera.YawPitch + wheel * 0.15f); });
        }
    }
    if (TransformGizmo::IsUsing() || OrientationGizmo::IsActive() || TransformModePillsHovered) return;

    const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
    if (SelectionMode == SelectionMode::Box && (interaction_mode == InteractionMode::Edit || interaction_mode == InteractionMode::Object)) {
        const auto mouse_pos = ToGlm(GetMousePos());
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            BoxSelectStart = mouse_pos;
            BoxSelectEnd = mouse_pos;
        } else if (IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart) {
            BoxSelectEnd = mouse_pos;
            if (const auto box_px = ComputeBoxSelectPixels(*BoxSelectStart, *BoxSelectEnd, ToGlm(GetCursorScreenPos()), extent); box_px) {
                const bool is_additive = IsKeyDown(ImGuiMod_Shift);
                if (interaction_mode == InteractionMode::Edit) {
                    Timer timer{"BoxSelectElements (all)"};

                    std::unordered_set<entt::entity> selected_mesh_entities;
                    for (const auto [_, mi] : R.view<const MeshInstance, const Selected>().each()) {
                        selected_mesh_entities.insert(mi.MeshEntity);
                    }

                    std::vector<ElementRange> ranges;
                    uint32_t offset = 0;
                    for (const auto mesh_entity : selected_mesh_entities) {
                        if (!is_additive && !R.get<MeshSelection>(mesh_entity).Handles.empty()) {
                            R.patch<MeshSelection>(mesh_entity, [](auto &s) { s.Handles.clear(); });
                        }
                        if (const uint32_t count = GetElementCount(R.get<Mesh>(mesh_entity), edit_mode); count > 0) {
                            ranges.emplace_back(mesh_entity, offset, count);
                            offset += count;
                        }
                    }

                    auto results = RunBoxSelectElements(ranges, edit_mode, *box_px);
                    for (size_t i = 0; i < results.size(); ++i) {
                        const auto e = ranges[i].MeshEntity;
                        R.patch<MeshSelection>(e, [&](auto &s) {
                            if (is_additive) s.Handles.insert(results[i].begin(), results[i].end());
                            else s.Handles = {results[i].begin(), results[i].end()};
                        });
                    }
                } else if (interaction_mode == InteractionMode::Object) {
                    const auto selected_entities = RunBoxSelect(*box_px);
                    if (!is_additive) R.clear<Selected>();
                    for (const auto e : selected_entities) R.emplace_or_replace<Selected>(e);
                }
            }
        } else if (!IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart) {
            BoxSelectStart.reset();
            BoxSelectEnd.reset();
        }
        if (BoxSelectStart) return;
    }

    const auto mouse_pos_rel = GetMousePos() - GetCursorScreenPos();
    // Flip y-coordinate: ImGui uses top-left origin, but Vulkan gl_FragCoord uses bottom-left origin
    const glm::uvec2 mouse_px{uint32_t(mouse_pos_rel.x), uint32_t(extent.height - mouse_pos_rel.y)};

    if (interaction_mode == InteractionMode::Excite) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            if (const auto hit_entities = RunClickSelect(mouse_px); !hit_entities.empty()) {
                if (const auto hit_entity = hit_entities.front(); R.all_of<Excitable>(hit_entity)) {
                    if (const auto vertex = RunClickSelectExcitableVertex(hit_entity, mouse_px)) {
                        R.emplace_or_replace<ExcitedVertex>(hit_entity, *vertex, 1.f);
                    }
                }
            }
        } else if (!IsMouseDown(ImGuiMouseButton_Left)) {
            R.clear<ExcitedVertex>();
        }
        return;
    }
    if (!IsSingleClicked(ImGuiMouseButton_Left)) return;
    if (interaction_mode == InteractionMode::Edit && edit_mode == Element::None) return;

    const auto hit_entities = RunClickSelect(mouse_px);
    if (interaction_mode == InteractionMode::Edit) {
        const auto hit_it = find_if(hit_entities, [&](auto e) { return R.all_of<Selected>(e); });
        const bool toggle = IsKeyDown(ImGuiMod_Shift) || IsKeyDown(ImGuiMod_Ctrl) || IsKeyDown(ImGuiMod_Super);
        if (!toggle) {
            std::unordered_set<entt::entity> selected_mesh_entities;
            for (const auto [_, mi] : R.view<const MeshInstance, const Selected>().each()) {
                selected_mesh_entities.insert(mi.MeshEntity);
            }
            for (const auto mesh_entity : selected_mesh_entities) {
                if (!R.get<MeshSelection>(mesh_entity).Handles.empty()) {
                    R.patch<MeshSelection>(mesh_entity, [](auto &s) { s.Handles.clear(); });
                }
            }
        }
        if (hit_it != hit_entities.end()) {
            const auto mesh_entity = R.get<MeshInstance>(*hit_it).MeshEntity;
            if (const auto element_index = RunClickSelectElement(mesh_entity, edit_mode, mouse_px)) {
                R.patch<MeshSelection>(mesh_entity, [&](auto &selection) {
                    if (!toggle) selection = {};
                    if (toggle && selection.Handles.contains(*element_index)) {
                        selection.Handles.erase(*element_index);
                        if (selection.ActiveHandle == *element_index) selection.ActiveHandle = {};
                    } else {
                        selection.Handles.emplace(*element_index);
                        selection.ActiveHandle = *element_index;
                    }
                });
            }
        }
    } else if (interaction_mode == InteractionMode::Object) {
        // Cycle through hit entities.
        entt::entity hit = entt::null;
        if (!hit_entities.empty()) {
            auto it = find(hit_entities, active_entity);
            if (it != hit_entities.end()) ++it;
            if (it == hit_entities.end()) it = hit_entities.begin();
            hit = *it;
        }
        if (hit != entt::null && IsKeyDown(ImGuiMod_Shift)) {
            if (active_entity == hit) {
                ToggleSelected(hit);
            } else {
                R.clear<Active>();
                R.emplace<Active>(hit);
                R.emplace_or_replace<Selected>(hit);
            }
        } else if (hit != entt::null || !IsKeyDown(ImGuiMod_Shift)) {
            Select(hit);
        }
    }
}

void ScenePipelines::SetExtent(vk::Extent2D extent) {
    Main.SetExtent(extent, Device, PhysicalDevice, Samples);
    Silhouette.SetExtent(extent, Device, PhysicalDevice);
    SilhouetteEdge.SetExtent(extent, Device, PhysicalDevice);
    SelectionFragment.SetExtent(extent, Device, PhysicalDevice, *Silhouette.Resources->DepthImage.View);
};

bool Scene::SubmitViewport(vk::Fence viewportConsumerFence) {
    auto &extent = R.get<ViewportExtent>(SceneEntity).Value;
    const auto content_region = ToGlm(GetContentRegionAvail());
    const bool extent_changed = extent.width != content_region.x || extent.height != content_region.y;
    if (extent_changed) {
        // uint(e.x), uint(e.y)
        extent.width = uint(content_region.x);
        extent.height = uint(content_region.y);
        R.patch<ViewportExtent>(SceneEntity, [](auto &) {});
    }

    const auto render_request = ProcessComponentEvents();

    if (auto descriptor_updates = Buffers->Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
        const Timer timer{"SubmitViewport->UpdateBufferDescriptorSets"};
        Vk.Device.updateDescriptorSets(std::move(descriptor_updates), {});
        Buffers->Ctx.ClearDeferredDescriptorUpdates();
    }
    if (!extent_changed && render_request == RenderRequest::None) return false;

    const Timer timer{"SubmitViewport"};
    if (extent_changed) {
        if (viewportConsumerFence) { // Wait for viewport consumer to finish sampling old resources
            std::ignore = Vk.Device.waitForFences(viewportConsumerFence, VK_TRUE, UINT64_MAX);
        }
        Pipelines->SetExtent(extent);
        Buffers->ResizeSelectionNodeBuffer(extent);
        {
            const Timer timer{"SubmitViewport->UpdateSelectionDescriptorSets"};
            const auto head_image_info = vk::DescriptorImageInfo{
                nullptr,
                *Pipelines->SelectionFragment.Resources->HeadImage.View,
                vk::ImageLayout::eGeneral
            };
            const vk::DescriptorBufferInfo selection_counter{*Buffers->SelectionCounterBuffer, 0, sizeof(SelectionCounters)};
            const vk::DescriptorBufferInfo click_result{*Buffers->ClickResultBuffer, 0, sizeof(ClickResult)};
            const vk::DescriptorBufferInfo click_element_result{*Buffers->ClickElementResultBuffer, 0, SceneBuffers::ClickSelectElementGroupCount * sizeof(ClickElementCandidate)};
            const auto &sil = Pipelines->Silhouette;
            const auto &sil_edge = Pipelines->SilhouetteEdge;
            const vk::DescriptorImageInfo silhouette_sampler{*sil.Resources->ImageSampler, *sil.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo object_id_sampler{*sil_edge.Resources->ImageSampler, *sil_edge.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo depth_sampler{*sil_edge.Resources->DepthSampler, *sil_edge.Resources->DepthImage.View, vk::ImageLayout::eDepthStencilReadOnlyOptimal};
            const auto box_result = Buffers->GetBoxSelectBitsetDescriptor();
            Vk.Device.updateDescriptorSets(
                {
                    Slots->MakeImageWrite(SelectionHandles->HeadImage, head_image_info),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->SelectionCounter}, selection_counter),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->ClickResult}, click_result),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->ClickElementResult}, click_element_result),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->BoxResult}, box_result),
                    Slots->MakeSamplerWrite(SelectionHandles->ObjectIdSampler, object_id_sampler),
                    Slots->MakeSamplerWrite(SelectionHandles->DepthSampler, depth_sampler),
                    Slots->MakeSamplerWrite(SelectionHandles->SilhouetteSampler, silhouette_sampler),
                },
                {}
            );
        }
        if (auto descriptor_updates = Buffers->Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
            Vk.Device.updateDescriptorSets(std::move(descriptor_updates), {});
            Buffers->Ctx.ClearDeferredDescriptorUpdates();
        }
    }

#ifdef MVK_FORCE_STAGED_TRANSFERS
    RecordTransferCommandBuffer();
#endif

    if (render_request == RenderRequest::ReRecord || extent_changed) RecordRenderCommandBuffer();

    vk::SubmitInfo submit;
#ifdef MVK_FORCE_STAGED_TRANSFERS
    const std::array command_buffers{*TransferCommandBuffer, *RenderCommandBuffer};
    submit.setCommandBuffers(command_buffers);
#else
    submit.setCommandBuffers(*RenderCommandBuffer);
#endif
    Vk.Queue.submit(submit, *RenderFence);
    RenderPending = true;
    return extent_changed;
}

void Scene::WaitForRender() {
    if (!RenderPending) return;

    const Timer timer{"WaitForRender"};
    WaitFor(*RenderFence, Vk.Device);
    Buffers->Ctx.ReclaimRetiredBuffers();
    RenderPending = false;
}

void Scene::RenderOverlay() {
    const auto window_pos = ToGlm(GetWindowPos());
    const auto window_size = ToGlm(GetContentRegionAvail());
    auto &camera = R.get<Camera>(SceneEntity);
    { // Transform mode pill buttons (top-left overlay)
        struct ButtonInfo {
            const SvgResource &Icon;
            TransformGizmo::Type ButtonType;
            ImDrawFlags Corners;
            bool Enabled;
        };

        using enum TransformGizmo::Type;
        const auto v = R.view<Selected, Frozen>();
        const bool scale_enabled = v.begin() == v.end();
        const ButtonInfo buttons[]{
            {*Icons.Select, None, ImDrawFlags_RoundCornersTop, true},
            {*Icons.SelectBox, None, ImDrawFlags_RoundCornersBottom, true},
            {*Icons.Move, Translate, ImDrawFlags_RoundCornersTop, true},
            {*Icons.Rotate, Rotate, ImDrawFlags_RoundCornersNone, true},
            {*Icons.Scale, Scale, ImDrawFlags_RoundCornersNone, scale_enabled},
            {*Icons.Universal, Universal, ImDrawFlags_RoundCornersBottom, true},
        };

        auto &element = MGizmo.Config.Type;
        if (!scale_enabled && element == Scale) element = Translate;

        const float padding = GetTextLineHeightWithSpacing() / 2.f;
        const auto start_pos = std::bit_cast<ImVec2>(window_pos) + GetWindowContentRegionMin() + ImVec2{padding, padding};
        const auto saved_cursor_pos = GetCursorScreenPos();

        auto &dl = *GetWindowDrawList();
        TransformModePillsHovered = false;
        static constexpr ImVec2 button_size{36, 30};
        static constexpr float gap{4}; // Gap between select buttons and transform buttons
        for (uint i = 0; i < 6; ++i) {
            const auto &[icon, button_type, corners, enabled] = buttons[i];
            static constexpr ImVec2 padding{0.5f, 0.5f};
            static constexpr float icon_dim{button_size.y * 0.75f};
            static constexpr ImVec2 icon_size{icon_dim, icon_dim};
            const float y_offset = i < 2 ? i * button_size.y : 2 * button_size.y + gap + (i - 2) * button_size.y;
            SetCursorScreenPos({start_pos.x, start_pos.y + y_offset});

            if (!enabled) BeginDisabled();
            PushID(i);
            const bool clicked = InvisibleButton("##icon", button_size);
            PopID();
            if (!enabled) EndDisabled();

            const bool hovered = IsItemHovered();
            if (hovered) TransformModePillsHovered = true;

            if (clicked) {
                if (i == 0) {
                    SelectionMode = SelectionMode::Click;
                    element = None;
                } else if (i == 1) {
                    SelectionMode = SelectionMode::Box;
                    element = None;
                } else { // Transform buttons
                    element = button_type;
                }
            }

            const bool is_active = i < 2 ?
                (element == None && ((i == 0 && SelectionMode == SelectionMode::Click) || (i == 1 && SelectionMode == SelectionMode::Box))) :
                element == button_type;
            const auto bg_color = GetColorU32(
                !enabled      ? ImGuiCol_FrameBg :
                    is_active ? ImGuiCol_ButtonActive :
                    hovered   ? ImGuiCol_ButtonHovered :
                                ImGuiCol_Button
            );
            dl.AddRectFilled(GetItemRectMin() + padding, GetItemRectMax() - padding, bg_color, 8.f, corners);
            SetCursorScreenPos(GetItemRectMin() + (button_size - icon_size) * 0.5f);
            icon.DrawIcon(std::bit_cast<vec2>(icon_size));
        }
        SetCursorScreenPos(saved_cursor_pos);
    }

    if (!R.storage<Selected>().empty()) { // Draw center-dot for active/selected entities
        const auto &theme = R.get<const ViewportTheme>(SceneEntity);
        const auto vp = camera.Projection(window_size.x / window_size.y) * camera.View();
        for (const auto [e, wm, ri] : R.view<const WorldMatrix, const RenderInstance>().each()) {
            if (!R.any_of<Active, Selected>(e)) continue;

            const auto p_cs = vp * wm.M[3]; // World to clip space (4th column is translation)
            const auto p_ndc = fabsf(p_cs.w) > FLT_EPSILON ? vec3{p_cs} / p_cs.w : vec3{p_cs}; // Clip space to NDC
            const auto p_uv = vec2{p_ndc.x + 1, 1 - p_ndc.y} * 0.5f; // NDC to UV [0,1] (top-left origin)
            const auto p_px = std::bit_cast<ImVec2>(window_pos + p_uv * window_size); // UV to px
            auto &dl = *GetWindowDrawList();
            dl.AddCircleFilled(p_px, 3.5f, colors::RgbToU32(R.all_of<Active>(e) ? theme.Colors.ObjectActive : theme.Colors.ObjectSelected), 10);
            dl.AddCircle(p_px, 3.5f, IM_COL32(0, 0, 0, 255), 10, 1.f);
        }
    }

    if (const auto selected_view = R.view<const Selected>(); !selected_view.empty()) { // Transform gizmo
        // Transform all root selected entities (whose parent is not also selected) around their average position,
        // using the active entity's rotation/scale.
        // Non-root selected entities already follow their parent's transform.
        const auto is_parent_selected = [&](entt::entity e) {
            if (const auto *node = R.try_get<SceneNode>(e)) {
                return node->Parent != entt::null && R.all_of<Selected>(node->Parent);
            }
            return false;
        };

        auto root_selected = selected_view | filter([&](auto e) { return !is_parent_selected(e); });
        const auto root_count = distance(root_selected);

        const auto active_entity = FindActiveEntity(R);
        const auto active_transform = active_entity != entt::null ? GetTransform(R, active_entity) : Transform{};
        const auto p = fold_left(root_selected | transform([&](auto e) { return R.get<Position>(e).Value; }), vec3{}, std::plus{}) / float(root_count);
        const auto start_transform_view = R.view<const StartTransform>();
        if (auto start_delta = TransformGizmo::Draw(
                {{.P = p, .R = active_transform.R, .S = active_transform.S}, MGizmo.Mode},
                MGizmo.Config, camera, window_pos, window_size, ToGlm(GetIO().MousePos) + AccumulatedWrapMouseDelta,
                StartScreenTransform
            )) {
            const auto &[ts, td] = *start_delta;
            if (start_transform_view.empty()) {
                for (const auto e : root_selected) R.emplace<StartTransform>(e, GetTransform(R, e));
            }
            // Compute delta transform from drag start
            const auto r = ts.R, rT = glm::conjugate(r);
            for (const auto &[e, ts_e_comp] : start_transform_view.each()) {
                const auto &ts_e = ts_e_comp.T;
                const bool frozen = R.all_of<Frozen>(e);
                const auto offset = ts_e.P - ts.P;
                SetTransform(
                    R, e,
                    {
                        .P = td.P + ts.P + glm::rotate(td.R, frozen ? offset : r * (rT * offset * td.S)),
                        .R = glm::normalize(td.R * ts_e.R),
                        .S = frozen ? ts_e.S : td.S * ts_e.S,
                    }
                );
            }
        } else if (!start_transform_view.empty()) {
            R.clear<StartTransform>();
        }
    }
    { // Orientation gizmo
        static constexpr float OGizmoSize{90};
        const float padding = GetTextLineHeightWithSpacing();
        const auto pos = window_pos + vec2{GetWindowContentRegionMax().x, GetWindowContentRegionMin().y} - vec2{OGizmoSize, 0} + vec2{-padding, padding};
        OrientationGizmo::Draw(pos, OGizmoSize, camera);
        if (camera.Tick()) R.patch<Camera>(SceneEntity, [](auto &) {});
    }

    if (BoxSelectStart.has_value() && BoxSelectEnd.has_value()) {
        auto &dl = *GetWindowDrawList();
        const auto box_min = glm::min(*BoxSelectStart, *BoxSelectEnd);
        const auto box_max = glm::max(*BoxSelectStart, *BoxSelectEnd);
        dl.AddRectFilled(std::bit_cast<ImVec2>(box_min), std::bit_cast<ImVec2>(box_max), IM_COL32(255, 255, 255, 30));

        // Dashed outline
        static constexpr auto outline_color{IM_COL32(255, 255, 255, 200)};
        static constexpr float dash_size{4}, gap_size{4};
        // Top
        for (float x = box_min.x; x < box_max.x; x += dash_size + gap_size) {
            dl.AddLine({x, box_min.y}, {glm::min(x + dash_size, box_max.x), box_min.y}, outline_color, 1.0f);
        }
        // Bottom
        for (float x = box_min.x; x < box_max.x; x += dash_size + gap_size) {
            dl.AddLine({x, box_max.y}, {glm::min(x + dash_size, box_max.x), box_max.y}, outline_color, 1.0f);
        }
        // Left
        for (float y = box_min.y; y < box_max.y; y += dash_size + gap_size) {
            dl.AddLine({box_min.x, y}, {box_min.x, glm::min(y + dash_size, box_max.y)}, outline_color, 1.0f);
        }
        // Right
        for (float y = box_min.y; y < box_max.y; y += dash_size + gap_size) {
            dl.AddLine({box_max.x, y}, {box_max.x, glm::min(y + dash_size, box_max.y)}, outline_color, 1.0f);
        }
    }

    StartScreenTransform = {};
}

void Scene::RenderEntityControls(entt::entity active_entity) {
    if (active_entity == entt::null) {
        TextUnformatted("Active object: None");
        return;
    }

    PushID("EntityControls");
    Text("Active entity: %s", GetName(R, active_entity).c_str());
    Indent();

    entt::entity activate_entity = entt::null, toggle_selected = entt::null;
    if (const auto *node = R.try_get<SceneNode>(active_entity)) {
        if (auto parent_entity = node->Parent; parent_entity != entt::null) {
            AlignTextToFramePadding();
            Text("Parent: %s", GetName(R, parent_entity).c_str());
            SameLine();
            if (active_entity != parent_entity && Button("Activate##Parent")) activate_entity = parent_entity;
            SameLine();
            if (Button(R.all_of<Selected>(parent_entity) ? "Deselect##Parent" : "Select##Parent")) toggle_selected = parent_entity;
        }
        if (node->FirstChild != entt::null && CollapsingHeader("Children")) {
            RenderEntitiesTable("Children", active_entity);
        }
    }

    if (const auto *mesh_instance = R.try_get<MeshInstance>(active_entity)) {
        Text("Mesh entity: %s", GetName(R, mesh_instance->MeshEntity).c_str());
    }
    {
        const auto model_buffer_index = GetModelBufferIndex(R, active_entity);
        Text("Model buffer index: %s", model_buffer_index ? std::to_string(*model_buffer_index).c_str() : "None");
    }
    const auto active_mesh_entity = R.get<MeshInstance>(active_entity).MeshEntity;
    const auto &active_mesh = R.get<const Mesh>(active_mesh_entity);
    TextUnformatted(
        std::format("Vertices | Edges | Faces: {:L} | {:L} | {:L}", active_mesh.VertexCount(), active_mesh.EdgeCount(), active_mesh.FaceCount()).c_str()
    );
    Unindent();
    if (CollapsingHeader("Transform")) {
        auto &position = R.get<Position>(active_entity).Value;
        bool model_changed = DragFloat3("Position", &position[0], 0.01f);
        if (model_changed) R.patch<Position>(active_entity, [](auto &) {});
        // Rotation editor
        {
            int mode_i = R.get<const RotationUiVariant>(active_entity).index();
            const char *modes[]{"Quat (WXYZ)", "XYZ Euler", "Axis Angle"};
            if (ImGui::Combo("Rotation mode", &mode_i, modes, IM_ARRAYSIZE(modes))) {
                R.replace<RotationUiVariant>(active_entity, CreateVariantByIndex<RotationUiVariant>(mode_i));
                SetRotation(R, active_entity, R.get<const Rotation>(active_entity).Value);
            }
        }
        auto &rotation_ui = R.get<RotationUiVariant>(active_entity);
        const bool rotation_changed = std::visit(
            overloaded{
                [&](RotationQuat &v) {
                    if (DragFloat4("Rotation (quat WXYZ)", &v.Value[0], 0.01f)) {
                        R.replace<Rotation>(active_entity, glm::normalize(v.Value));
                        return true;
                    }
                    return false;
                },
                [&](RotationEuler &v) {
                    if (DragFloat3("Rotation (XYZ Euler, deg)", &v.Value[0], 1.f)) {
                        const auto rads = glm::radians(v.Value);
                        R.replace<Rotation>(active_entity, glm::normalize(glm::quat_cast(glm::eulerAngleXYZ(rads.x, rads.y, rads.z))));
                        return true;
                    }
                    return false;
                },
                [&](RotationAxisAngle &v) {
                    bool changed = DragFloat3("Rotation axis (XYZ)", &v.Value[0], 0.01f);
                    changed |= DragFloat("Angle (deg)", &v.Value.w, 1.f);
                    if (changed) {
                        const auto axis = glm::normalize(vec3{v.Value});
                        const auto angle = glm::radians(v.Value.w);
                        R.replace<Rotation>(active_entity, glm::normalize(quat{std::cos(angle / 2), axis * std::sin(angle / 2)}));
                        return true;
                    }
                    return false;
                },
            },
            rotation_ui
        );
        if (rotation_changed) {
            R.patch<RotationUiVariant>(active_entity, [](auto &) {});
        }
        model_changed |= rotation_changed;

        const bool frozen = R.all_of<Frozen>(active_entity);
        if (frozen) BeginDisabled();
        const auto label = std::format("Scale{}", frozen ? " (frozen)" : "");
        auto &scale = R.get<Scale>(active_entity).Value;
        const bool scale_changed = DragFloat3(label.c_str(), &scale[0], 0.01f, 0.01f, 10);
        if (scale_changed) R.patch<Scale>(active_entity, [](auto &) {});
        model_changed |= scale_changed;
        if (frozen) EndDisabled();
        if (model_changed) {
            UpdateWorldMatrix(R, active_entity);
        }
        Spacing();
        {
            AlignTextToFramePadding();
            Text("Mode:");
            SameLine();
            using enum TransformGizmo::Mode;
            auto &mode = MGizmo.Mode;
            if (RadioButton("Local", mode == Local)) mode = Local;
            SameLine();
            if (RadioButton("World", mode == World)) mode = World;
            Spacing();
            Checkbox("Snap", &MGizmo.Config.Snap);
            if (MGizmo.Config.Snap) {
                SameLine();
                // todo link/unlink snap values
                DragFloat3("Snap", &MGizmo.Config.SnapValue.x, 1.f, 0.01f, 100.f);
            }
        }
        Spacing();
        if (TreeNode("Debug")) {
            if (const auto label = TransformGizmo::ToString(); label != "") {
                Text("%s op: %s", TransformGizmo::IsUsing() ? "Active" : "Hovered", label.data());
            } else {
                TextUnformatted("Not hovering");
            }
            TreePop();
        }
        if (TreeNode("World matrix")) {
            TextUnformatted("M");
            const auto &world_matrix = R.get<WorldMatrix>(active_entity);
            RenderMat4(world_matrix.M);
            Spacing();
            TextUnformatted("MInv");
            RenderMat4(world_matrix.MInv);
            TreePop();
        }
    }
    if (const auto *primitive_type = R.try_get<PrimitiveType>(active_mesh_entity)) {
        if (CollapsingHeader("Update primitive")) {
            if (auto mesh_data = PrimitiveEditor(*primitive_type, false)) {
                SetMeshPositions(active_mesh_entity, std::move(mesh_data->Positions));
            }
        }
    }
    PopID();

    if (activate_entity != entt::null) Select(activate_entity);
    else if (toggle_selected != entt::null) ToggleSelected(toggle_selected);
}

void Scene::RenderControls() {
    if (BeginTabBar("Scene controls")) {
        if (BeginTabItem("Object")) {
            {
                const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
                const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
                PushID("InteractionMode");
                AlignTextToFramePadding();
                TextUnformatted("Interaction mode:");
                auto interaction_mode_value = int(interaction_mode);
                bool interaction_mode_changed = false;
                for (const auto mode : InteractionModes) {
                    SameLine();
                    interaction_mode_changed |= RadioButton(to_string(mode).c_str(), &interaction_mode_value, int(mode));
                }
                if (interaction_mode_changed) SetInteractionMode(InteractionMode(interaction_mode_value));
                if (interaction_mode == InteractionMode::Edit || interaction_mode == InteractionMode::Excite) {
                    Checkbox("Orbit to active", &OrbitToActive);
                }
                if (interaction_mode == InteractionMode::Edit) {
                    Checkbox("X-ray selection", &SelectionXRay);
                    if (R.get<const SceneSettings>(SceneEntity).ViewportShading == ViewportShadingMode::Wireframe) {
                        SameLine();
                        TextDisabled("(wireframe)");
                    }
                }
                if (interaction_mode == InteractionMode::Edit) {
                    AlignTextToFramePadding();
                    TextUnformatted("Edit mode:");
                    auto type_interaction_mode = int(edit_mode);
                    for (const auto element : Elements) {
                        auto name = Capitalize(label(element));
                        SameLine();
                        if (RadioButton(name.c_str(), &type_interaction_mode, int(element))) {
                            SetEditMode(element);
                        }
                    }
                    const auto active_entity = FindActiveEntity(R);
                    if (active_entity != entt::null) {
                        const auto mesh_entity = R.get<MeshInstance>(active_entity).MeshEntity;
                        const auto &selection = R.get<MeshSelection>(mesh_entity);
                        Text("Editing %s: %zu selected", label(edit_mode).data(), selection.Handles.size());
                        if (edit_mode == Element::Vertex && !selection.Handles.empty()) {
                            const auto &mesh = R.get<Mesh>(mesh_entity);
                            for (const auto vh : selection.Handles) {
                                const auto pos = mesh.GetPosition(VH{vh});
                                Text("Vertex %u: (%.4f, %.4f, %.4f)", vh, pos.x, pos.y, pos.z);
                            }
                        }
                    }
                }
                PopID();
            }
            if (!R.storage<Selected>().empty()) {
                SeparatorText("Selection actions");
                const auto visible_view = R.view<Selected, RenderInstance>();
                const auto hidden_view = R.view<Selected>(entt::exclude<RenderInstance>);
                const bool any_visible = visible_view.begin() != visible_view.end();
                const bool any_hidden = hidden_view.begin() != hidden_view.end();
                const bool mixed_visible = any_visible && any_hidden;
                if (mixed_visible) ImGui::PushItemFlag(ImGuiItemFlags_MixedValue, true);
                if (bool set_visible = any_visible && !any_hidden; Checkbox("Visible", &set_visible)) {
                    for (const auto e : R.view<Selected>()) SetVisible(e, set_visible);
                }
                if (mixed_visible) ImGui::PopItemFlag();
                if (Button("Duplicate")) Duplicate();
                SameLine();
                if (Button("Duplicate linked")) DuplicateLinked();
                if (Button("Delete")) Delete();
            }
            RenderEntityControls(FindActiveEntity(R));

            if (CollapsingHeader("Add primitive")) {
                PushID("AddPrimitive");
                static auto selected_type_i = int(PrimitiveType::Cube);
                for (uint i = 0; i < PrimitiveTypes.size(); ++i) {
                    if (i % 3 != 0) SameLine();
                    RadioButton(ToString(PrimitiveTypes[i]).c_str(), &selected_type_i, i);
                }
                const auto selected_type = PrimitiveType(selected_type_i);
                if (auto mesh_data = PrimitiveEditor(selected_type, true)) {
                    R.emplace<PrimitiveType>(AddMesh(std::move(*mesh_data), MeshInstanceCreateInfo{.Name = ToString(selected_type)}).first, selected_type);
                    StartScreenTransform = TransformGizmo::TransformType::Translate;
                }
                PopID();
            }

            if (CollapsingHeader("All objects")) {
                RenderEntitiesTable("All objects", entt::null);
            }
            EndTabItem();
        }

        if (BeginTabItem("Render")) {
            auto &settings = R.get<SceneSettings>(SceneEntity);
            bool settings_changed = false;
            if (ColorEdit3("Background color", settings.ClearColor.float32)) {
                settings.ClearColor.float32[3] = 1.f;
                settings_changed = true;
            }
            bool show_grid = settings.ShowGrid;
            if (Checkbox("Show grid", &show_grid)) {
                settings.ShowGrid = show_grid;
                settings_changed = true;
            }
            if (Button("Recompile shaders")) ShaderRecompileRequested = true;
            SeparatorText("Viewport shading");
            PushID("ViewportShading");
            auto viewport_shading = int(settings.ViewportShading);
            bool viewport_shading_changed = RadioButton("Solid", &viewport_shading, int(ViewportShadingMode::Solid));
            SameLine();
            viewport_shading_changed |= RadioButton("Wireframe", &viewport_shading, int(ViewportShadingMode::Wireframe));
            PopID();

            bool smooth_shading_changed = false;
            bool smooth_shading = settings.SmoothShading;
            if (settings.ViewportShading == ViewportShadingMode::Solid) {
                smooth_shading_changed = Checkbox("Smooth shading", &smooth_shading);
            }

            auto color_mode = int(settings.FaceColorMode);
            bool color_mode_changed = false;
            if (settings.ViewportShading == ViewportShadingMode::Solid) {
                PushID("FaceColorMode");
                AlignTextToFramePadding();
                TextUnformatted("Fill color mode");
                SameLine();
                color_mode_changed |= RadioButton("Mesh", &color_mode, int(FaceColorMode::Mesh));
                SameLine();
                color_mode_changed |= RadioButton("Normals", &color_mode, int(FaceColorMode::Normals));
                PopID();
            }
            if (viewport_shading_changed || color_mode_changed || smooth_shading_changed) {
                settings.ViewportShading = ViewportShadingMode(viewport_shading);
                settings.FaceColorMode = FaceColorMode(color_mode);
                settings.SmoothShading = smooth_shading;
                settings_changed = true;
            }
            if (!R.view<Selected>().empty()) {
                SeparatorText("Selection overlays");
                AlignTextToFramePadding();
                TextUnformatted("Normals");
                for (const auto element : NormalElements) {
                    SameLine();
                    bool show = ElementMaskContains(settings.NormalOverlays, element);
                    const auto type_name = Capitalize(label(element));
                    if (Checkbox(type_name.c_str(), &show)) {
                        SetElementMask(settings.NormalOverlays, element, show);
                        settings_changed = true;
                    }
                }
                bool show_bboxes = settings.ShowBoundingBoxes;
                if (Checkbox("Bounding boxes", &show_bboxes)) {
                    settings.ShowBoundingBoxes = show_bboxes;
                    settings_changed = true;
                }
            }
            {
                SeparatorText("Viewport theme");
                auto &theme = R.get<ViewportTheme>(SceneEntity);
                bool changed{false};
                if (Button("Reset##ViewportTheme")) {
                    theme = Defaults.ViewportTheme;
                    changed = true;
                }
                changed |= ColorEdit3("Wire", &theme.Colors.Wire.x);
                changed |= ColorEdit3("Wire edit", &theme.Colors.WireEdit.x);
                changed |= ColorEdit3("Face normal", &theme.Colors.FaceNormal.x);
                changed |= ColorEdit3("Vertex normal", &theme.Colors.VertexNormal.x);
                changed |= ColorEdit3("Vertex", &theme.Colors.Vertex.x);
                changed |= ColorEdit3("Vertex selected", &theme.Colors.VertexSelected.x);
                changed |= ColorEdit3("Edge selected (incidental)", &theme.Colors.EdgeSelectedIncidental.x);
                changed |= ColorEdit3("Edge selected", &theme.Colors.EdgeSelected.x);
                changed |= ColorEdit4("Face selected (incidental)", &theme.Colors.FaceSelectedIncidental.x);
                changed |= ColorEdit4("Face selected", &theme.Colors.FaceSelected.x);
                changed |= ColorEdit4("Element active", &theme.Colors.ElementActive.x);
                changed |= ColorEdit3("Object active", &theme.Colors.ObjectActive.x);
                changed |= ColorEdit3("Object selected", &theme.Colors.ObjectSelected.x);
                changed |= SliderUInt("Silhouette edge width", &theme.SilhouetteEdgeWidth, 1, 4);
                if (changed) R.patch<ViewportTheme>(SceneEntity, [](auto &) {});
            }
            if (settings_changed) R.patch<SceneSettings>(SceneEntity, [](auto &) {});
            EndTabItem();
        }

        if (BeginTabItem("Camera")) {
            auto &camera = R.get<Camera>(SceneEntity);
            bool changed = false;
            if (Button("Reset##Camera")) {
                camera = Defaults.Camera;
                changed = true;
            }
            changed |= SliderFloat3("Target", &camera.Target.x, -10, 10);
            float fov_deg = glm::degrees(camera.FieldOfViewRad);
            if (SliderFloat("Field of view (deg)", &fov_deg, 1, 180)) {
                camera.FieldOfViewRad = glm::radians(fov_deg);
                changed = true;
            }
            changed |= SliderFloat("Near clip", &camera.NearClip, 0.001f, 10, "%.3f", ImGuiSliderFlags_Logarithmic);
            changed |= SliderFloat("Far clip", &camera.FarClip, 10, 1000, "%.1f", ImGuiSliderFlags_Logarithmic);
            if (changed) R.patch<Camera>(SceneEntity, [](auto &camera) { camera.StopMoving(); });
            EndTabItem();
        }

        if (BeginTabItem("Lights")) {
            auto &lights = R.get<Lights>(SceneEntity);
            bool changed = false;
            if (Button("Reset##Lights")) {
                lights = Defaults.Lights;
                changed = true;
            }
            SeparatorText("View light");
            changed |= ColorEdit3("Color##View", &lights.ViewColor[0]);
            SeparatorText("Ambient light");
            changed |= SliderFloat("Intensity##Ambient", &lights.AmbientIntensity, 0, 1);
            SeparatorText("Directional light");
            changed |= SliderFloat3("Direction##Directional", &lights.Direction[0], -1, 1);
            changed |= ColorEdit3("Color##Directional", &lights.DirectionalColor[0]);
            changed |= SliderFloat("Intensity##Directional", &lights.DirectionalIntensity, 0, 1);
            if (changed) R.patch<Lights>(SceneEntity, [](auto &) {});
            EndTabItem();
        }
        EndTabBar();
    }
}

void Scene::RenderEntitiesTable(std::string name, entt::entity parent) {
    if (MeshEditor::BeginTable(name.c_str(), 3)) {
        static const float CharWidth = CalcTextSize("A").x;
        TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, CharWidth * 10);
        TableSetupColumn("Name");
        TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, CharWidth * 16);
        TableHeadersRow();
        entt::entity activate_entity = entt::null, toggle_selected = entt::null;

        auto render_entity = [&](entt::entity e) {
            PushID(uint(e));
            TableNextColumn();
            AlignTextToFramePadding();
            if (R.all_of<Active>(e)) TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(ImGuiCol_TextSelectedBg));
            TextUnformatted(IdString(e).c_str());
            TableNextColumn();
            TextUnformatted(R.get<Name>(e).Value.c_str());
            TableNextColumn();
            if (!R.all_of<Active>(e) && Button("Activate")) activate_entity = e;
            SameLine();
            if (Button(R.all_of<Selected>(e) ? "Deselect" : "Select")) toggle_selected = e;
            PopID();
        };

        if (parent == entt::null) { // Iterate root entities
            for (const auto &[entity, name] : R.view<const Name>().each()) {
                const auto *node = R.try_get<SceneNode>(entity);
                if (!node || node->Parent == entt::null) render_entity(entity);
            }
        } else { // Iterate children
            for (const auto child : Children{&R, parent}) render_entity(child);
        }

        if (activate_entity != entt::null) Select(activate_entity);
        else if (toggle_selected != entt::null) ToggleSelected(toggle_selected);
        EndTable();
    }
}
