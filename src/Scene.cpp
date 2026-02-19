#include "Scene.h"
#include "SceneDefaults.h"
#include "Widgets.h" // imgui

#include "Armature.h"
#include "BBox.h"
#include "Bindless.h"
#include "CameraData.h"
#include "Entity.h"
#include "Excitable.h"
#include "LightTypes.h"
#include "OrientationGizmo.h"
#include "Shader.h"
#include "SvgResource.h"
#include "Timer.h"
#include "gltf/GltfLoader.h"
#include "gpu/ObjectPickPushConstants.h"
#include "gpu/PunctualLight.h"
#include "gpu/SilhouetteEdgeColorPushConstants.h"
#include "gpu/SilhouetteEdgeDepthObjectPushConstants.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"

#include <entt/entity/registry.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <imgui_internal.h>

#include "Variant.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <numbers>
#include <unordered_map>
#include <unordered_set>

using std::ranges::any_of, std::ranges::all_of, std::ranges::distance, std::ranges::find, std::ranges::find_if, std::ranges::fold_left, std::ranges::to;
using std::views::filter, std::views::iota, std::views::take, std::views::transform;

namespace {
const SceneDefaults Defaults{};
constexpr float Pi = std::numbers::pi_v<float>;
constexpr float DefaultSpotOuterAngle = Pi / 4.f; // 45 deg
constexpr float DefaultSpotBlend = 0.15f;
constexpr float DefaultLightIntensity = 75.f;
constexpr float DefaultPointRange = 10.f;

using namespace he;

struct MeshSelection {
    std::unordered_set<uint32_t> Handles{};
};

// Most recently selected element per mesh (remembered even when not selected).
struct MeshActiveElement {
    uint32_t Handle;
};

// Tag to request overlay + element-state buffer refresh after mesh geometry changes.
struct MeshGeometryDirty {};
// Tags for light events.
struct LightDataDirty {};
struct LightWireframeDirty {};

// Tracks pending transform for shader-based preview during Edit mode gizmo manipulation.
// Presence indicates active transform; removal triggers UBO clear.
struct PendingTransform {
    vec3 Pivot{};
    quat PivotR{1, 0, 0, 0};
    vec3 P{}; // Translation delta
    quat R{1, 0, 0, 0}; // Rotation delta
    vec3 S{1, 1, 1}; // Scale delta
};

void WaitFor(vk::Fence fence, vk::Device device) {
    if (auto wait_result = device.waitForFences(fence, VK_TRUE, UINT64_MAX); wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    device.resetFences(fence);
}

float ClampCos(float cos_theta) {
    return std::clamp(cos_theta, -1.f, 1.f);
}

float AngleFromCos(float cos_theta) {
    return std::acos(ClampCos(cos_theta));
}

PunctualLight MakeDefaultLight(uint32_t type = LightTypePoint) {
    const float outer = DefaultSpotOuterAngle;
    const float inner = outer * (1.f - DefaultSpotBlend);
    return {
        .Range = type == LightTypeDirectional ? 0.f : DefaultPointRange,
        .Color = {1.f, 1.f, 1.f},
        .Intensity = DefaultLightIntensity,
        .InnerConeCos = type == LightTypeSpot ? std::cos(inner) : 0.f,
        .OuterConeCos = type == LightTypeSpot ? std::cos(outer) : 0.f,
        .Type = type,
    };
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

struct ExtrasWireframe {
    MeshData Data;
    std::vector<uint8_t> VertexClasses{}; // Empty means all VCLASS_NONE (no buffer needed).

    uint32_t AddVertex(vec3 pos, uint8_t vclass) {
        const uint32_t i = uint32_t(Data.Positions.size());
        Data.Positions.push_back(pos);
        VertexClasses.push_back(vclass);
        return i;
    }
    void AddEdge(uint32_t a, uint32_t b) { Data.Edges.push_back({a, b}); }

    void AddCircle(float radius, uint32_t segments, float z, uint8_t vclass, uint32_t edge_stride = 1) {
        const uint32_t base = uint32_t(Data.Positions.size());
        for (uint32_t i = 0; i < segments; ++i) {
            const float angle = float(i) * 2.f * Pi / float(segments);
            AddVertex({radius * std::cos(angle), radius * std::sin(angle), z}, vclass);
        }
        for (uint32_t i = 0; i < segments; i += edge_stride) AddEdge(base + i, base + (i + 1) % segments);
    }

    void AddDiamond(float radius, uint8_t vclass, vec3 axis1, vec3 axis2, vec3 center = {}) {
        const uint32_t base = uint32_t(Data.Positions.size());
        for (auto a : {axis1, axis2, -axis1, -axis2}) AddVertex(center + radius * a, vclass);
        for (uint32_t i = 0; i < 4; ++i) AddEdge(base + i, base + (i + 1) % 4);
    }
};

constexpr uint8_t VClassNone = 0, VClassBillboard = 1, VClassSpotCone = 2, VClassScreenspace = 3, VClassGroundPoint = 4;
constexpr uint32_t SpotConeSegments = 32;

template<typename F>
void AppendExtrasDraw(entt::registry &R, DrawListBuilder &dl, DrawBatchInfo &batch, F &&customize_draw) {
    batch = dl.BeginBatch();
    for (auto [entity, mesh_buffers, models] : R.view<ObjectExtrasTag, MeshBuffers, ModelsBuffer>().each()) {
        if (mesh_buffers.EdgeIndices.Count == 0) continue;
        auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.EdgeIndices, models);
        if (const auto *vcr = R.try_get<VertexClass>(entity)) draw.VertexClassOffset = vcr->Offset;
        customize_draw(draw, models);
        AppendDraw(dl, batch, mesh_buffers.EdgeIndices, models, draw);
    }
}

namespace {
// Build a wireframe frustum mesh in camera-local space (camera looks down -Z).
// Returns positions + edges for a line-only mesh (no faces).
MeshData BuildCameraFrustumMesh(const CameraData &cd) {
    float display_near{0.01f}, display_far{5.f};
    float near_half_w{0.f}, near_half_h{0.f}, far_half_w{0.f}, far_half_h{0.f};
    if (const auto *perspective = std::get_if<Perspective>(&cd)) {
        // Clamp far plane for display so wireframe doesn't extend to infinity.
        display_near = perspective->NearClip;
        display_far = std::min(perspective->FarClip.value_or(5.f), 5.f);

        const float aspect = AspectRatio(cd);
        near_half_h = display_near * std::tan(perspective->FieldOfViewRad * 0.5f);
        near_half_w = near_half_h * aspect;
        far_half_h = display_far * std::tan(perspective->FieldOfViewRad * 0.5f);
        far_half_w = far_half_h * aspect;
    } else if (const auto *orthographic = std::get_if<Orthographic>(&cd)) {
        display_near = orthographic->NearClip;
        display_far = std::min(orthographic->FarClip, 5.f);
        near_half_w = far_half_w = orthographic->Mag.x;
        near_half_h = far_half_h = orthographic->Mag.y;
    }

    // 8 corner vertices: near plane (0-3), far plane (4-7)
    // Up-triangle indicator: 2 extra vertices (8-9)
    std::vector<vec3> positions{
        // Near plane (looking down -Z, so near is at -near)
        {-near_half_w, -near_half_h, -display_near}, // near bottom-left
        {near_half_w, -near_half_h, -display_near}, // near bottom-right
        {near_half_w, near_half_h, -display_near}, // near top-right
        {-near_half_w, near_half_h, -display_near}, // near top-left
        // Far plane
        {-far_half_w, -far_half_h, -display_far}, // far bottom-left
        {far_half_w, -far_half_h, -display_far}, // far bottom-right
        {far_half_w, far_half_h, -display_far}, // far top-right
        {-far_half_w, far_half_h, -display_far}, // far top-left
        // Up-triangle indicator (above far top edge)
        {-far_half_w, far_half_h, -display_far}, // triangle base left
        {far_half_w, far_half_h, -display_far}, // triangle base right
        {0.f, far_half_h + far_half_h * 0.3f, -display_far}, // triangle apex
    };

    // clang-format off
    std::vector<std::array<uint32_t, 2>> edges{
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Near plane quad
        {4, 5}, {5, 6}, {6, 7}, {7, 4}, // Far plane quad
        {0, 4}, {1, 5}, {2, 6}, {3, 7}, // Connecting edges (near to far)
        {8, 10}, {10, 9}, // Up-triangle
    };
    // clang-format on

    return {.Positions = std::move(positions), .Edges = std::move(edges)};
}

ExtrasWireframe BuildLightMesh(const PunctualLight &light) {
    ExtrasWireframe wf;

    const auto add_range_circle = [&](float range) {
        if (range > 0.f) wf.AddCircle(range, 32, 0.f, VClassBillboard);
    };

    if (light.Type == LightTypePoint) {
        add_range_circle(light.Range);
    } else if (light.Type == LightTypeDirectional) {
        // Sun rays: 8 radial directions, each with two dashed line segments (screenspace units).
        constexpr uint32_t ray_count = 8;
        constexpr float d0s = 14.f, d0e = 16.f, d1s = 18.f, d1e = 20.f;
        for (uint32_t i = 0; i < ray_count; ++i) {
            const float angle = float(i) * 2.f * Pi / float(ray_count);
            const float dx = std::cos(angle), dy = std::sin(angle);
            const auto a = wf.AddVertex({d0s * dx, d0s * dy, 0.f}, VClassScreenspace);
            const auto b = wf.AddVertex({d0e * dx, d0e * dy, 0.f}, VClassScreenspace);
            const auto c = wf.AddVertex({d1s * dx, d1s * dy, 0.f}, VClassScreenspace);
            const auto d = wf.AddVertex({d1e * dx, d1e * dy, 0.f}, VClassScreenspace);
            wf.AddEdge(a, b);
            wf.AddEdge(c, d);
        }
    } else if (light.Type == LightTypeSpot) {
        constexpr float depth = 2.f;
        const float outer_angle = std::min(AngleFromCos(light.OuterConeCos), glm::radians(89.f));
        const float inner_angle = std::min(AngleFromCos(light.InnerConeCos), outer_angle);
        const float outer_radius = depth * std::tan(outer_angle);
        const float inner_radius = depth * std::tan(inner_angle);
        add_range_circle(light.Range);
        wf.AddCircle(outer_radius, SpotConeSegments, -depth, VClassNone);
        if (inner_radius > 0.f) wf.AddCircle(inner_radius, SpotConeSegments, -depth, VClassNone);

        // Cone spokes: apex (VCLASS_NONE) to base (VCLASS_SPOT_CONE).
        for (uint32_t i = 0; i < SpotConeSegments; ++i) {
            const float angle = float(i) * 2.f * Pi / float(SpotConeSegments);
            const auto ai = wf.AddVertex({0.f, 0.f, 0.f}, VClassNone);
            const auto bi = wf.AddVertex({outer_radius * std::cos(angle), outer_radius * std::sin(angle), -depth}, VClassSpotCone);
            wf.AddEdge(ai, bi);
        }
    }

    // Light icon shared by all types: center diamond, dashed indicator circles, ground line + diamond.
    wf.AddDiamond(2.7f, VClassScreenspace, {1, 0, 0}, {0, 1, 0});
    wf.AddCircle(9.f, 16, 0.f, VClassScreenspace, 2);
    wf.AddCircle(9.f * 1.33f, 20, 0.f, VClassScreenspace, 2);

    // Ground line from light origin (y=1) down to ground plane (y=0).
    const auto top = wf.AddVertex({0.f, 1.f, 0.f}, VClassGroundPoint);
    const auto bot = wf.AddVertex({0.f, 0.f, 0.f}, VClassGroundPoint);
    wf.AddEdge(top, bot);
    wf.AddDiamond(3.f, VClassGroundPoint, {1, 0, 0}, {0, 0, 1});

    return wf;
}

struct ViewportContext {
    float Distance, AspectRatio;
};

bool RenderCameraLensEditor(CameraData &camera_data, std::optional<ViewportContext> viewport = {}) {
    bool lens_changed = false;

    int proj_i = std::holds_alternative<Orthographic>(camera_data) ? 1 : 0;
    const char *proj_names[]{"Perspective", "Orthographic"};
    if (Combo("Projection", &proj_i, proj_names, IM_ARRAYSIZE(proj_names))) {
        if (proj_i == 0 && !std::holds_alternative<Perspective>(camera_data)) {
            camera_data = PerspectiveFromOrthographic(std::get<Orthographic>(camera_data), viewport ? std::optional<float>{viewport->Distance} : std::nullopt);
            lens_changed = true;
        } else if (proj_i == 1 && !std::holds_alternative<Orthographic>(camera_data)) {
            camera_data = OrthographicFromPerspective(
                std::get<Perspective>(camera_data),
                viewport ? std::optional<float>{viewport->Distance} : std::nullopt,
                viewport ? std::optional<float>{viewport->AspectRatio} : std::nullopt
            );
            lens_changed = true;
        }
    }

    if (auto *perspective = std::get_if<Perspective>(&camera_data)) {
        float fov_deg = glm::degrees(perspective->FieldOfViewRad);
        if (SliderFloat("Field of view (deg)", &fov_deg, 1.f, 179.f)) {
            perspective->FieldOfViewRad = glm::radians(fov_deg);
            lens_changed = true;
        }
        const float near_max = perspective->FarClip ? std::max(*perspective->FarClip - MinNearFarDelta, MinNearFarDelta) : MaxFarClip;
        lens_changed |= SliderFloat("Near clip", &perspective->NearClip, 0.001f, near_max);
        bool infinite_far = !perspective->FarClip.has_value();
        if (Checkbox("Infinite far clip", &infinite_far)) {
            if (infinite_far) perspective->FarClip.reset();
            else perspective->FarClip = std::max(perspective->NearClip + MinNearFarDelta, MaxFarClip);
            lens_changed = true;
        }
        if (perspective->FarClip) {
            lens_changed |= SliderFloat("Far clip", &*perspective->FarClip, perspective->NearClip + MinNearFarDelta, MaxFarClip);
        }
        if (!viewport) {
            float aspect = perspective->AspectRatio.value_or(DefaultAspectRatio);
            if (SliderFloat("Aspect ratio", &aspect, 0.1f, 5.f)) {
                perspective->AspectRatio = aspect;
                lens_changed = true;
            }
        }
    } else if (auto *orthographic = std::get_if<Orthographic>(&camera_data)) {
        lens_changed |= SliderFloat("X Mag", &orthographic->Mag.x, 0.01f, 100.f);
        lens_changed |= SliderFloat("Y Mag", &orthographic->Mag.y, 0.01f, 100.f);
        lens_changed |= SliderFloat("Near clip", &orthographic->NearClip, 0.001f, orthographic->FarClip - MinNearFarDelta);
        lens_changed |= SliderFloat("Far clip", &orthographic->FarClip, orthographic->NearClip + MinNearFarDelta, MaxFarClip);
    }
    return lens_changed;
}

vec3 ComputeElementLocalPosition(const Mesh &mesh, Element element, uint32_t handle) {
    if (element == Element::Vertex) return mesh.GetPosition(VH{handle});
    if (element == Element::Edge) {
        const auto heh = mesh.GetHalfedge(EH{handle}, 0);
        return (mesh.GetPosition(mesh.GetFromVertex(heh)) + mesh.GetPosition(mesh.GetToVertex(heh))) * 0.5f;
    }
    return mesh.CalcFaceCentroid(FH{handle});
}

vec3 ComputeElementWorldPosition(const entt::registry &r, entt::entity instance_entity, Element element, uint32_t handle) {
    const auto &mesh = r.get<Mesh>(r.get<MeshInstance>(instance_entity).MeshEntity);
    const auto local_pos = ComputeElementLocalPosition(mesh, element, handle);
    const auto &wt = r.get<WorldTransform>(instance_entity);
    return {wt.Position + glm::rotate(Vec4ToQuat(wt.Rotation), wt.Scale * local_pos)};
}

struct EditTransformContext {
    std::unordered_map<entt::entity, entt::entity> TransformInstances; // excludes frozen, for transforms
};

EditTransformContext BuildEditTransformContext(const entt::registry &r) {
    return {ComputePrimaryEditInstances(r, false)};
}

PunctualLight GetLight(const SceneBuffers &buffers, uint32_t index) {
    return reinterpret_cast<const PunctualLight *>(buffers.LightBuffer.GetMappedData().data())[index];
}
void SetLight(SceneBuffers &buffers, uint32_t index, const PunctualLight &light) {
    buffers.LightBuffer.Update(as_bytes(light), vk::DeviceSize(index) * sizeof(PunctualLight));
}

void ResetObjectPickKeys(SceneBuffers &buffers) {
    auto bytes = buffers.ObjectPickKeyBuffer.GetMappedData();
    auto *keys = reinterpret_cast<uint32_t *>(bytes.data());
    std::fill_n(keys, SceneBuffers::MaxSelectableObjects, std::numeric_limits<uint32_t>::max());
}

namespace changes {
using namespace entt::literals;
constexpr auto
    Selected = "selected_changes"_hs,
    ActiveInstance = "active_instance_changes"_hs,
    Rerecord = "rerecord_changes"_hs,
    MeshSelection = "mesh_selection_changes"_hs,
    MeshActiveElement = "mesh_active_element_changes"_hs,
    MeshGeometry = "mesh_geometry_changes"_hs,
    Excitable = "excitable_changes"_hs,
    ExcitedVertex = "excited_vertex_changes"_hs,
    ModelsBuffer = "models_buffer_changes"_hs,
    SceneSettings = "scene_settings_changes"_hs,
    InteractionMode = "interaction_mode_changes"_hs,
    ViewportTheme = "viewport_theme_changes"_hs,
    SceneView = "scene_view_changes"_hs,
    CameraLens = "camera_lens_changes"_hs,
    TransformPending = "transform_pending_changes"_hs,
    TransformEnd = "transform_end_changes"_hs;
} // namespace changes

struct DeformSlots {
    uint32_t BoneDeformOffset{InvalidOffset}, ArmatureDeformOffset{InvalidOffset}, MorphDeformOffset{InvalidOffset};
    uint32_t MorphTargetCount{0};
    // Per-instance morph weights: buffer_index -> offset (weights are per-node in glTF)
    std::unordered_map<uint32_t, uint32_t> MorphWeightsByBufferIndex;
};

std::unordered_map<entt::entity, DeformSlots> BuildDeformSlots(const entt::registry &r, const MeshStore &meshes) {
    std::unordered_map<entt::entity, DeformSlots> result;
    for (const auto [_, mi, modifier] : r.view<const MeshInstance, const ArmatureModifier>().each()) {
        if (result.contains(mi.MeshEntity)) continue;
        const auto &mesh = r.get<const Mesh>(mi.MeshEntity);
        const auto bone_deform = meshes.GetBoneDeformRange(mesh.GetStoreId());
        if (bone_deform.Count == 0) continue;
        if (const auto *pose_state = r.try_get<const ArmaturePoseState>(modifier.ArmatureDataEntity)) {
            result[mi.MeshEntity] = {
                .BoneDeformOffset = bone_deform.Offset,
                .ArmatureDeformOffset = pose_state->GpuDeformRange.Offset,
                .MorphDeformOffset = InvalidOffset,
                .MorphTargetCount = 0,
                .MorphWeightsByBufferIndex = {},
            };
        }
    }
    // Add morph target slots for mesh instances with morph data (per-instance weights)
    for (const auto [instance_entity, mi, morph_state, ri] : r.view<const MeshInstance, const MorphWeightState, const RenderInstance>().each()) {
        const auto mesh_entity = mi.MeshEntity;
        const auto &mesh = r.get<const Mesh>(mesh_entity);
        const auto morph_range = meshes.GetMorphTargetRange(mesh.GetStoreId());
        if (morph_range.Count == 0) continue;
        auto &slots = result[mesh_entity];
        slots.MorphDeformOffset = morph_range.Offset;
        slots.MorphTargetCount = meshes.GetMorphTargetCount(mesh.GetStoreId());
        slots.MorphWeightsByBufferIndex[ri.BufferIndex] = morph_state.GpuWeightRange.Offset;
    }
    return result;
}

void PatchMorphWeights(DrawListBuilder &dl, size_t draws_before, const DeformSlots &deform) {
    if (deform.MorphWeightsByBufferIndex.empty()) return;
    for (size_t i = draws_before; i < dl.Draws.size(); ++i) {
        if (auto it = deform.MorphWeightsByBufferIndex.find(dl.Draws[i].FirstInstance); it != deform.MorphWeightsByBufferIndex.end()) {
            dl.Draws[i].MorphWeightsOffset = it->second;
        }
    }
}
} // namespace

struct Scene::SelectionSlotHandles {
    explicit SelectionSlotHandles(DescriptorSlots &slots)
        : Slots(slots),
          HeadImage(slots.Allocate(SlotType::Image)),
          SelectionCounter(slots.Allocate(SlotType::Buffer)),
          ObjectPickKey(slots.Allocate(SlotType::Buffer)),
          ElementPickCandidates(slots.Allocate(SlotType::Buffer)),
          SelectionBitset(slots.Allocate(SlotType::Buffer)),
          ObjectIdSampler(slots.Allocate(SlotType::Sampler)),
          DepthSampler(slots.Allocate(SlotType::Sampler)),
          SilhouetteSampler(slots.Allocate(SlotType::Sampler)) {}

    ~SelectionSlotHandles() {
        Slots.Release({SlotType::Image, HeadImage});
        Slots.Release({SlotType::Buffer, SelectionCounter});
        Slots.Release({SlotType::Buffer, ObjectPickKey});
        Slots.Release({SlotType::Buffer, ElementPickCandidates});
        Slots.Release({SlotType::Buffer, SelectionBitset});
        Slots.Release({SlotType::Sampler, ObjectIdSampler});
        Slots.Release({SlotType::Sampler, DepthSampler});
        Slots.Release({SlotType::Sampler, SilhouetteSampler});
    }

    DescriptorSlots &Slots;
    uint32_t HeadImage, SelectionCounter, ObjectPickKey, ElementPickCandidates, SelectionBitset, ObjectIdSampler, DepthSampler, SilhouetteSampler;
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
      Meshes{std::make_unique<MeshStore>(Buffers->Ctx)} {
    // Reactive storage subscriptions for deferred once-per-frame processing
    using namespace entt::literals;

    R.storage<entt::reactive>(changes::Selected)
        .on_construct<Selected>()
        .on_destroy<Selected>();
    R.storage<entt::reactive>(changes::ActiveInstance)
        .on_construct<Active>()
        .on_destroy<Active>();
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
    R.storage<entt::reactive>(changes::MeshActiveElement)
        .on_construct<MeshActiveElement>()
        .on_update<MeshActiveElement>();
    R.storage<entt::reactive>(changes::MeshGeometry)
        .on_construct<MeshGeometryDirty>();
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
        .on_construct<ViewCamera>()
        .on_update<ViewCamera>()
        .on_construct<Lights>()
        .on_update<Lights>()
        .on_construct<ViewportExtent>()
        .on_update<ViewportExtent>()
        .on_construct<SceneEditMode>()
        .on_update<SceneEditMode>();
    R.storage<entt::reactive>(changes::CameraLens)
        .on_construct<CameraData>()
        .on_update<CameraData>();
    R.storage<entt::reactive>(changes::TransformPending)
        .on_construct<PendingTransform>()
        .on_update<PendingTransform>();
    R.storage<entt::reactive>(changes::TransformEnd)
        .on_destroy<StartTransform>();

    DestroyTracker->Bind(R);

    SceneEntity = R.create();
    R.emplace<SceneSettings>(SceneEntity);
    R.emplace<SceneInteraction>(SceneEntity);
    R.emplace<SceneEditMode>(SceneEntity);
    R.emplace<ViewportTheme>(SceneEntity, Defaults.ViewportTheme);
    R.emplace<ViewCamera>(SceneEntity, Defaults.ViewCamera);
    R.emplace<Lights>(SceneEntity, Defaults.Lights);
    R.emplace<ViewportExtent>(SceneEntity);
    R.emplace<AnimationTimeline>(SceneEntity);

    BoxSelectZeroBits.assign(SceneBuffers::BoxSelectBitsetWords, 0);
    ResetObjectPickKeys(*Buffers);

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
        mvk::RecordBufferToSampledImageUpload(*cb, *staging_buffer, *image.Image, width, height, ColorSubresourceRange);
        cb->end();

        vk::SubmitInfo submit;
        submit.setCommandBuffers(*cb);
        Vk.Queue.submit(submit, *OneShotFence);
        WaitFor(*OneShotFence, Vk.Device);
    } // staging buffer is destroyed here

    Buffers->Ctx.ReclaimRetiredBuffers();

    return image;
}

void Scene::CreateSvgResource(std::unique_ptr<SvgResource> &svg, std::filesystem::path path) {
    const auto RenderBitmap = [this](std::span<const std::byte> data, uint32_t width, uint32_t height) {
        return RenderBitmapToImage(data, width, height);
    };
    svg = std::make_unique<SvgResource>(Vk.Device, RenderBitmap, std::move(path));
}

void Scene::LoadIcons() {
    static const std::filesystem::path svg_path{"res/svg/"};
    CreateSvgResource(Icons.Select, svg_path / "select.svg");
    CreateSvgResource(Icons.SelectBox, svg_path / "select_box.svg");
    CreateSvgResource(Icons.Move, svg_path / "move.svg");
    CreateSvgResource(Icons.Rotate, svg_path / "rotate.svg");
    CreateSvgResource(Icons.Scale, svg_path / "scale.svg");
    CreateSvgResource(Icons.Universal, svg_path / "transform.svg");

    CreateSvgResource(AnimIcons.Play, svg_path / "play.svg");
    CreateSvgResource(AnimIcons.Pause, svg_path / "pause.svg");
    CreateSvgResource(AnimIcons.JumpStart, svg_path / "jump_start.svg");
    CreateSvgResource(AnimIcons.JumpEnd, svg_path / "jump_end.svg");
}

const AnimationTimeline &Scene::GetTimeline() const { return R.get<const AnimationTimeline>(SceneEntity); }

void Scene::ApplyTimelineAction(const AnimationTimelineAction &action) {
    auto set_frame = [&](int frame) {
        R.patch<AnimationTimeline>(SceneEntity, [&](auto &tl) { tl.CurrentFrame = frame; });
        PlaybackFrame = frame;
    };
    std::visit(
        overloaded{
            [&](timeline_action::TogglePlay) { R.patch<AnimationTimeline>(SceneEntity, [](auto &tl) { tl.Playing = !tl.Playing; }); },
            [&](timeline_action::SetFrame a) { set_frame(a.Frame); },
            [&](timeline_action::SetStartFrame a) { R.patch<AnimationTimeline>(SceneEntity, [&](auto &tl) { tl.StartFrame = a.Frame; }); },
            [&](timeline_action::SetEndFrame a) { R.patch<AnimationTimeline>(SceneEntity, [&](auto &tl) { tl.EndFrame = a.Frame; }); },
            [&](timeline_action::JumpToStart) { set_frame(R.get<AnimationTimeline>(SceneEntity).StartFrame); },
            [&](timeline_action::JumpToEnd) { set_frame(R.get<AnimationTimeline>(SceneEntity).EndFrame); },
        },
        action
    );
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
    { // Selected/Active instance changes - update instance state buffer
        auto &selected_tracker = R.storage<entt::reactive>(changes::Selected);
        auto &active_tracker = R.storage<entt::reactive>(changes::ActiveInstance);
        if (!selected_tracker.empty()) request(RenderRequest::ReRecord);

        // Helper to update instance state for a given entity
        const auto update_instance_state = [&](entt::entity instance_entity) {
            if (auto *mi = R.try_get<MeshInstance>(instance_entity)) {
                if (const auto buffer_index = GetModelBufferIndex(R, instance_entity)) {
                    uint8_t state = 0;
                    if (R.all_of<Selected>(instance_entity)) state |= ElementStateSelected;
                    if (R.all_of<Active>(instance_entity)) state |= ElementStateActive;
                    const auto mesh_entity = mi->MeshEntity;
                    R.patch<ModelsBuffer>(mesh_entity, [&](auto &mb) {
                        mb.InstanceStates.Update(as_bytes(state), *buffer_index * sizeof(uint8_t));
                    });
                }
            }
        };

        for (auto instance_entity : selected_tracker) {
            update_instance_state(instance_entity);
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
        for (auto instance_entity : active_tracker) {
            update_instance_state(instance_entity);
            // If looking through a camera and a different camera becomes active, snap to it.
            if (SavedViewCamera && R.all_of<Active>(instance_entity) && R.all_of<CameraData>(instance_entity)) {
                SnapToCamera(instance_entity);
            }
        }
    }
    if (!R.storage<entt::reactive>(changes::Rerecord).empty() || !DestroyTracker->Storage.empty()) {
        request(RenderRequest::ReRecord);
    }

    const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
    const bool is_edit_mode = interaction_mode == InteractionMode::Edit;
    const auto edit_transform_context = is_edit_mode ? BuildEditTransformContext(R) : EditTransformContext{};
    const auto orbit_to_active = [&](entt::entity instance_entity, Element element, uint32_t handle) {
        if (!OrbitToActive) return;
        const auto world_pos = ComputeElementWorldPosition(R, instance_entity, element, handle);
        R.patch<ViewCamera>(SceneEntity, [&](auto &camera) {
            if (const auto dir = world_pos - camera.Target; glm::dot(dir, dir) >= 1e-6f) {
                camera.SetTargetDirection(glm::normalize(dir));
            }
        });
    };

    for (auto mesh_entity : R.storage<entt::reactive>(changes::MeshSelection)) {
        if (R.all_of<MeshSelection>(mesh_entity)) dirty_element_state_meshes.insert(mesh_entity);
    }
    if (const auto &tracker = R.storage<entt::reactive>(changes::MeshActiveElement); !tracker.empty()) {
        const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
        const auto active_entity = FindActiveEntity(R);
        const auto *active_mi = R.try_get<MeshInstance>(active_entity);
        for (auto mesh_entity : tracker) {
            dirty_element_state_meshes.insert(mesh_entity);
            if (const auto *active_element = R.try_get<MeshActiveElement>(mesh_entity);
                active_element && edit_mode != Element::None && active_mi && active_mi->MeshEntity == mesh_entity) {
                orbit_to_active(active_entity, edit_mode, active_element->Handle);
            }
        }
    }
    for (auto instance_entity : R.storage<entt::reactive>(changes::ExcitedVertex)) {
        if (const auto *mi = R.try_get<MeshInstance>(instance_entity)) dirty_element_state_meshes.insert(mi->MeshEntity);
        if (const auto *ev = R.try_get<ExcitedVertex>(instance_entity)) orbit_to_active(instance_entity, Element::Vertex, ev->Vertex);
    }
    for (auto camera_entity : R.storage<entt::reactive>(changes::CameraLens)) {
        if (const auto *cd = R.try_get<CameraData>(camera_entity)) {
            SetMeshPositions(R.get<MeshInstance>(camera_entity).MeshEntity, BuildCameraFrustumMesh(*cd).Positions);
            // If looking through this camera, trigger a ViewCamera update so the SceneView
            // handler re-derives the widened FOV from the updated CameraData.
            if (SavedViewCamera && R.all_of<Active>(camera_entity)) {
                R.patch<ViewCamera>(SceneEntity, [](auto &) {});
            }
        }
    }
    bool light_count_changed = false;
    {
        const auto required_size = vk::DeviceSize(R.storage<LightIndex>().size()) * sizeof(PunctualLight);
        if (Buffers->LightBuffer.UsedSize != required_size) {
            Buffers->LightBuffer.Reserve(required_size);
            Buffers->LightBuffer.UsedSize = required_size;
            light_count_changed = true;
        }
    }
    if (!R.view<LightDataDirty>().empty() || light_count_changed) request(RenderRequest::Submit);
    for (auto light_entity : R.view<LightWireframeDirty, LightIndex, MeshInstance>()) {
        const auto light = GetLight(*Buffers, R.get<const LightIndex>(light_entity).Value);
        auto wireframe = BuildLightMesh(light);
        const auto mesh_entity = R.get<MeshInstance>(light_entity).MeshEntity;
        if (const auto *old_vcr = R.try_get<VertexClass>(mesh_entity)) {
            const auto old_vertex_count = R.get<const Mesh>(mesh_entity).VertexCount();
            Buffers->VertexClassBuffer.Release({old_vcr->Offset, old_vertex_count});
            R.remove<VertexClass>(mesh_entity);
        }

        R.replace<Mesh>(mesh_entity, Meshes->CreateMesh(std::move(wireframe.Data)));
        const auto &mesh = R.get<const Mesh>(mesh_entity);

        if (auto *mb = R.try_get<MeshBuffers>(mesh_entity)) Buffers->Release(*mb);
        R.erase<MeshBuffers>(mesh_entity);
        R.emplace<MeshBuffers>(
            mesh_entity, Meshes->GetVerticesRange(mesh.GetStoreId()),
            Buffers->CreateIndices(mesh.CreateTriangleIndices(), IndexKind::Face),
            Buffers->CreateIndices(mesh.CreateEdgeIndices(), IndexKind::Edge),
            Buffers->CreateIndices(CreateVertexIndices(mesh), IndexKind::Vertex)
        );

        if (!wireframe.VertexClasses.empty()) {
            const auto range = Buffers->VertexClassBuffer.Allocate(std::span<const uint8_t>(wireframe.VertexClasses));
            R.emplace<VertexClass>(mesh_entity, range.Offset);
        }

        request(RenderRequest::ReRecord);
    }
    if (auto &tracker = R.storage<entt::reactive>(changes::MeshGeometry); !tracker.empty()) {
        for (auto mesh_entity : tracker) {
            if (HasSelectedInstance(R, mesh_entity)) dirty_overlay_meshes.insert(mesh_entity);
        }
        request(RenderRequest::Submit);
    }
    if (!R.storage<entt::reactive>(changes::ModelsBuffer).empty()) request(RenderRequest::Submit);
    if (!R.storage<entt::reactive>(changes::ViewportTheme).empty()) {
        Buffers->ViewportThemeUBO.Update(as_bytes(R.get<const ViewportTheme>(SceneEntity)));
        request(RenderRequest::Submit);
    }
    if (!R.storage<entt::reactive>(changes::SceneSettings).empty()) {
        request(RenderRequest::ReRecord);
        dirty_overlay_meshes.merge(GetSelectedMeshEntities(R));
    }
    if (!R.storage<entt::reactive>(changes::InteractionMode).empty()) {
        request(RenderRequest::ReRecord);
        for (const auto [mesh_entity, selection] : R.view<MeshSelection>().each()) {
            if (!selection.Handles.empty()) dirty_element_state_meshes.insert(mesh_entity);
        }
        for (const auto [_, mi, __] : R.view<const MeshInstance, const Excitable>().each()) {
            dirty_element_state_meshes.insert(mi.MeshEntity);
        }
    }
    // Handle Edit mode transform commit when StartTransform is cleared.
    if (!R.storage<entt::reactive>(changes::TransformEnd).empty()) {
        if (is_edit_mode) {
            const auto &pending = R.get<const PendingTransform>(SceneEntity);
            const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
            // Apply edit transform once per selected mesh via a representative selected instance.
            // This keeps linked instances from receiving duplicate per-instance edits.
            for (const auto &[mesh_entity, instance_entity] : edit_transform_context.TransformInstances) {
                if (HasFrozenInstance(R, mesh_entity)) continue;
                const auto &mesh = R.get<const Mesh>(mesh_entity);
                const auto vertices = mesh.GetVerticesSpan();
                const auto &selection = R.get<const MeshSelection>(mesh_entity);
                if (selection.Handles.empty()) continue;
                const auto &wt = R.get<const WorldTransform>(instance_entity);
                const auto wt_rot = Vec4ToQuat(wt.Rotation);
                const auto inv_rot = glm::conjugate(wt_rot);
                const auto inv_scale = 1.f / wt.Scale;
                const auto vertex_handles = ConvertSelectionElement(selection, mesh, edit_mode, Element::Vertex);
                for (const auto vi : vertex_handles) {
                    const auto local_pos = vertices[vi].Position;
                    const auto world_pos = wt.Position + glm::rotate(wt_rot, wt.Scale * local_pos);
                    auto offset = world_pos - pending.Pivot;
                    offset = pending.S * offset;
                    offset = glm::rotate(pending.R, offset);
                    const auto new_world = pending.Pivot + offset + pending.P;
                    const auto new_local = inv_scale * glm::rotate(inv_rot, new_world - wt.Position);
                    Meshes->SetPosition(mesh, vi, new_local);
                }
                Meshes->UpdateNormals(mesh);
                dirty_overlay_meshes.insert(mesh_entity);
            }
        }
        R.remove<PendingTransform>(SceneEntity);
    }
    if (!R.storage<entt::reactive>(changes::SceneView).empty() ||
        !R.storage<entt::reactive>(changes::TransformPending).empty() ||
        !R.storage<entt::reactive>(changes::SceneSettings).empty() ||
        !R.storage<entt::reactive>(changes::InteractionMode).empty() ||
        !R.storage<entt::reactive>(changes::TransformEnd).empty() ||
        light_count_changed) {
        // When looking through a scene camera, keep the ViewCamera's widened FOV in sync
        // with the current viewport aspect ratio (handles viewport resize).
        if (SavedViewCamera) {
            const auto active_entity = FindActiveEntity(R);
            if (const auto *cd = active_entity != entt::null ? R.try_get<CameraData>(active_entity) : nullptr) {
                const auto extent = R.get<const ViewportExtent>(SceneEntity).Value;
                const float viewport_aspect = extent.width == 0 || extent.height == 0 ? 1.f : float(extent.width) / float(extent.height);
                R.get<ViewCamera>(SceneEntity).Data = WidenForLookThrough(*cd, viewport_aspect);
            }
        }
        const auto &camera = R.get<const ViewCamera>(SceneEntity);
        const auto &lights = R.get<const Lights>(SceneEntity);
        const auto *pending = R.try_get<const PendingTransform>(SceneEntity);
        const auto extent = R.get<const ViewportExtent>(SceneEntity).Value;
        const float viewport_height = extent.height > 0 ? float(extent.height) : 1.f;
        // ScreenPixelScale: world-space size per pixel at unit distance (perspective) or absolute (ortho).
        // Sign encodes camera type: positive = perspective (shader multiplies by distance), negative = orthographic.
        const float screen_pixel_scale = ScreenPixelScale(camera.Data, viewport_height);
        Buffers->SceneViewUBO.Update(as_bytes(SceneViewUBO{
            .ViewProj = camera.Projection(extent.width == 0 || extent.height == 0 ? 1.f : float(extent.width) / float(extent.height)) * camera.View(),
            .CameraPosition = camera.Position(),
            .CameraNear = camera.NearClip(),
            .CameraFar = camera.FarClip(),
            .ViewColor = lights.ViewColor,
            .AmbientIntensity = lights.AmbientIntensity,
            .DirectionalColor = lights.DirectionalColor,
            .DirectionalIntensity = lights.DirectionalIntensity,
            .LightDirection = lights.Direction,
            .LightCount = uint32_t(Buffers->LightBuffer.UsedSize / sizeof(PunctualLight)),
            .LightSlot = Buffers->LightBuffer.Slot,
            .InteractionMode = uint32_t(interaction_mode),
            .EditElement = uint32_t(R.get<const SceneEditMode>(SceneEntity).Value),
            .IsTransforming = pending ? 1u : 0u,
            .PendingPivot = pending ? pending->Pivot : vec3{},
            .PendingTranslation = pending ? pending->P : vec3{},
            .PendingRotation = pending ? QuatToVec4(pending->R) : vec4{0, 0, 0, 1},
            .PendingScale = pending ? pending->S : vec3{1},
            .ScreenPixelScale = screen_pixel_scale,
            .FaceFirstTriSlot = Meshes->FaceFirstTriangleBuffer.Buffer.Slot,
            .BoneDeformSlot = Meshes->BoneDeformBuffer.Buffer.Slot,
            .ArmatureDeformSlot = Buffers->ArmatureDeformBuffer.Buffer.Slot,
            .MorphDeformSlot = Meshes->MorphTargetBuffer.Buffer.Slot,
            .MorphWeightsSlot = Buffers->MorphWeightBuffer.Buffer.Slot,
            .VertexClassSlot = Buffers->VertexClassBuffer.Buffer.Slot,
            .DrawDataSlot = Buffers->RenderDrawData.Slot,
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
        } else if (const auto *selection = R.try_get<const MeshSelection>(mesh_entity);
                   selection && interaction_mode == InteractionMode::Edit && HasSelectedInstance(R, mesh_entity)) {
            element = edit_mode;
            handles = selection->Handles;
            if (const auto *active_element = R.try_get<MeshActiveElement>(mesh_entity)) active_handle = active_element->Handle;
            {
                const auto vertex_handles = ConvertSelectionElement(*selection, mesh, edit_mode, Element::Vertex);
                selected_vertices.insert(vertex_handles.begin(), vertex_handles.end());
            }
            if (element == Element::Edge) {
                selected_edges.insert(selection->Handles.begin(), selection->Handles.end());
            } else if (element == Element::Face) {
                for (auto h : selection->Handles) {
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

        Meshes->UpdateElementStates(mesh, element, selected_vertices, selected_edges, active_edges, selected_faces, active_handle);
        SelectionStale = true;
    }
    if (!dirty_element_state_meshes.empty()) request(RenderRequest::Submit);
    { // Animation timeline tick
        auto &tl = R.get<AnimationTimeline>(SceneEntity);
        if (tl.Playing) {
            PlaybackFrame += ImGui::GetIO().DeltaTime * tl.Fps;
            if (PlaybackFrame > float(tl.EndFrame)) PlaybackFrame = float(tl.StartFrame);
            const int new_frame = int(std::floor(PlaybackFrame));
            if (new_frame != tl.CurrentFrame) R.patch<AnimationTimeline>(SceneEntity, [&](auto &t) { t.CurrentFrame = new_frame; });
        } else {
            PlaybackFrame = float(tl.CurrentFrame);
        }
        if (tl.CurrentFrame != LastEvaluatedFrame) {
            LastEvaluatedFrame = tl.CurrentFrame;
            const float eval_seconds = float(tl.CurrentFrame) / tl.Fps;
            for (auto [entity, anim_data, pose_state, armature_data] :
                 R.view<const ArmatureAnimationData, ArmaturePoseState, ArmatureData>().each()) {
                if (anim_data.Clips.empty() || anim_data.ActiveClipIndex >= anim_data.Clips.size()) continue;
                const auto &clip = anim_data.Clips[anim_data.ActiveClipIndex];
                const float clip_time = clip.DurationSeconds > 0 ? std::fmod(eval_seconds, clip.DurationSeconds) : 0.0f;
                for (uint32_t i = 0; i < armature_data.Bones.size(); ++i) pose_state.BonePoseLocal[i] = armature_data.Bones[i].RestLocal;
                EvaluateAnimation(clip, clip_time, pose_state.BonePoseLocal);

                auto gpu_span = Buffers->ArmatureDeformBuffer.GetMutable(pose_state.GpuDeformRange);
                ComputeDeformMatrices(armature_data, pose_state.BonePoseLocal, armature_data.ImportedSkin->InverseBindMatrices, gpu_span);
                request(RenderRequest::ReRecord);
            }

            // Evaluate morph weight animations
            for (auto [entity, morph_anim, morph_state, mi] :
                 R.view<const MorphWeightAnimationData, MorphWeightState, const MeshInstance>().each()) {
                if (morph_anim.Clips.empty() || morph_anim.ActiveClipIndex >= morph_anim.Clips.size()) continue;
                const auto &clip = morph_anim.Clips[morph_anim.ActiveClipIndex];
                const float clip_time = clip.DurationSeconds > 0 ? std::fmod(eval_seconds, clip.DurationSeconds) : 0.0f;
                // Reset to default weights from mesh-level data
                const auto &mesh = R.get<const Mesh>(mi.MeshEntity);
                const auto default_weights = Meshes->GetDefaultMorphWeights(mesh.GetStoreId());
                std::copy(default_weights.begin(), default_weights.end(), morph_state.Weights.begin());
                EvaluateMorphWeights(clip, clip_time, morph_state.Weights);
                // Write to GPU
                auto gpu_weights = Buffers->MorphWeightBuffer.GetMutable(morph_state.GpuWeightRange);
                std::copy(morph_state.Weights.begin(), morph_state.Weights.end(), gpu_weights.begin());
                request(RenderRequest::ReRecord);
            }
        }
    }

    for (auto &&[id, storage] : R.storage()) {
        if (storage.info() == entt::type_id<entt::reactive>()) storage.clear();
    }
    DestroyTracker->Storage.clear();
    R.clear<MeshGeometryDirty>();
    R.clear<LightDataDirty>();
    R.clear<LightWireframeDirty>();

    return render_request;
}

void Scene::SetVisible(entt::entity entity, bool visible) {
    const auto *mesh_instance = R.try_get<MeshInstance>(entity);
    if (!mesh_instance) return;

    const bool already_visible = R.all_of<RenderInstance>(entity);
    if ((visible && already_visible) || (!visible && !already_visible)) return;

    const auto mesh_entity = mesh_instance->MeshEntity;
    if (visible) {
        const auto buffer_index = R.get<const ModelsBuffer>(mesh_entity).Buffer.UsedSize / sizeof(WorldTransform);
        const uint32_t object_id = NextObjectId++;
        R.emplace<RenderInstance>(entity, buffer_index, object_id);
        const uint8_t initial_state = R.all_of<Selected>(entity) ? static_cast<uint8_t>(ElementStateSelected) : uint8_t{0};
        R.patch<ModelsBuffer>(mesh_entity, [&](auto &mb) {
            mb.Buffer.Insert(as_bytes(R.get<WorldTransform>(entity)), mb.Buffer.UsedSize);
            mb.ObjectIds.Insert(as_bytes(object_id), mb.ObjectIds.UsedSize);
            mb.InstanceStates.Insert(as_bytes(initial_state), mb.InstanceStates.UsedSize);
        });
    } else {
        const uint old_model_index = R.get<const RenderInstance>(entity).BufferIndex;
        R.remove<RenderInstance>(entity);
        R.patch<ModelsBuffer>(mesh_entity, [old_model_index](auto &mb) {
            mb.Buffer.Erase(old_model_index * sizeof(WorldTransform), sizeof(WorldTransform));
            mb.ObjectIds.Erase(old_model_index * sizeof(uint32_t), sizeof(uint32_t));
            mb.InstanceStates.Erase(old_model_index * sizeof(uint8_t), sizeof(uint8_t));
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
        mvk::Buffer{Buffers->Ctx, sizeof(WorldTransform), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ModelBuffer},
        mvk::Buffer{Buffers->Ctx, sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ObjectIdBuffer},
        mvk::Buffer{Buffers->Ctx, sizeof(uint8_t), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::InstanceStateBuffer}
    );
    R.emplace<MeshSelection>(
        mesh_entity,
        iota(0u, GetElementCount(mesh, R.get<const SceneEditMode>(SceneEntity).Value)) | to<std::unordered_set>()
    );
    R.emplace<MeshBuffers>(
        mesh_entity, Meshes->GetVerticesRange(mesh.GetStoreId()),
        Buffers->CreateIndices(mesh.CreateTriangleIndices(), IndexKind::Face),
        Buffers->CreateIndices(mesh.CreateEdgeIndices(), IndexKind::Edge),
        Buffers->CreateIndices(CreateVertexIndices(mesh), IndexKind::Vertex)
    );
    R.emplace<Mesh>(mesh_entity, std::move(mesh));
    return {mesh_entity, info ? AddMeshInstance(mesh_entity, *info) : entt::null};
}

entt::entity Scene::AddMeshInstance(entt::entity mesh_entity, MeshInstanceCreateInfo info) {
    const auto instance_entity = R.create();
    R.emplace<MeshInstance>(instance_entity, mesh_entity);
    R.emplace<ObjectKind>(instance_entity, ObjectType::Mesh);
    SetTransform(R, instance_entity, info.Transform);
    R.emplace<Name>(instance_entity, CreateName(R, info.Name));

    R.patch<ModelsBuffer>(mesh_entity, [](auto &mb) {
        mb.Buffer.Reserve(mb.Buffer.UsedSize + sizeof(WorldTransform));
        mb.ObjectIds.Reserve(mb.ObjectIds.UsedSize + sizeof(uint32_t));
        mb.InstanceStates.Reserve(mb.InstanceStates.UsedSize + sizeof(uint8_t));
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

entt::entity Scene::AddEmpty(ObjectCreateInfo info) {
    // Plain axes: 3 unit-length axes lines
    return CreateExtrasObject(
        {{
            .Positions = {{0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 0, -1}},
            .Edges = {{0, 1}, {2, 3}, {4, 5}},
        }},
        ObjectType::Empty, info, "Empty"
    );
}

entt::entity Scene::AddArmature(ObjectCreateInfo info) {
    const auto data_entity = R.create();
    R.emplace<ArmatureData>(data_entity);

    const auto entity = R.create();
    R.emplace<ObjectKind>(entity, ObjectType::Armature);
    R.emplace<ArmatureObject>(entity, data_entity);
    SetTransform(R, entity, info.Transform);
    R.emplace<Name>(entity, CreateName(R, info.Name.empty() ? "Armature" : info.Name));

    switch (info.Select) {
        case MeshInstanceCreateInfo::SelectBehavior::Exclusive:
            Select(entity);
            break;
        case MeshInstanceCreateInfo::SelectBehavior::Additive:
            R.emplace<Selected>(entity);
            // Fallthrough
        case MeshInstanceCreateInfo::SelectBehavior::None:
            if (R.storage<Active>().empty()) {
                R.emplace<Active>(entity);
                R.emplace_or_replace<Selected>(entity);
            }
            break;
    }
    return entity;
}

entt::entity Scene::CreateExtrasMeshEntity(ExtrasWireframe &&wireframe) {
    const auto [mesh_entity, _] = AddMesh(std::move(wireframe.Data), std::nullopt);
    R.emplace<ObjectExtrasTag>(mesh_entity);
    if (!wireframe.VertexClasses.empty()) {
        const auto range = Buffers->VertexClassBuffer.Allocate(std::span<const uint8_t>(wireframe.VertexClasses));
        R.emplace<VertexClass>(mesh_entity, range.Offset);
    }
    return mesh_entity;
}

entt::entity Scene::CreateExtrasObject(ExtrasWireframe &&wireframe, ObjectType type, ObjectCreateInfo info, const std::string &default_name) {
    const auto mesh_entity = CreateExtrasMeshEntity(std::move(wireframe));

    const auto entity = R.create();
    R.emplace<ObjectKind>(entity, type);
    R.emplace<MeshInstance>(entity, mesh_entity);
    SetTransform(R, entity, info.Transform);
    R.emplace<Name>(entity, CreateName(R, info.Name.empty() ? default_name : info.Name));

    R.patch<ModelsBuffer>(mesh_entity, [](auto &mb) {
        mb.Buffer.Reserve(mb.Buffer.UsedSize + sizeof(WorldTransform));
        mb.ObjectIds.Reserve(mb.ObjectIds.UsedSize + sizeof(uint32_t));
        mb.InstanceStates.Reserve(mb.InstanceStates.UsedSize + sizeof(uint8_t));
    });
    SetVisible(entity, true);

    switch (info.Select) {
        case MeshInstanceCreateInfo::SelectBehavior::Exclusive:
            Select(entity);
            break;
        case MeshInstanceCreateInfo::SelectBehavior::Additive:
            R.emplace<Selected>(entity);
            // Fallthrough
        case MeshInstanceCreateInfo::SelectBehavior::None:
            if (R.storage<Active>().empty()) {
                R.emplace<Active>(entity);
                R.emplace_or_replace<Selected>(entity);
            }
            break;
    }
    return entity;
}

entt::entity Scene::AddCamera(ObjectCreateInfo info) {
    CameraData cd{Perspective{.FieldOfViewRad = glm::radians(60.f), .FarClip = 100.f, .NearClip = 0.01f}};
    const auto entity = CreateExtrasObject({.Data = BuildCameraFrustumMesh(cd)}, ObjectType::Camera, info, "Camera");
    R.emplace<CameraData>(entity, cd);
    return entity;
}

entt::entity Scene::AddLight(ObjectCreateInfo info, std::optional<PunctualLight> props) {
    auto light = props.value_or(MakeDefaultLight(LightTypePoint));
    auto wireframe = BuildLightMesh(light);
    const auto entity = CreateExtrasObject(std::move(wireframe), ObjectType::Light, info, "Light");
    const uint32_t light_index = uint32_t(R.storage<LightIndex>().size());
    R.emplace<LightIndex>(entity, light_index);
    R.emplace<LightDataDirty>(entity);
    R.emplace<LightWireframeDirty>(entity);
    const auto mesh_entity = R.get<MeshInstance>(entity).MeshEntity;
    const auto &ri = R.get<const RenderInstance>(entity);
    light.TransformSlotOffset = {R.get<const ModelsBuffer>(mesh_entity).Buffer.Slot, ri.BufferIndex};
    SetLight(*Buffers, light_index, light);
    return entity;
}

std::pair<entt::entity, entt::entity> Scene::AddMesh(MeshData &&data, std::optional<MeshInstanceCreateInfo> info) {
    return AddMesh(Meshes->CreateMesh(std::move(data)), std::move(info));
}

std::pair<entt::entity, entt::entity> Scene::AddMesh(const std::filesystem::path &path, std::optional<MeshInstanceCreateInfo> info) {
    auto result = Meshes->LoadMesh(path);
    if (!result) throw std::runtime_error(result.error());

    const auto e = AddMesh(std::move(*result), std::move(info));
    R.emplace<Path>(e.first, path);
    return e;
}

std::expected<std::pair<entt::entity, entt::entity>, std::string> Scene::AddGltfScene(const std::filesystem::path &path) {
    auto loaded_scene = gltf::LoadSceneData(path);
    if (!loaded_scene) return std::unexpected{std::move(loaded_scene.error())};

    std::vector<entt::entity> mesh_entities;
    mesh_entities.reserve(loaded_scene->Meshes.size());
    // Track morph data per mesh for later component setup
    std::vector<std::optional<MorphTargetData>> mesh_morph_data;
    mesh_morph_data.reserve(loaded_scene->Meshes.size());
    entt::entity first_mesh_entity = entt::null;
    // Per-mesh: optional line entity + optional point entity
    struct ExtraPrimitiveEntities {
        entt::entity Lines{entt::null}, Points{entt::null};
    };
    std::vector<ExtraPrimitiveEntities> extra_entities_per_mesh(loaded_scene->Meshes.size());
    for (uint32_t mi = 0; mi < loaded_scene->Meshes.size(); ++mi) {
        auto &scene_mesh = loaded_scene->Meshes[mi];
        entt::entity mesh_entity = entt::null;
        if (scene_mesh.Triangles) {
            auto morph_data_copy = scene_mesh.MorphData; // Keep a copy for component setup
            auto mesh = Meshes->CreateMesh(std::move(*scene_mesh.Triangles), std::move(scene_mesh.DeformData), std::move(scene_mesh.MorphData));
            const auto [me, _] = AddMesh(std::move(mesh), std::nullopt);
            mesh_entity = me;
            R.emplace<Path>(mesh_entity, path);
            mesh_morph_data.emplace_back(std::move(morph_data_copy));
        } else {
            mesh_morph_data.emplace_back(std::nullopt);
        }
        if (first_mesh_entity == entt::null && mesh_entity != entt::null) first_mesh_entity = mesh_entity;
        mesh_entities.emplace_back(mesh_entity);

        auto create_extra = [&](std::optional<MeshData> &data) -> entt::entity {
            if (!data) return entt::null;
            auto m = Meshes->CreateMesh(std::move(*data));
            const auto [e, _] = AddMesh(std::move(m), std::nullopt);
            R.emplace<Path>(e, path);
            if (first_mesh_entity == entt::null) first_mesh_entity = e;
            return e;
        };
        extra_entities_per_mesh[mi] = {create_extra(scene_mesh.Lines), create_extra(scene_mesh.Points)};
    }

    const std::string name_prefix = path.stem().string();
    std::unordered_map<uint32_t, entt::entity> object_entities_by_node;
    object_entities_by_node.reserve(loaded_scene->Objects.size());
    std::unordered_map<uint32_t, std::vector<entt::entity>> skinned_mesh_instances_by_skin;
    skinned_mesh_instances_by_skin.reserve(loaded_scene->Skins.size());

    entt::entity first_object_entity = entt::null,
                 first_mesh_object_entity = entt::null,
                 first_root_empty_entity = entt::null,
                 first_armature_entity = entt::null;
    for (uint32_t i = 0; i < loaded_scene->Objects.size(); ++i) {
        const auto &object = loaded_scene->Objects[i];
        const auto object_name = object.Name.empty() ? std::format("{}_{}", name_prefix, i) : object.Name;
        entt::entity object_entity = entt::null;
        if (object.ObjectType == gltf::SceneObjectData::Type::Mesh &&
            object.MeshIndex &&
            *object.MeshIndex < mesh_entities.size() &&
            mesh_entities[*object.MeshIndex] != entt::null) {
            object_entity = AddMeshInstance(
                mesh_entities[*object.MeshIndex],
                {
                    .Name = object_name,
                    .Transform = object.WorldTransform,
                    .Select = MeshInstanceCreateInfo::SelectBehavior::None,
                    .Visible = true,
                }
            );
        } else if (object.ObjectType == gltf::SceneObjectData::Type::Camera &&
                   object.CameraIndex &&
                   *object.CameraIndex < loaded_scene->Cameras.size()) {
            object_entity = AddCamera({
                .Name = object_name,
                .Transform = object.WorldTransform,
                .Select = MeshInstanceCreateInfo::SelectBehavior::None,
            });
            const auto &scd = loaded_scene->Cameras[*object.CameraIndex];
            R.replace<CameraData>(object_entity, scd.Camera);
        } else if (object.ObjectType == gltf::SceneObjectData::Type::Light &&
                   object.LightIndex &&
                   *object.LightIndex < loaded_scene->Lights.size()) {
            const auto &sld = loaded_scene->Lights[*object.LightIndex];
            object_entity = AddLight({.Name = object_name, .Transform = object.WorldTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None}, sld.Light);
            R.emplace_or_replace<LightDataDirty>(object_entity);
            R.emplace_or_replace<LightWireframeDirty>(object_entity);
        } else {
            object_entity = AddEmpty(
                {
                    .Name = object_name,
                    .Transform = object.WorldTransform,
                    .Select = MeshInstanceCreateInfo::SelectBehavior::None,
                }
            );
        }
        // Create instances for non-triangle primitives (lines/points) associated with this mesh
        if (object.ObjectType == gltf::SceneObjectData::Type::Mesh &&
            object.MeshIndex &&
            *object.MeshIndex < extra_entities_per_mesh.size()) {
            const auto &extras = extra_entities_per_mesh[*object.MeshIndex];
            for (const auto extra_entity : {extras.Lines, extras.Points}) {
                if (extra_entity == entt::null) continue;
                AddMeshInstance(
                    extra_entity,
                    {
                        .Name = object_name,
                        .Transform = object.WorldTransform,
                        .Select = MeshInstanceCreateInfo::SelectBehavior::None,
                        .Visible = true,
                    }
                );
            }
        }

        object_entities_by_node[object.NodeIndex] = object_entity;
        if (object.SkinIndex && R.all_of<MeshInstance>(object_entity)) {
            // glTF node.skin is deform linkage, not a transform-parent relationship.
            skinned_mesh_instances_by_skin[*object.SkinIndex].emplace_back(object_entity);
        }
        if (first_object_entity == entt::null) first_object_entity = object_entity;
        if (first_mesh_object_entity == entt::null && object.ObjectType == gltf::SceneObjectData::Type::Mesh) {
            first_mesh_object_entity = object_entity;
        }
        if (first_root_empty_entity == entt::null && object.ObjectType == gltf::SceneObjectData::Type::Empty && !object.ParentNodeIndex) {
            first_root_empty_entity = object_entity;
        }
    }

    for (const auto &object : loaded_scene->Objects) {
        if (!object.ParentNodeIndex) continue;

        const auto child_it = object_entities_by_node.find(object.NodeIndex);
        if (child_it == object_entities_by_node.end()) continue;
        const auto parent_it = object_entities_by_node.find(*object.ParentNodeIndex);
        if (parent_it == object_entities_by_node.end()) continue;
        SetParent(R, child_it->second, parent_it->second);
    }

    std::unordered_map<uint32_t, const gltf::SceneNodeData *> scene_nodes_by_index;
    scene_nodes_by_index.reserve(loaded_scene->Nodes.size());
    for (const auto &node : loaded_scene->Nodes) scene_nodes_by_index.emplace(node.NodeIndex, &node);

    for (const auto &skin : loaded_scene->Skins) {
        const auto armature_data_entity = R.create();
        auto &armature_data = R.emplace<ArmatureData>(armature_data_entity);

        ArmatureImportedSkin imported_skin{
            .SkinIndex = skin.SkinIndex,
            .SkeletonNodeIndex = skin.SkeletonNodeIndex,
            .AnchorNodeIndex = skin.AnchorNodeIndex,
            .OrderedJointNodeIndices = {},
            .InverseBindMatrices = skin.InverseBindMatrices,
        };
        imported_skin.OrderedJointNodeIndices.reserve(skin.Joints.size());

        std::unordered_map<uint32_t, BoneId> joint_node_to_bone_id;
        joint_node_to_bone_id.reserve(skin.Joints.size());
        for (const auto &joint : skin.Joints) {
            std::optional<BoneId> parent_bone_id;
            if (joint.ParentJointNodeIndex) {
                if (const auto parent_it = joint_node_to_bone_id.find(*joint.ParentJointNodeIndex);
                    parent_it != joint_node_to_bone_id.end()) {
                    parent_bone_id = parent_it->second;
                }
            }

            const auto joint_name = joint.Name.empty() ? std::format("Joint{}", joint.JointNodeIndex) : joint.Name;
            const auto bone_id = armature_data.AddBone(joint_name, parent_bone_id, joint.RestLocal, joint.JointNodeIndex);
            joint_node_to_bone_id.emplace(joint.JointNodeIndex, bone_id);
            if (const auto object_it = object_entities_by_node.find(joint.JointNodeIndex);
                object_it != object_entities_by_node.end() &&
                R.all_of<MeshInstance>(object_it->second) &&
                !R.all_of<BoneAttachment>(object_it->second)) {
                R.emplace<BoneAttachment>(object_it->second, armature_data_entity, bone_id);
            }
            imported_skin.OrderedJointNodeIndices.emplace_back(joint.JointNodeIndex);
        }
        armature_data.FinalizeStructure();

        imported_skin.InverseBindMatrices.resize(imported_skin.OrderedJointNodeIndices.size(), I4);
        armature_data.ImportedSkin = std::move(imported_skin);
        if (!skin.AnchorNodeIndex) {
            return std::unexpected{std::format("glTF import failed for '{}': skin {} has no deterministic anchor node.", path.string(), skin.SkinIndex)};
        }

        const auto anchor_it = scene_nodes_by_index.find(*skin.AnchorNodeIndex);
        if (anchor_it == scene_nodes_by_index.end() || !anchor_it->second->InScene) {
            return std::unexpected{std::format("glTF import failed for '{}': skin {} anchor node {} is not in the imported scene.", path.string(), skin.SkinIndex, *skin.AnchorNodeIndex)};
        }

        const auto armature_entity = R.create();
        R.emplace<ObjectKind>(armature_entity, ObjectType::Armature);
        R.emplace<ArmatureObject>(armature_entity, armature_data_entity);
        SetTransform(R, armature_entity, anchor_it->second->WorldTransform);
        R.emplace<Name>(armature_entity, CreateName(R, skin.Name.empty() ? std::format("{}_Armature{}", name_prefix, skin.SkinIndex) : skin.Name));

        if (skin.ParentObjectNodeIndex) {
            if (const auto parent_it = object_entities_by_node.find(*skin.ParentObjectNodeIndex);
                parent_it != object_entities_by_node.end()) {
                SetParent(R, armature_entity, parent_it->second);
            }
        }

        if (first_armature_entity == entt::null) first_armature_entity = armature_entity;
        if (first_object_entity == entt::null) first_object_entity = armature_entity;

        if (const auto skinned_it = skinned_mesh_instances_by_skin.find(skin.SkinIndex);
            skinned_it != skinned_mesh_instances_by_skin.end()) {
            for (const auto mesh_instance_entity : skinned_it->second) {
                if (!R.valid(mesh_instance_entity) || !R.all_of<MeshInstance>(mesh_instance_entity)) continue;
                R.emplace_or_replace<ArmatureModifier>(mesh_instance_entity, armature_data_entity, armature_entity);
            }
        } else {
            return std::unexpected{std::format("glTF import failed '{}': skin {} is used but no mesh instances were emitted for skin binding.", path.string(), skin.SkinIndex)};
        }

        // Allocate pose state and GPU deform buffer for this armature
        {
            ArmaturePoseState pose_state;
            pose_state.BonePoseLocal.resize(armature_data.Bones.size());
            for (uint32_t i = 0; i < armature_data.Bones.size(); ++i) pose_state.BonePoseLocal[i] = armature_data.Bones[i].RestLocal;
            pose_state.GpuDeformRange = Buffers->ArmatureDeformBuffer.Allocate(armature_data.ImportedSkin->OrderedJointNodeIndices.size());

            // Compute initial rest-pose deform matrices
            auto gpu_span = Buffers->ArmatureDeformBuffer.GetMutable(pose_state.GpuDeformRange);
            ComputeDeformMatrices(armature_data, pose_state.BonePoseLocal, armature_data.ImportedSkin->InverseBindMatrices, gpu_span);

            R.emplace<ArmaturePoseState>(armature_data_entity, std::move(pose_state));
        }
    }

    // Resolve animation data: map glTF animation channels to bone indices
    for (auto &anim_clip : loaded_scene->Animations) {
        for (const auto &skin : loaded_scene->Skins) {
            entt::entity target_data_entity = entt::null;
            for (const auto [entity, ad] : R.view<ArmatureData>().each()) {
                if (ad.ImportedSkin && ad.ImportedSkin->SkinIndex == skin.SkinIndex) {
                    target_data_entity = entity;
                    break;
                }
            }
            if (target_data_entity == entt::null) continue;

            const auto &ad = R.get<const ArmatureData>(target_data_entity);
            AnimationClip resolved_clip{.Name = std::move(anim_clip.Name), .DurationSeconds = anim_clip.DurationSeconds, .Channels = {}};

            for (auto &ch : anim_clip.Channels) {
                if (const auto bone_id = ad.FindBoneIdByJointNodeIndex(ch.TargetNodeIndex)) {
                    if (const auto bone_index = ad.FindBoneIndex(*bone_id)) {
                        resolved_clip.Channels.emplace_back(AnimationChannel{
                            .BoneIndex = *bone_index,
                            .Target = ch.Target,
                            .Interp = ch.Interp,
                            .TimesSeconds = std::move(ch.TimesSeconds),
                            .Values = std::move(ch.Values),
                        });
                    }
                }
            }

            if (!resolved_clip.Channels.empty()) {
                if (auto *existing = R.try_get<ArmatureAnimationData>(target_data_entity)) {
                    existing->Clips.emplace_back(std::move(resolved_clip));
                } else {
                    R.emplace<ArmatureAnimationData>(target_data_entity, ArmatureAnimationData{.Clips = {std::move(resolved_clip)}});
                }
            }
        }
    }

    // Set up morph weight state for mesh instances with morph targets
    // Build a map: node_index -> mesh instance entity, for resolving weight animation channels
    std::unordered_map<uint32_t, entt::entity> morph_instance_by_node;
    for (const auto &object : loaded_scene->Objects) {
        if (object.ObjectType != gltf::SceneObjectData::Type::Mesh || !object.MeshIndex) continue;
        if (*object.MeshIndex >= mesh_morph_data.size() || !mesh_morph_data[*object.MeshIndex]) continue;
        const auto obj_it = object_entities_by_node.find(object.NodeIndex);
        if (obj_it == object_entities_by_node.end()) continue;
        const auto instance_entity = obj_it->second;
        if (!R.all_of<MeshInstance>(instance_entity)) continue;

        const auto &morph = *mesh_morph_data[*object.MeshIndex];
        if (morph.TargetCount == 0) continue;

        MorphWeightState state;
        if (object.NodeWeights) {
            // Per-node morph weight overrides (glTF node.weights) take priority over mesh.weights
            state.Weights.resize(morph.TargetCount, 0.f);
            std::copy_n(object.NodeWeights->begin(), std::min(uint32_t(object.NodeWeights->size()), morph.TargetCount), state.Weights.begin());
        } else {
            state.Weights = morph.DefaultWeights;
        }
        state.GpuWeightRange = Buffers->MorphWeightBuffer.Allocate(morph.TargetCount);
        auto gpu_weights = Buffers->MorphWeightBuffer.GetMutable(state.GpuWeightRange);
        std::copy(state.Weights.begin(), state.Weights.end(), gpu_weights.begin());
        R.emplace<MorphWeightState>(instance_entity, std::move(state));
        morph_instance_by_node[object.NodeIndex] = instance_entity;
    }

    // Resolve morph weight animation channels
    for (auto &anim_clip : loaded_scene->Animations) {
        // Group weight channels by target node
        std::unordered_map<uint32_t, std::vector<gltf::AnimationChannelData *>> weight_channels_by_node;
        for (auto &ch : anim_clip.Channels) {
            if (ch.Target == AnimationPath::Weights) weight_channels_by_node[ch.TargetNodeIndex].emplace_back(&ch);
        }
        for (auto &[node_index, channels] : weight_channels_by_node) {
            const auto inst_it = morph_instance_by_node.find(node_index);
            if (inst_it == morph_instance_by_node.end()) continue;
            const auto instance_entity = inst_it->second;

            MorphWeightClip resolved_clip{.Name = anim_clip.Name, .DurationSeconds = anim_clip.DurationSeconds, .Channels = {}};
            for (auto *ch : channels) {
                resolved_clip.Channels.emplace_back(MorphWeightChannel{
                    .Interp = ch->Interp,
                    .TimesSeconds = std::move(ch->TimesSeconds),
                    .Values = std::move(ch->Values),
                });
            }
            if (!resolved_clip.Channels.empty()) {
                if (auto *existing = R.try_get<MorphWeightAnimationData>(instance_entity)) {
                    existing->Clips.emplace_back(std::move(resolved_clip));
                } else {
                    R.emplace<MorphWeightAnimationData>(instance_entity, MorphWeightAnimationData{.Clips = {std::move(resolved_clip)}});
                }
            }
        }
    }

    { // Get timeline range from imported animation durations
        float max_dur = 0;
        for (const auto [_, anim] : R.view<const ArmatureAnimationData>().each()) {
            for (const auto &clip : anim.Clips) max_dur = std::max(max_dur, clip.DurationSeconds);
        }
        for (const auto [_, anim] : R.view<const MorphWeightAnimationData>().each()) {
            for (const auto &clip : anim.Clips) max_dur = std::max(max_dur, clip.DurationSeconds);
        }
        if (max_dur > 0) R.patch<AnimationTimeline>(SceneEntity, [&](auto &tl) { tl.EndFrame = int(std::ceil(max_dur * tl.Fps)); });
    }

    const auto selected_entity =
        first_mesh_object_entity != entt::null ? first_mesh_object_entity :
        first_armature_entity != entt::null    ? first_armature_entity :
        first_root_empty_entity != entt::null  ? first_root_empty_entity :
                                                 first_object_entity;
    if (selected_entity != entt::null) Select(selected_entity);

    return std::pair{first_mesh_entity, selected_entity};
}

entt::entity Scene::Duplicate(entt::entity e, std::optional<MeshInstanceCreateInfo> info) {
    const auto select_behavior = info ? info->Select : (R.all_of<Selected>(e) ? MeshInstanceCreateInfo::SelectBehavior::Additive : MeshInstanceCreateInfo::SelectBehavior::None);
    const ObjectCreateInfo create_info{
        .Name = info && !info->Name.empty() ? info->Name : std::format("{}_copy", GetName(R, e)),
        .Transform = info ? info->Transform : GetTransform(R, e),
        .Select = select_behavior,
    };

    if (!R.all_of<MeshInstance>(e)) {
        const auto object_type = R.all_of<ObjectKind>(e) ? R.get<const ObjectKind>(e).Value : ObjectType::Empty;
        if (object_type == ObjectType::Armature) {
            const auto copy_entity = AddArmature(create_info);
            if (const auto *src_armature = R.try_get<ArmatureObject>(e)) {
                auto &dst_data = R.get<ArmatureData>(R.get<ArmatureObject>(copy_entity).DataEntity);
                dst_data = R.get<const ArmatureData>(src_armature->DataEntity);
            }
            return copy_entity;
        }
        return AddEmpty(create_info);
    }

    // Object extras (Camera, Empty, Light) have MeshInstance but create their own wireframe mesh.
    if (R.all_of<ObjectExtrasTag>(R.get<MeshInstance>(e).MeshEntity)) {
        if (const auto *src_cd = R.try_get<CameraData>(e)) {
            const auto copy_entity = AddCamera(create_info);
            R.replace<CameraData>(copy_entity, *src_cd);
            return copy_entity;
        }
        if (R.all_of<LightIndex>(e)) {
            const auto copy_entity = AddLight(create_info, GetLight(*Buffers, R.get<const LightIndex>(e).Value));
            R.emplace_or_replace<LightDataDirty>(copy_entity);
            R.emplace_or_replace<LightWireframeDirty>(copy_entity);
            return copy_entity;
        }
        return AddEmpty(create_info);
    }

    const auto mesh_entity = R.get<MeshInstance>(e).MeshEntity;
    const auto e_new = AddMesh(
        Meshes->CloneMesh(R.get<const Mesh>(mesh_entity)),
        info.value_or(MeshInstanceCreateInfo{
            .Name = std::format("{}_copy", GetName(R, e)),
            .Transform = GetTransform(R, e),
            .Select = R.all_of<Selected>(e) ? MeshInstanceCreateInfo::SelectBehavior::Additive : MeshInstanceCreateInfo::SelectBehavior::None,
            .Visible = R.all_of<RenderInstance>(e),
        })
    );
    if (auto primitive_type = R.try_get<PrimitiveType>(mesh_entity)) R.emplace<PrimitiveType>(e_new.first, *primitive_type);
    if (const auto *armature_modifier = R.try_get<ArmatureModifier>(e)) R.emplace<ArmatureModifier>(e_new.second, *armature_modifier);
    if (const auto *bone_attachment = R.try_get<BoneAttachment>(e)) R.emplace<BoneAttachment>(e_new.second, *bone_attachment);
    return e_new.second;
}

entt::entity Scene::DuplicateLinked(entt::entity e, std::optional<MeshInstanceCreateInfo> info) {
    if (!R.all_of<MeshInstance>(e)) {
        const auto select_behavior = info ? info->Select : (R.all_of<Selected>(e) ? MeshInstanceCreateInfo::SelectBehavior::Additive : MeshInstanceCreateInfo::SelectBehavior::None);

        if (const auto *armature = R.try_get<ArmatureObject>(e)) {
            const auto e_new = R.create();
            R.emplace<Name>(e_new, !info || info->Name.empty() ? CreateName(R, std::format("{}_copy", GetName(R, e))) : CreateName(R, info->Name));
            R.emplace<ObjectKind>(e_new, ObjectType::Armature);
            R.emplace<ArmatureObject>(e_new, armature->DataEntity);
            SetTransform(R, e_new, info ? info->Transform : GetTransform(R, e));

            if (select_behavior == MeshInstanceCreateInfo::SelectBehavior::Additive) R.emplace<Selected>(e_new);
            else if (select_behavior == MeshInstanceCreateInfo::SelectBehavior::Exclusive) Select(e_new);
            else if (R.storage<Active>().empty()) {
                R.emplace<Active>(e_new);
                R.emplace_or_replace<Selected>(e_new);
            }
            return e_new;
        }

        return AddEmpty(
            {
                .Name = info && !info->Name.empty() ? info->Name : std::format("{}_copy", GetName(R, e)),
                .Transform = info ? info->Transform : GetTransform(R, e),
                .Select = select_behavior,
            }
        );
    }

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
    R.emplace<ObjectKind>(e_new, ObjectType::Mesh);
    R.patch<ModelsBuffer>(mesh_entity, [](auto &mb) {
        mb.Buffer.Reserve(mb.Buffer.UsedSize + sizeof(WorldTransform));
        mb.ObjectIds.Reserve(mb.ObjectIds.UsedSize + sizeof(uint32_t));
        mb.InstanceStates.Reserve(mb.InstanceStates.UsedSize + sizeof(uint8_t));
    });
    SetTransform(R, e_new, info ? info->Transform : GetTransform(R, e));
    SetVisible(e_new, !info || info->Visible);
    if (const auto *armature_modifier = R.try_get<ArmatureModifier>(e)) R.emplace<ArmatureModifier>(e_new, *armature_modifier);
    if (const auto *bone_attachment = R.try_get<BoneAttachment>(e)) R.emplace<BoneAttachment>(e_new, *bone_attachment);

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
    if (HasFrozenInstance(R, e)) return;

    Meshes->SetPositions(R.get<const Mesh>(e), positions);
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
    std::vector<entt::entity> armature_data_entities;
    if (const auto *armature = R.try_get<ArmatureObject>(e); armature && R.valid(armature->DataEntity)) {
        armature_data_entities.emplace_back(armature->DataEntity);
    }
    if (const auto *armature_modifier = R.try_get<ArmatureModifier>(e); armature_modifier && R.valid(armature_modifier->ArmatureDataEntity)) {
        armature_data_entities.emplace_back(armature_modifier->ArmatureDataEntity);
    }
    if (const auto *bone_attachment = R.try_get<BoneAttachment>(e); bone_attachment && R.valid(bone_attachment->ArmatureDataEntity)) {
        armature_data_entities.emplace_back(bone_attachment->ArmatureDataEntity);
    }
    std::vector<entt::entity> unique_armature_data_entities;
    unique_armature_data_entities.reserve(armature_data_entities.size());
    for (const auto data_entity : armature_data_entities) {
        if (find(unique_armature_data_entities, data_entity) == unique_armature_data_entities.end()) {
            unique_armature_data_entities.emplace_back(data_entity);
        }
    }
    armature_data_entities = std::move(unique_armature_data_entities);

    if (const auto *light_index = R.try_get<LightIndex>(e)) {
        const uint32_t remove_index = light_index->Value;
        const uint32_t last_index = uint32_t(R.storage<LightIndex>().size()) - 1u;
        if (remove_index != last_index) {
            SetLight(*Buffers, remove_index, GetLight(*Buffers, last_index));
            for (const auto [other_entity, other_light_index] : R.view<LightIndex>().each()) {
                if (other_entity != e && other_light_index.Value == last_index) {
                    R.replace<LightIndex>(other_entity, remove_index);
                    break;
                }
            }
        }
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
            if (const auto *vcr = R.try_get<VertexClass>(mesh_entity)) {
                if (const auto *mesh = R.try_get<Mesh>(mesh_entity)) {
                    Buffers->VertexClassBuffer.Release({vcr->Offset, mesh->VertexCount()});
                }
            }
            R.destroy(mesh_entity);
        }
    }
    for (const auto armature_data_entity : armature_data_entities) {
        if (!R.valid(armature_data_entity)) continue;

        const auto used_by_armature_object = any_of(
            R.view<ArmatureObject>().each(),
            [armature_data_entity](const auto &entry) { return std::get<1>(entry).DataEntity == armature_data_entity; }
        );
        const auto used_by_armature_modifier = any_of(
            R.view<ArmatureModifier>().each(),
            [armature_data_entity](const auto &entry) { return std::get<1>(entry).ArmatureDataEntity == armature_data_entity; }
        );
        const auto used_by_bone_attachment = any_of(
            R.view<BoneAttachment>().each(),
            [armature_data_entity](const auto &entry) { return std::get<1>(entry).ArmatureDataEntity == armature_data_entity; }
        );

        if (!(used_by_armature_object || used_by_armature_modifier || used_by_bone_attachment)) {
            R.destroy(armature_data_entity);
        }
    }
}

void Scene::SetInteractionMode(InteractionMode mode) {
    if (R.get<const SceneInteraction>(SceneEntity).Mode == mode) return;

    if (mode == InteractionMode::Edit && !AllSelectedAreMeshes(R)) return;

    R.patch<SceneInteraction>(SceneEntity, [mode](auto &s) { s.Mode = mode; });
    R.patch<ViewportTheme>(SceneEntity, [](auto &) {});
}

void Scene::SetEditMode(Element mode) {
    const auto current_mode = R.get<const SceneEditMode>(SceneEntity).Value;
    if (current_mode == mode) return;

    for (const auto &[e, selection, mesh] : R.view<MeshSelection, Mesh>().each()) {
        R.replace<MeshSelection>(e, ConvertSelectionElement(selection, mesh, current_mode, mode));
        R.remove<MeshActiveElement>(e);
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
    const bool show_rendered = settings.ViewportShading == ViewportShadingMode::Rendered;
    const bool show_fill = settings.ViewportShading == ViewportShadingMode::Solid || show_rendered;
    const SPT fill_pipeline = show_rendered ? SPT::PBRFill :
                                              (settings.FaceColorMode == FaceColorMode::Mesh ? SPT::Fill : SPT::DebugNormals);
    const auto primary_edit_instances = is_edit_mode ? ComputePrimaryEditInstances(R) : std::unordered_map<entt::entity, entt::entity>{};
    const bool has_pending_transform = is_edit_mode && R.all_of<PendingTransform>(SceneEntity);
    const auto edit_transform_context = is_edit_mode ? BuildEditTransformContext(R) : EditTransformContext{};

    const auto set_edit_pending_local_transform = [&](DrawData &draw, entt::entity mesh_entity) {
        if (!has_pending_transform) return;
        if (edit_transform_context.TransformInstances.contains(mesh_entity)) {
            draw.HasPendingVertexTransform = 1;
        }
    };
    std::unordered_set<entt::entity> silhouette_instances;
    if (is_edit_mode) {
        for (const auto [e, mi, ri] : R.view<const MeshInstance, const Selected, const RenderInstance>().each()) {
            if (auto it = primary_edit_instances.find(mi.MeshEntity); it == primary_edit_instances.end() || it->second != e) {
                silhouette_instances.insert(e);
            }
        }
    }

    std::unordered_set<entt::entity> excitable_mesh_entities;
    if (is_excite_mode) {
        for (const auto [e, mi, excitable] : R.view<const MeshInstance, const Excitable>().each()) {
            excitable_mesh_entities.emplace(mi.MeshEntity);
        }
    }

    // Build mesh_entity -> deform slots mapping for skinned meshes (edit mode shows rest pose)
    const auto mesh_deform_slots = is_edit_mode ? std::unordered_map<entt::entity, DeformSlots>{} : BuildDeformSlots(R, *Meshes);
    const auto get_deform_slots = [&](entt::entity mesh_entity) -> DeformSlots {
        if (auto it = mesh_deform_slots.find(mesh_entity); it != mesh_deform_slots.end()) return it->second;
        return {};
    };

    const bool render_silhouette = !R.view<Selected>().empty() &&
        (interaction_mode == InteractionMode::Object || !silhouette_instances.empty());

    DrawListBuilder draw_list;
    DrawBatchInfo fill_batch{}, line_batch{}, point_batch{};
    DrawBatchInfo extras_line_batch{};
    DrawBatchInfo silhouette_batch{};
    DrawBatchInfo overlay_face_normals_batch{}, overlay_vertex_normals_batch{}, overlay_bbox_batch{};

    if (render_silhouette) {
        silhouette_batch = draw_list.BeginBatch();
        auto append_silhouette = [&](entt::entity e) {
            const auto mesh_entity = R.get<MeshInstance>(e).MeshEntity;
            const auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
            const auto &models = R.get<ModelsBuffer>(mesh_entity);
            const auto deform = get_deform_slots(mesh_entity);
            auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models, deform.BoneDeformOffset, deform.ArmatureDeformOffset, deform.MorphDeformOffset, deform.MorphTargetCount);
            draw.ObjectIdSlot = models.ObjectIds.Slot;
            set_edit_pending_local_transform(draw, mesh_entity);
            const auto draws_before = draw_list.Draws.size();
            AppendDraw(draw_list, silhouette_batch, mesh_buffers.FaceIndices, models, draw, R.get<RenderInstance>(e).BufferIndex);
            PatchMorphWeights(draw_list, draws_before, deform);
        };
        if (is_edit_mode) {
            for (const auto e : silhouette_instances) append_silhouette(e);
        } else {
            for (const auto e : R.view<Selected, RenderInstance>()) append_silhouette(e);
        }
    }

    if (show_fill) {
        fill_batch = draw_list.BeginBatch();
        for (auto [entity, mesh_buffers, models, mesh] : R.view<MeshBuffers, ModelsBuffer, Mesh>().each()) {
            if (mesh.FaceCount() == 0) continue;
            const auto deform = get_deform_slots(entity);
            auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models, deform.BoneDeformOffset, deform.ArmatureDeformOffset, deform.MorphDeformOffset, deform.MorphTargetCount);
            const auto face_id_buffer = Meshes->GetFaceIdRange(mesh.GetStoreId());
            const auto face_first_tri = Meshes->GetFaceFirstTriRange(mesh.GetStoreId());
            const auto face_state_buffer = Meshes->GetFaceStateRange(mesh.GetStoreId());
            draw.ObjectIdSlot = face_id_buffer.Slot;
            draw.FaceIdOffset = face_id_buffer.Offset;
            draw.FaceFirstTriOffset = settings.SmoothShading ? InvalidOffset : face_first_tri.Offset;
            set_edit_pending_local_transform(draw, entity);
            if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                // Draw primary with element state first, then all without (depth LESS won't overwrite)
                draw.ElementStateSlotOffset = face_state_buffer;
                const auto db1 = draw_list.Draws.size();
                AppendDraw(draw_list, fill_batch, mesh_buffers.FaceIndices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
                PatchMorphWeights(draw_list, db1, deform);
                draw.ElementStateSlotOffset = {};
                const auto db2 = draw_list.Draws.size();
                AppendDraw(draw_list, fill_batch, mesh_buffers.FaceIndices, models, draw);
                PatchMorphWeights(draw_list, db2, deform);
            } else {
                draw.ElementStateSlotOffset = face_state_buffer;
                const auto db = draw_list.Draws.size();
                AppendDraw(draw_list, fill_batch, mesh_buffers.FaceIndices, models, draw);
                PatchMorphWeights(draw_list, db, deform);
            }
        }
    }

    {
        line_batch = draw_list.BeginBatch();
        for (auto [entity, mesh_buffers, models, mesh] : R.view<MeshBuffers, ModelsBuffer, Mesh>().each()) {
            if (R.all_of<ObjectExtrasTag>(entity)) continue;
            if (mesh_buffers.EdgeIndices.Count == 0) continue;
            const bool is_line_mesh = mesh.FaceCount() == 0 && mesh.EdgeCount() > 0;
            if (!is_line_mesh && !is_edit_mode && !is_excite_mode) continue;
            const auto deform = get_deform_slots(entity);
            auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.EdgeIndices, models, deform.BoneDeformOffset, deform.ArmatureDeformOffset, deform.MorphDeformOffset, deform.MorphTargetCount);
            const auto edge_state_buffer = Meshes->GetEdgeStateRange(mesh.GetStoreId());
            draw.ElementStateSlotOffset = edge_state_buffer;
            set_edit_pending_local_transform(draw, entity);
            const auto db = draw_list.Draws.size();
            if (is_line_mesh) {
                AppendDraw(draw_list, line_batch, mesh_buffers.EdgeIndices, models, draw);
            } else if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                AppendDraw(draw_list, line_batch, mesh_buffers.EdgeIndices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
            } else if (excitable_mesh_entities.contains(entity)) {
                AppendDraw(draw_list, line_batch, mesh_buffers.EdgeIndices, models, draw);
            }
            PatchMorphWeights(draw_list, db, deform);
        }
    }

    AppendExtrasDraw(R, draw_list, extras_line_batch, [](auto &, const auto &) {});

    {
        point_batch = draw_list.BeginBatch();
        for (auto [entity, mesh_buffers, models] : R.view<MeshBuffers, ModelsBuffer>().each()) {
            const auto *mesh = R.try_get<Mesh>(entity);
            const bool is_point_mesh = mesh && mesh->FaceCount() == 0 && mesh->EdgeCount() == 0;
            if (!is_point_mesh && !((is_edit_mode && edit_mode == Element::Vertex) || is_excite_mode)) continue;
            const auto deform = get_deform_slots(entity);
            auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.VertexIndices, models, deform.BoneDeformOffset, deform.ArmatureDeformOffset, deform.MorphDeformOffset, deform.MorphTargetCount);
            draw.ElementStateSlotOffset = {Meshes->GetVertexStateSlot(), mesh_buffers.Vertices.Offset};
            set_edit_pending_local_transform(draw, entity);
            const auto db = draw_list.Draws.size();
            if (is_point_mesh) {
                AppendDraw(draw_list, point_batch, mesh_buffers.VertexIndices, models, draw);
            } else if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                AppendDraw(draw_list, point_batch, mesh_buffers.VertexIndices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
            } else if (excitable_mesh_entities.contains(entity)) {
                AppendDraw(draw_list, point_batch, mesh_buffers.VertexIndices, models, draw);
            }
            PatchMorphWeights(draw_list, db, deform);
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
    const uint32_t transform_vertex_state_slot = is_edit_mode ? Meshes->GetVertexStateSlot() : InvalidSlot;
    auto record_draw_batch = [&](const PipelineRenderer &renderer, SPT spt, const DrawBatchInfo &batch) {
        if (batch.DrawCount == 0) return;
        const auto &pipeline = renderer.Bind(cb, spt);
        const DrawPassPushConstants pc{batch.DrawDataSlotOffset, transform_vertex_state_slot};
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
        if (show_fill) record_draw_batch(main.Renderer, fill_pipeline, fill_batch);
        // Wireframe edges (always recorded — batch is empty when nothing qualifies)
        record_draw_batch(main.Renderer, SPT::Line, line_batch);
        // Vertex points (always recorded — batch is empty when nothing qualifies)
        record_draw_batch(main.Renderer, SPT::Point, point_batch);
        // Object extras (cameras, lights, empties)
        record_draw_batch(main.Renderer, SPT::ObjectExtrasLine, extras_line_batch);
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
        const SilhouetteEdgeColorPushConstants pc{
            TransformGizmo::IsUsing() && interaction_mode == InteractionMode::Object, SelectionHandles->ObjectIdSampler, active_object_id
        };
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
    Buffers->Ctx.RecordDeferredCopies(*TransferCommandBuffer);
    TransferCommandBuffer->end();
}
#endif

void Scene::RenderSelectionPass(vk::Semaphore signal_semaphore) {
    const Timer timer{"RenderSelectionPass"};
    const auto primary_edit_instances = R.get<const SceneInteraction>(SceneEntity).Mode == InteractionMode::Edit ?
        ComputePrimaryEditInstances(R) :
        std::unordered_map<entt::entity, entt::entity>{};
    const bool is_edit_mode = R.get<const SceneInteraction>(SceneEntity).Mode == InteractionMode::Edit;
    const auto selection_deform_slots = is_edit_mode ? std::unordered_map<entt::entity, DeformSlots>{} : BuildDeformSlots(R, *Meshes);

    // Object selection never uses depth testing - we want all visible pixels regardless of occlusion.
    // Build separate batches per topology since each requires a different pipeline primitive topology.
    RenderSelectionPassWith(
        false,
        [&](DrawListBuilder &draw_list) -> std::vector<SelectionDrawInfo> {
            auto append_topology_batch = [&](auto filter) {
                auto batch = draw_list.BeginBatch();
                for (auto [mesh_entity, mesh_buffers, models, mesh] : R.view<MeshBuffers, ModelsBuffer, Mesh>().each()) {
                    if (R.all_of<ObjectExtrasTag>(mesh_entity)) continue;
                    const auto *indices = filter(mesh, mesh_buffers);
                    if (!indices || indices->Count == 0) continue;
                    const auto deform_it = selection_deform_slots.find(mesh_entity);
                    const auto has_deform = deform_it != selection_deform_slots.end();
                    const auto bone_deform = has_deform ? deform_it->second.BoneDeformOffset : InvalidOffset;
                    const auto armature_deform = has_deform ? deform_it->second.ArmatureDeformOffset : InvalidOffset;
                    const auto morph_deform = has_deform ? deform_it->second.MorphDeformOffset : InvalidOffset;
                    const auto morph_target_count = has_deform ? deform_it->second.MorphTargetCount : 0u;
                    auto draw = MakeDrawData(mesh_buffers.Vertices, *indices, models, bone_deform, armature_deform, morph_deform, morph_target_count);
                    draw.ObjectIdSlot = models.ObjectIds.Slot;
                    const auto db = draw_list.Draws.size();
                    if (auto it = primary_edit_instances.find(mesh_entity); it != primary_edit_instances.end()) {
                        AppendDraw(draw_list, batch, *indices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
                    } else {
                        AppendDraw(draw_list, batch, *indices, models, draw);
                    }
                    if (has_deform) PatchMorphWeights(draw_list, db, deform_it->second);
                }
                return batch;
            };

            auto tri_batch = append_topology_batch([](const Mesh &m, const MeshBuffers &b) -> const SlottedRange * {
                return m.FaceCount() > 0 ? &b.FaceIndices : nullptr;
            });
            auto line_batch = append_topology_batch([](const Mesh &m, const MeshBuffers &b) -> const SlottedRange * {
                return m.FaceCount() == 0 && m.EdgeCount() > 0 ? &b.EdgeIndices : nullptr;
            });
            auto point_batch = append_topology_batch([](const Mesh &m, const MeshBuffers &b) -> const SlottedRange * {
                return m.FaceCount() == 0 && m.EdgeCount() == 0 ? &b.VertexIndices : nullptr;
            });

            DrawBatchInfo extras_batch;
            AppendExtrasDraw(R, draw_list, extras_batch, [](auto &draw, const auto &models) {
                draw.ObjectIdSlot = models.ObjectIds.Slot;
            });

            return {
                {SPT::SelectionFragmentTriangles, tri_batch},
                {SPT::SelectionFragmentLines, line_batch},
                {SPT::SelectionFragmentPoints, point_batch},
                {SPT::SelectionObjectExtrasLines, extras_batch},
            };
        },
        signal_semaphore
    );

    SelectionStale = false;
}

void Scene::RenderSelectionPassWith(bool render_depth, const SelectionBuildFn &build_fn, vk::Semaphore signal_semaphore, bool render_silhouette) {
    const Timer timer{"RenderSelectionPassWith"};
    DrawListBuilder draw_list;
    DrawBatchInfo silhouette_batch{};
    if (render_depth) {
        silhouette_batch = draw_list.BeginBatch();
        if (render_silhouette) {
            for (const auto e : R.view<Selected>()) {
                const auto *mesh_instance = R.try_get<MeshInstance>(e);
                if (!mesh_instance) continue;
                const auto mesh_entity = mesh_instance->MeshEntity;
                const auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
                const auto &models = R.get<ModelsBuffer>(mesh_entity);
                if (const auto model_index = GetModelBufferIndex(R, e)) {
                    auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models);
                    draw.ObjectIdSlot = models.ObjectIds.Slot;
                    AppendDraw(draw_list, silhouette_batch, mesh_buffers.FaceIndices, models, draw, *model_index);
                }
            }
        }
    }
    const auto selection_draws = build_fn(draw_list);

    if (!draw_list.Draws.empty()) Buffers->SelectionDrawData.Update(as_bytes(draw_list.Draws));
    if (!draw_list.IndirectCommands.empty()) Buffers->SelectionIndirect.Update(as_bytes(draw_list.IndirectCommands));
    Buffers->EnsureIdentityIndexBuffer(draw_list.MaxIndexCount);
    if (auto descriptor_updates = Buffers->Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
        Vk.Device.updateDescriptorSets(std::move(descriptor_updates), {});
        Buffers->Ctx.ClearDeferredDescriptorUpdates();
    }
    // Update DrawDataSlot to point to selection draw data for this pass.
    Buffers->SceneViewUBO.Update(as_bytes(Buffers->SelectionDrawData.Slot), offsetof(SceneViewUBO, DrawDataSlot));

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
            batch.DrawDataSlotOffset,
            InvalidSlot,
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
    for (const auto &selection_draw : selection_draws) {
        record_draw_batch(selection.Renderer, selection_draw.Pipeline, selection_draw.Batch);
    }
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
    const bool xray_selection = SelectionXRay;
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
            if (element == Element::Face) {
                const auto face_id_buffer = Meshes->GetFaceIdRange(mesh.GetStoreId());
                draw.ObjectIdSlot = face_id_buffer.Slot;
                draw.FaceIdOffset = face_id_buffer.Offset;
            } else {
                draw.ObjectIdSlot = InvalidSlot;
                draw.FaceIdOffset = 0;
            }
            draw.VertexCountOrHeadImageSlot = 0;
            draw.ElementIdOffset = r.Offset;
            if (auto it = primary_edit_instances.find(r.MeshEntity); it != primary_edit_instances.end()) {
                AppendDraw(draw_list, batch, indices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
            } else { // todo can we guarantee only selected instances are rendered here and thus we can drop this check?
                AppendDraw(draw_list, batch, indices, models, draw);
            }
        }
        return std::vector{SelectionDrawInfo{selection_pipeline(element), batch}}; }, signal_semaphore, element != Element::Face);

    // Edit selection pass overwrites the shared head image used for object selection.
    SelectionStale = true;
}

std::vector<std::vector<uint32_t>> Scene::RunBoxSelectElements(std::span<const ElementRange> ranges, Element element, std::pair<uvec2, uvec2> box_px) {
    if (ranges.empty()) return {};

    std::vector<std::vector<uint32_t>> results(ranges.size());
    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return results;

    const Timer timer{"RunBoxSelectElements"};
    const auto element_count = fold_left(
        ranges, uint32_t{0},
        [](uint32_t total, const auto &range) { return std::max(total, range.Offset + range.Count); }
    );
    if (element_count == 0) return results;

    const uint32_t bitset_words = (element_count + 31) / 32;
    if (bitset_words > BoxSelectZeroBits.size()) return results;

    RenderEditSelectionPass(ranges, element, *SelectionReadySemaphore);

    // Element occlusion semantics are determined by RenderEditSelectionPass depth state.
    // Here we always accumulate all surviving fragments to avoid nondeterministic "one per pixel"
    // drops for overlapping visible vertices/edges.
    DispatchBoxSelect(box_min, box_max, element_count, *SelectionReadySemaphore);

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

std::optional<std::pair<entt::entity, uint32_t>> Scene::RunElementPickFromRanges(std::span<const ElementRange> ranges, Element element, uvec2 mouse_px) {
    if (ranges.empty() || element == Element::None) return {};
    const auto element_count = fold_left(
        ranges, uint32_t{0},
        [](uint32_t total, const auto &range) { return std::max(total, range.Offset + range.Count); }
    );
    if (element_count == 0) return {};

    const Timer timer{"RunElementPick"};
    RenderEditSelectionPass(ranges, element, *SelectionReadySemaphore);
    if (const auto index = FindNearestPickedElement(
            *Buffers, Pipelines->ElementPick, *ClickCommandBuffer,
            Vk.Queue, *OneShotFence, Vk.Device,
            SelectionHandles->HeadImage, Buffers->SelectionNodeBuffer.Slot, SelectionHandles->ElementPickCandidates,
            mouse_px, element_count, element,
            *SelectionReadySemaphore
        )) {
        for (const auto &range : ranges) {
            if (*index < range.Offset || *index >= range.Offset + range.Count) continue;
            return std::pair{range.MeshEntity, *index - range.Offset};
        }
    }
    return {};
}

std::optional<uint32_t> Scene::RunExcitableVertexPick(entt::entity instance_entity, uvec2 mouse_px) {
    if (!R.all_of<Excitable>(instance_entity)) return {};
    const auto *mesh_instance = R.try_get<MeshInstance>(instance_entity);
    if (!mesh_instance) return {};

    const Timer timer{"RunExcitableVertexPick"};
    const auto mesh_entity = mesh_instance->MeshEntity;
    const auto &mesh = R.get<Mesh>(mesh_entity);
    const uint32_t vertex_count = mesh.VertexCount();
    if (vertex_count == 0) return {};

    const auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
    const auto &models = R.get<ModelsBuffer>(mesh_entity);
    const auto model_index = R.get<RenderInstance>(instance_entity).BufferIndex;
    RenderSelectionPassWith(true, [&](DrawListBuilder &draw_list) {
        auto batch = draw_list.BeginBatch();
        auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.VertexIndices, models);
        draw.VertexCountOrHeadImageSlot = 0;
        draw.ElementStateSlotOffset = {Meshes->GetVertexStateSlot(), mesh_buffers.Vertices.Offset};
        AppendDraw(draw_list, batch, mesh_buffers.VertexIndices, models, draw, model_index);
        return std::vector{SelectionDrawInfo{SPT::SelectionElementVertex, batch}}; }, *SelectionReadySemaphore);
    SelectionStale = true;

    return FindNearestPickedElement(
        *Buffers, Pipelines->ElementPick, *ClickCommandBuffer,
        Vk.Queue, *OneShotFence, Vk.Device,
        SelectionHandles->HeadImage, Buffers->SelectionNodeBuffer.Slot, SelectionHandles->ElementPickCandidates,
        mouse_px, vertex_count, Element::Vertex,
        *SelectionReadySemaphore
    );
}

// Returns unique object-hit entities sorted by (distance, depth, object id).
std::vector<entt::entity> Scene::RunObjectPick(uvec2 mouse_px, uint32_t radius_px) {
    if (NextObjectId <= 1) return {}; // No objects have been assigned IDs yet
    const uint32_t max_object_id = std::min(NextObjectId - 1, SceneBuffers::MaxSelectableObjects);
    if (max_object_id == 0) return {};

    const bool selection_rendered = SelectionStale;
    if (selection_rendered) RenderSelectionPass(*SelectionReadySemaphore);

    const Timer timer{"RunObjectPick"};
    // ObjectPickKeyBuffer is persistent across clicks: high 8 bits of each packed key store
    // a per-click epoch tag. We therefore avoid clearing all keys every click and only do a
    // full reset when the 8-bit epoch wraps; stale keys are filtered out by epoch on readback.
    if (ObjectPickEpochTag == 0) {
        ResetObjectPickKeys(*Buffers);
        ObjectPickEpochTag = 255;
    }
    const uint32_t epoch_inv = ObjectPickEpochTag--;

    auto cb = *ClickCommandBuffer;
    RunSelectionCompute(
        cb, Vk.Queue, *OneShotFence, Vk.Device, Pipelines->ObjectPick,
        ObjectPickPushConstants{
            .TargetPx = mouse_px,
            .Radius = radius_px,
            .MaxId = max_object_id,
            .EpochInv = epoch_inv,
            .HeadImageIndex = SelectionHandles->HeadImage,
            .SelectionNodesIndex = Buffers->SelectionNodeBuffer.Slot,
            .BestKeyIndex = SelectionHandles->ObjectPickKey,
            .SeenBitsIndex = SelectionHandles->SelectionBitset,
        },
        [](vk::CommandBuffer dispatch_cb) { dispatch_cb.dispatch(1, 1, 1); }, // Single workgroup; threads cooperatively scan the radius.
        selection_rendered ? *SelectionReadySemaphore : vk::Semaphore{}
    );

    std::unordered_map<uint32_t, entt::entity> object_id_to_entity;
    for (const auto [e, ri] : R.view<RenderInstance>().each()) {
        if (ri.ObjectId > 0 && ri.ObjectId <= max_object_id) object_id_to_entity[ri.ObjectId] = e;
    }

    struct SortedHit {
        uint32_t Key24;
        uint32_t Id;
        auto operator<=>(const SortedHit &) const = default;
    };

    const auto *bits = reinterpret_cast<const uint32_t *>(Buffers->BoxSelectBitsetBuffer.GetData().data());
    const auto *keys = reinterpret_cast<const uint32_t *>(Buffers->ObjectPickKeyBuffer.GetData().data());
    std::vector<SortedHit> hits;
    for (uint32_t object_id = 1; object_id <= max_object_id; ++object_id) {
        const uint32_t idx = object_id - 1;
        if ((bits[idx / 32] & (1u << (idx % 32))) == 0) continue;
        if (!object_id_to_entity.contains(object_id)) continue;
        const uint32_t packed_key = keys[idx];
        if ((packed_key >> 24) != epoch_inv) continue;
        hits.emplace_back(SortedHit{packed_key & 0x00ffffffu, object_id});
    }
    std::ranges::sort(hits);

    std::vector<entt::entity> entities;
    entities.reserve(hits.size());
    for (const auto &hit : hits) {
        entities.emplace_back(object_id_to_entity.at(hit.Id));
    }
    return entities;
}

void Scene::DispatchBoxSelect(uvec2 box_min, uvec2 box_max, uint32_t max_id, vk::Semaphore wait_semaphore) {
    const uint32_t bitset_words = (max_id + 31) / 32;
    const std::span<const uint32_t> zero_bits{BoxSelectZeroBits.data(), bitset_words};
    Buffers->BoxSelectBitsetBuffer.Write(std::as_bytes(zero_bits));

    const auto group_counts = glm::max((box_max - box_min + 15u) / 16u, uvec2{1, 1});
    RunSelectionCompute(
        *ClickCommandBuffer, Vk.Queue, *OneShotFence, Vk.Device, Pipelines->BoxSelect,
        BoxSelectPushConstants{
            .BoxMin = box_min,
            .BoxMax = box_max,
            .MaxId = max_id,
            .HeadImageIndex = SelectionHandles->HeadImage,
            .SelectionNodesIndex = Buffers->SelectionNodeBuffer.Slot,
            .BoxResultIndex = SelectionHandles->SelectionBitset,
        },
        [group_counts](vk::CommandBuffer dispatch_cb) { dispatch_cb.dispatch(group_counts.x, group_counts.y, 1); },
        wait_semaphore
    );
}

std::vector<entt::entity> Scene::RunBoxSelect(std::pair<uvec2, uvec2> box_px) {
    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return {};
    if (NextObjectId <= 1) return {}; // No objects have been assigned IDs yet

    const uint32_t max_object_id = std::min(NextObjectId - 1, SceneBuffers::MaxSelectableObjects);

    const Timer timer{"RunBoxSelect"};
    const bool selection_rendered = SelectionStale;
    if (selection_rendered) RenderSelectionPass(*SelectionReadySemaphore);
    DispatchBoxSelect(box_min, box_max, max_object_id, selection_rendered ? *SelectionReadySemaphore : vk::Semaphore{});

    // Build ObjectId -> entity map for lookup
    std::unordered_map<uint32_t, entt::entity> object_id_to_entity;
    for (const auto [e, ri] : R.view<RenderInstance>().each()) object_id_to_entity[ri.ObjectId] = e;

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
    const bool has_frozen_selected = R.view<Selected, Frozen>().begin() != R.view<Selected, Frozen>().end();
    const bool edit_transform_locked = interaction_mode == InteractionMode::Edit &&
        any_of(GetSelectedMeshEntities(R), [&](entt::entity mesh_entity) { return HasFrozenInstance(R, mesh_entity); });
    const bool transform_shortcuts_enabled = !edit_transform_locked;
    const bool scale_shortcut_enabled = transform_shortcuts_enabled && !has_frozen_selected;
    // Handle keyboard input.
    if (IsWindowFocused()) {
        if (IsKeyPressed(ImGuiKey_Space, false)) R.patch<AnimationTimeline>(SceneEntity, [](auto &tl) { tl.Playing = !tl.Playing; });
        if (IsKeyPressed(ImGuiKey_Z, false) && !GetIO().KeyCtrl && !GetIO().KeyShift && !GetIO().KeyAlt && !GetIO().KeySuper) {
            R.patch<SceneSettings>(SceneEntity, [](auto &settings) {
                settings.ViewportShading = settings.ViewportShading == ViewportShadingMode::Solid ?
                    ViewportShadingMode::Rendered :
                    ViewportShadingMode::Solid;
            });
        }
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
        if (IsKeyPressed(ImGuiKey_E, false) && GetIO().KeyCtrl && GetIO().KeyShift) {
            AddEmpty({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
            StartScreenTransform = TransformGizmo::TransformType::Translate;
        } else if (IsKeyPressed(ImGuiKey_A, false) && GetIO().KeyCtrl && GetIO().KeyShift) {
            AddArmature({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
            StartScreenTransform = TransformGizmo::TransformType::Translate;
        } else if (IsKeyPressed(ImGuiKey_C, false) && GetIO().KeyCtrl && GetIO().KeyShift) {
            AddCamera({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
            StartScreenTransform = TransformGizmo::TransformType::Translate;
        } else if (IsKeyPressed(ImGuiKey_L, false) && GetIO().KeyCtrl && GetIO().KeyShift) {
            AddLight({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
            StartScreenTransform = TransformGizmo::TransformType::Translate;
        }
        if (!R.storage<Selected>().empty()) {
            if (IsKeyPressed(ImGuiKey_D, false) && GetIO().KeyShift) Duplicate();
            else if (IsKeyPressed(ImGuiKey_D, false) && GetIO().KeyAlt) DuplicateLinked();
            else if (IsKeyPressed(ImGuiKey_Delete, false) || IsKeyPressed(ImGuiKey_Backspace, false)) Delete();
            else if (IsKeyPressed(ImGuiKey_G, false) && transform_shortcuts_enabled) {
                // Start transform gizmo in both Object and Edit modes.
                // In Edit mode, shader applies transform to selected vertices.
                // In Object mode, shader applies transform to selected instances.
                StartScreenTransform = TransformGizmo::TransformType::Translate;
            } else if (IsKeyPressed(ImGuiKey_R, false) && transform_shortcuts_enabled) StartScreenTransform = TransformGizmo::TransformType::Rotate;
            else if (IsKeyPressed(ImGuiKey_S, false) && scale_shortcut_enabled) StartScreenTransform = TransformGizmo::TransformType::Scale;
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

    // Handle mouse input.
    if (!IsMouseDown(ImGuiMouseButton_Left)) R.clear<ExcitedVertex>();

    if (TransformGizmo::IsUsing()) {
        // TransformGizmo overrides this mouse cursor during some actions - this is a default.
        SetMouseCursor(ImGuiMouseCursor_ResizeAll);
        WrapMousePos(GetCurrentWindowRead()->InnerClipRect, AccumulatedWrapMouseDelta);
    } else {
        AccumulatedWrapMouseDelta = {0, 0};
    }
    if (!IsWindowHovered() && !BoxSelectStart) return;

    // Mouse wheel for camera rotation, Cmd+wheel to zoom.
    const auto &io = GetIO();
    if (const vec2 wheel{io.MouseWheelH, io.MouseWheel}; wheel != vec2{0, 0}) {
        // Exit "look through" camera view on any orbit/zoom interaction.
        ExitLookThroughCamera();
        if (io.KeyCtrl || io.KeySuper) {
            R.patch<ViewCamera>(SceneEntity, [&](auto &camera) { camera.SetTargetDistance(std::max(camera.Distance * (1 - wheel.y / 16.f), 0.01f)); });
        } else {
            R.patch<ViewCamera>(SceneEntity, [&](auto &camera) { camera.SetTargetYawPitch(camera.YawPitch + wheel * 0.15f); });
        }
    }
    if (TransformGizmo::IsUsing() || OrientationGizmo::IsActive() || TransformModePillsHovered) return;

    const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
    if (SelectionMode == SelectionMode::Box && (interaction_mode == InteractionMode::Edit || interaction_mode == InteractionMode::Object)) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            BoxSelectStart = BoxSelectEnd = ToGlm(GetMousePos());
        } else if (IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart) {
            BoxSelectEnd = ToGlm(GetMousePos());
            if (const auto box_px = ComputeBoxSelectPixels(*BoxSelectStart, *BoxSelectEnd, ToGlm(GetCursorScreenPos()), extent); box_px) {
                const bool is_additive = IsKeyDown(ImGuiMod_Shift);
                if (interaction_mode == InteractionMode::Edit) {
                    Timer timer{"BoxSelectElements (all)"};

                    const auto selected_mesh_entities = GetSelectedMeshEntities(R);

                    std::vector<ElementRange> ranges;
                    uint32_t offset = 0;
                    for (const auto mesh_entity : selected_mesh_entities) {
                        if (!is_additive && !R.get<MeshSelection>(mesh_entity).Handles.empty()) {
                            R.replace<MeshSelection>(mesh_entity, MeshSelection{});
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
    const uvec2 mouse_px{uint32_t(mouse_pos_rel.x), uint32_t(extent.height - mouse_pos_rel.y)};

    if (interaction_mode == InteractionMode::Excite) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            if (const auto hit_entities = RunObjectPick(mouse_px); !hit_entities.empty()) {
                if (const auto hit_entity = hit_entities.front(); R.all_of<Excitable>(hit_entity)) {
                    if (const auto vertex = RunExcitableVertexPick(hit_entity, mouse_px)) {
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

    if (interaction_mode == InteractionMode::Edit) {
        const bool toggle = IsKeyDown(ImGuiMod_Shift) || IsKeyDown(ImGuiMod_Ctrl) || IsKeyDown(ImGuiMod_Super);
        std::vector<ElementRange> ranges;
        uint32_t offset = 0;
        std::unordered_set<entt::entity> seen_meshes;
        for (const auto [_, mesh_instance] : R.view<const MeshInstance, const Selected>().each()) {
            if (!seen_meshes.emplace(mesh_instance.MeshEntity).second) continue;
            const uint32_t count = GetElementCount(R.get<Mesh>(mesh_instance.MeshEntity), edit_mode);
            if (count == 0) continue;
            ranges.emplace_back(ElementRange{mesh_instance.MeshEntity, offset, count});
            offset += count;
        }
        if (!toggle) {
            for (const auto mesh_entity : seen_meshes) {
                if (!R.get<MeshSelection>(mesh_entity).Handles.empty()) {
                    R.replace<MeshSelection>(mesh_entity, MeshSelection{});
                }
            }
        }
        if (const auto hit = RunElementPickFromRanges(ranges, edit_mode, mouse_px)) {
            const auto [mesh_entity, element_index] = *hit;
            const auto *current_active = R.try_get<MeshActiveElement>(mesh_entity);
            const bool is_active = current_active && current_active->Handle == element_index;
            R.patch<MeshSelection>(mesh_entity, [&](auto &selection) {
                if (!toggle) selection = {};
                if (toggle && selection.Handles.contains(element_index)) {
                    selection.Handles.erase(element_index);
                } else {
                    selection.Handles.emplace(element_index);
                }
            });
            if (toggle && is_active) {
                R.remove<MeshActiveElement>(mesh_entity);
            } else {
                R.emplace_or_replace<MeshActiveElement>(mesh_entity, element_index);
            }
        }
    } else if (interaction_mode == InteractionMode::Object) {
        const auto hit_entities = RunObjectPick(mouse_px, ObjectSelectRadiusPx);

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

void Scene::Render(vk::Fence viewportConsumerFence) {
    auto &dl = *GetWindowDrawList();
    // Split draw list into two channels: one for the viewport texture (background) and one for overlay visuals (foreground).
    // This allows doing overlay interact+draw in one pass, applying any changes to the current frame's viewport render.
    // Channel 1: overlay, Channel 0: viewport
    dl.ChannelsSplit(2);
    dl.ChannelsSetCurrent(1);
    RenderOverlay();

    dl.ChannelsSetCurrent(0);
    if (SubmitViewport(viewportConsumerFence)) {
        // Recreate the ImGui texture wrapper for the new resolve image.
        ViewportTexture = std::make_unique<mvk::ImGuiTexture>(Vk.Device, *Pipelines->Main.Resources->ResolveImage.View, vec2{0, 1}, vec2{1, 0});
    }
    if (ViewportTexture) {
        const auto p = std::bit_cast<ImVec2>(ToGlm(GetCursorScreenPos()));
        const auto extent = R.get<const ViewportExtent>(SceneEntity).Value;
        const auto &t = *ViewportTexture;
        dl.AddImage(ImTextureID(VkDescriptorSet(t.DescriptorSet)), p, {p.x + float(extent.width), p.y + float(extent.height)}, {t.Uv0.x, t.Uv0.y}, {t.Uv1.x, t.Uv1.y});
    }
    dl.ChannelsMerge();
}

bool Scene::SubmitViewport(vk::Fence viewportConsumerFence) {
    auto &extent = R.get<ViewportExtent>(SceneEntity).Value;
    const auto content_region = ToGlm(GetContentRegionAvail());
    const bool extent_changed = extent.width != content_region.x || extent.height != content_region.y;
    if (extent_changed) {
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
            const vk::DescriptorBufferInfo object_pick_key{*Buffers->ObjectPickKeyBuffer, 0, SceneBuffers::MaxSelectableObjects * sizeof(uint32_t)};
            const vk::DescriptorBufferInfo element_pick_candidates{*Buffers->ElementPickCandidateBuffer, 0, SceneBuffers::ElementPickGroupCount * sizeof(ElementPickCandidate)};
            const auto &sil = Pipelines->Silhouette;
            const auto &sil_edge = Pipelines->SilhouetteEdge;
            const vk::DescriptorImageInfo silhouette_sampler{*sil.Resources->ImageSampler, *sil.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo object_id_sampler{*sil_edge.Resources->ImageSampler, *sil_edge.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo depth_sampler{*sil_edge.Resources->DepthSampler, *sil_edge.Resources->DepthImage.View, vk::ImageLayout::eDepthStencilReadOnlyOptimal};
            const auto selection_bitset = Buffers->GetSelectionBitsetDescriptor();
            Vk.Device.updateDescriptorSets(
                {
                    Slots->MakeImageWrite(SelectionHandles->HeadImage, head_image_info),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->SelectionCounter}, selection_counter),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->ObjectPickKey}, object_pick_key),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->ElementPickCandidates}, element_pick_candidates),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->SelectionBitset}, selection_bitset),
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

    // Always ensure DrawDataSlot points to render draw data before submitting (may have been overwritten by a selection pass).
    Buffers->SceneViewUBO.Update(as_bytes(Buffers->RenderDrawData.Slot), offsetof(SceneViewUBO, DrawDataSlot));

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

void Scene::ExitLookThroughCamera() {
    if (!SavedViewCamera) return;
    R.replace<ViewCamera>(SceneEntity, *SavedViewCamera);
    SavedViewCamera.reset();
}

void Scene::SnapToCamera(entt::entity camera_entity) {
    const auto &wt = R.get<WorldTransform>(camera_entity);
    const vec3 pos = wt.Position;
    const vec3 fwd = -glm::normalize(glm::rotate(Vec4ToQuat(wt.Rotation), vec3{0.f, 0.f, 1.f}));
    R.replace<ViewCamera>(SceneEntity, ViewCamera{pos, pos + fwd, R.get<CameraData>(camera_entity)});
}
void Scene::AnimateToCamera(entt::entity camera_entity) {
    const auto &wt = R.get<WorldTransform>(camera_entity);
    const vec3 pos = wt.Position;
    const vec3 fwd = -glm::normalize(glm::rotate(Vec4ToQuat(wt.Rotation), vec3{0.f, 0.f, 1.f}));
    const vec3 away = -fwd; // Forward() points from target to position
    R.patch<ViewCamera>(SceneEntity, [&](auto &vc) {
        vc.AnimateTo(pos + fwd, {std::atan2(away.z, away.x), std::asin(away.y)}, 1.f);
    });
}

void Scene::RenderOverlay() {
    const auto viewport = GetViewportRect();
    { // Transform mode pill buttons (top-left overlay)
        struct ButtonInfo {
            const SvgResource &Icon;
            TransformGizmo::Type ButtonType;
            ImDrawFlags Corners;
            bool Enabled;
        };

        using enum TransformGizmo::Type;
        const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
        const bool has_frozen_selected = R.view<Selected, Frozen>().begin() != R.view<Selected, Frozen>().end();
        const bool edit_transform_locked = interaction_mode == InteractionMode::Edit &&
            any_of(GetSelectedMeshEntities(R), [&](entt::entity mesh_entity) { return HasFrozenInstance(R, mesh_entity); });
        const bool transform_enabled = !edit_transform_locked;
        const bool scale_enabled = transform_enabled && !has_frozen_selected;
        const ButtonInfo buttons[]{
            {*Icons.SelectBox, None, ImDrawFlags_RoundCornersTop, true},
            {*Icons.Select, None, ImDrawFlags_RoundCornersBottom, true},
            {*Icons.Move, Translate, ImDrawFlags_RoundCornersTop, transform_enabled},
            {*Icons.Rotate, Rotate, ImDrawFlags_RoundCornersNone, transform_enabled},
            {*Icons.Scale, Scale, ImDrawFlags_RoundCornersNone, scale_enabled},
            {*Icons.Universal, Universal, ImDrawFlags_RoundCornersBottom, transform_enabled},
        };

        auto &transform_type = MGizmo.Config.Type;
        if (!transform_enabled) transform_type = None;
        else if (!scale_enabled && transform_type == Scale) transform_type = Translate;

        const float padding = GetTextLineHeightWithSpacing() / 2.f;
        const auto start_pos = std::bit_cast<ImVec2>(viewport.pos) + GetWindowContentRegionMin() + ImVec2{padding, padding};
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
                    SelectionMode = SelectionMode::Box;
                    transform_type = None;
                } else if (i == 1) {
                    SelectionMode = SelectionMode::Click;
                    transform_type = None;
                } else { // Transform buttons
                    transform_type = button_type;
                }
            }

            const bool is_active = i < 2 ?
                (transform_type == None && ((i == 0 && SelectionMode == SelectionMode::Box) || (i == 1 && SelectionMode == SelectionMode::Click))) :
                transform_type == button_type;
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

    // Exit "look through" camera view if the user interacts with the orientation gizmo.
    if (OrientationGizmo::IsActive()) ExitLookThroughCamera();
    auto &camera = R.get<ViewCamera>(SceneEntity);
    { // Orientation gizmo (drawn before tick so camera animations it initiates begin this frame)
        static constexpr float OGizmoSize{90};
        const float padding = GetTextLineHeightWithSpacing();
        const auto pos = viewport.pos + vec2{GetWindowContentRegionMax().x, GetWindowContentRegionMin().y} - vec2{OGizmoSize, 0} + vec2{-padding, padding};
        OrientationGizmo::Draw(pos, OGizmoSize, camera);
    }
    if (camera.Tick()) R.patch<ViewCamera>(SceneEntity, [](auto &) {});

    const auto selected_view = R.view<const Selected>();
    const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;

    // Check if there's anything to transform:
    // - Object mode: at least one object selected
    // - Edit mode: at least one element selected within selected meshes
    const auto has_transform_target = [&]() {
        if (selected_view.empty()) return false;
        if (interaction_mode != InteractionMode::Edit) return true;
        for (const auto [e, mi] : R.view<const MeshInstance, const Selected>(entt::exclude<Frozen>).each()) {
            if (!R.get<const MeshSelection>(mi.MeshEntity).Handles.empty()) return true;
        }
        return false;
    }();
    if (has_transform_target) { // Transform gizmo
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
        const auto edit_transform_instances = interaction_mode == InteractionMode::Edit ?
            ComputePrimaryEditInstances(R, false) :
            std::unordered_map<entt::entity, entt::entity>{};

        vec3 pivot{};
        if (interaction_mode == InteractionMode::Edit) {
            // Compute world-space centroid of selected vertices once per selected mesh
            // (using a representative selected instance for world transform).
            uint32_t vertex_count = 0;
            const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
            for (const auto &[mesh_entity, instance_entity] : edit_transform_instances) {
                const auto &mesh = R.get<const Mesh>(mesh_entity);
                const auto vertices = mesh.GetVerticesSpan();
                const auto &selection = R.get<const MeshSelection>(mesh_entity);
                if (selection.Handles.empty()) continue;
                const auto &wt = R.get<const WorldTransform>(instance_entity);
                const auto vertex_handles = ConvertSelectionElement(selection, mesh, edit_mode, Element::Vertex);
                for (const auto vi : vertex_handles) {
                    pivot += wt.Position + glm::rotate(Vec4ToQuat(wt.Rotation), wt.Scale * vertices[vi].Position);
                    ++vertex_count;
                }
            }
            if (vertex_count > 0) pivot /= float(vertex_count);
            // Apply pending transform to gizmo position (vertices aren't modified until commit).
            if (const auto *pending = R.try_get<const PendingTransform>(SceneEntity)) {
                pivot += pending->P;
            }
        } else {
            pivot = fold_left(root_selected | transform([&](auto e) { return R.get<Position>(e).Value; }), vec3{}, std::plus{}) / float(root_count);
        }

        const auto start_transform_view = R.view<const StartTransform>();
        const auto gizmo_transform = GizmoTransform{{.P = pivot, .R = active_transform.R, .S = active_transform.S}, MGizmo.Mode};
        auto interact_result = TransformGizmo::Interact(
            gizmo_transform,
            MGizmo.Config, camera, viewport, ToGlm(GetMousePos()) + AccumulatedWrapMouseDelta,
            StartScreenTransform
        );
        if (interact_result) {
            const auto &[ts, td] = *interact_result;
            if (start_transform_view.empty()) {
                if (interaction_mode == InteractionMode::Edit) {
                    for (const auto &[_, instance_entity] : edit_transform_instances) {
                        R.emplace<StartTransform>(instance_entity, GetTransform(R, instance_entity));
                    }
                } else {
                    for (const auto e : root_selected) R.emplace<StartTransform>(e, GetTransform(R, e));
                }
            }
            if (interaction_mode == InteractionMode::Edit) {
                // Edit mode: store pending transform for shader-based preview.
                // Actual vertex positions are only modified on commit.
                R.emplace_or_replace<PendingTransform>(SceneEntity, ts.P, ts.R, td.P, td.R, td.S);
            } else {
                // Object mode: apply transform to entity components immediately during drag.
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
            }
        } else if (!start_transform_view.empty()) {
            R.clear<StartTransform>(); // Transform ended - triggers commit in ProcessComponentEvents.
        }

        // Render gizmo at the post-delta position so it matches the applied transform.
        auto render_transform = gizmo_transform;
        if (interact_result) render_transform.P = interact_result->Start.P + interact_result->Delta.P;
        TransformGizmo::Render(render_transform, MGizmo.Config.Type, camera, viewport);
    }

    if (!R.storage<Selected>().empty()) { // Draw center-dot for active/selected entities
        const auto &theme = R.get<const ViewportTheme>(SceneEntity);
        const auto vp = camera.Projection(viewport.size.x / viewport.size.y) * camera.View();
        for (const auto [e, wt] : R.view<const WorldTransform>().each()) {
            if (!R.any_of<Active, Selected>(e)) continue;

            const auto p_cs = vp * vec4{wt.Position, 1.f}; // World to clip space
            const auto p_ndc = fabsf(p_cs.w) > FLT_EPSILON ? vec3{p_cs} / p_cs.w : vec3{p_cs}; // Clip space to NDC
            const auto p_uv = vec2{p_ndc.x + 1, 1 - p_ndc.y} * 0.5f; // NDC to UV [0,1] (top-left origin)
            const auto p_px = std::bit_cast<ImVec2>(viewport.pos + p_uv * viewport.size); // UV to px
            auto &dl = *GetWindowDrawList();
            dl.AddCircleFilled(p_px, 3.5f, colors::RgbToU32(R.all_of<Active>(e) ? theme.Colors.ObjectActive : theme.Colors.ObjectSelected), 10);
            dl.AddCircle(p_px, 3.5f, IM_COL32(0, 0, 0, 255), 10, 1.f);
        }
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

    // Camera look-through frame overlay: show the active camera's view as a centered frame.
    // The ViewCamera's FOV is widened so the active camera's view fits inside with padding.
    // The frame marks exactly what the active camera captures.
    if (SavedViewCamera && !camera.IsAnimating()) {
        const auto active_entity = FindActiveEntity(R);
        if (const auto *cd = active_entity != entt::null ? R.try_get<CameraData>(active_entity) : nullptr) {
            const float cam_aspect = AspectRatio(*cd);
            const float frame_ratio = LookThroughFrameRatio(cam_aspect, viewport.size.x / viewport.size.y);
            const vec2 frame_size{viewport.size.y * frame_ratio * cam_aspect, viewport.size.y * frame_ratio};
            const vec2 vp_center = viewport.pos + viewport.size * 0.5f;
            const vec2 fmin = vp_center - frame_size * 0.5f, fmax = vp_center + frame_size * 0.5f;
            const auto vmin = viewport.pos, vmax = viewport.pos + viewport.size;

            auto &dl = *GetWindowDrawList();
            static constexpr auto dim = IM_COL32(0, 0, 0, 100);
            auto iv = [](vec2 v) { return std::bit_cast<ImVec2>(v); };
            dl.AddRectFilled(iv(vmin), iv({vmax.x, fmin.y}), dim);
            dl.AddRectFilled(iv({vmin.x, fmax.y}), iv(vmax), dim);
            dl.AddRectFilled(iv({vmin.x, fmin.y}), iv({fmin.x, fmax.y}), dim);
            dl.AddRectFilled(iv({fmax.x, fmin.y}), iv({vmax.x, fmax.y}), dim);
            dl.AddRect(iv(fmin), iv(fmax), IM_COL32(255, 255, 255, 140), 0.f, 0, 1.5f);
        }
    }

    { // Viewport info overlay
        const auto &settings = R.get<const SceneSettings>(SceneEntity);
        const char *label = settings.ViewportShading == ViewportShadingMode::Rendered ? "Rendered" : "Solid";
        const auto text = std::format("Shading: {}", label);
        const ImVec2 text_size = CalcTextSize(text.c_str());
        const ImVec2 text_pos{
            viewport.pos.x + viewport.size.x - text_size.x - 10.f,
            viewport.pos.y + viewport.size.y - text_size.y - 10.f,
        };
        auto &dl = *GetWindowDrawList();
        dl.AddRectFilled(
            {text_pos.x - 6.f, text_pos.y - 4.f},
            {text_pos.x + text_size.x + 6.f, text_pos.y + text_size.y + 4.f},
            IM_COL32(0, 0, 0, 110),
            4.f
        );
        dl.AddText(text_pos, IM_COL32(230, 230, 230, 255), text.c_str());
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
    if (const auto *armature_modifier = R.try_get<ArmatureModifier>(active_entity)) {
        Text("Armature data: %s", GetName(R, armature_modifier->ArmatureDataEntity).c_str());
        if (armature_modifier->ArmatureObjectEntity != entt::null) {
            Text("Armature object: %s", GetName(R, armature_modifier->ArmatureObjectEntity).c_str());
        }
    }
    if (const auto *bone_attachment = R.try_get<BoneAttachment>(active_entity)) {
        Text("Attached bone ID: %u", bone_attachment->Bone);
    }
    const auto object_type = R.all_of<ObjectKind>(active_entity) ? R.get<const ObjectKind>(active_entity).Value : ObjectType::Empty;
    Text("Object type: %s", ObjectTypeName(object_type).data());
    {
        const auto model_buffer_index = GetModelBufferIndex(R, active_entity);
        Text("Model buffer index: %s", model_buffer_index ? std::to_string(*model_buffer_index).c_str() : "None");
    }
    const auto *active_mesh_instance = R.try_get<MeshInstance>(active_entity);
    if (active_mesh_instance) {
        const auto active_mesh_entity = active_mesh_instance->MeshEntity;
        const auto &active_mesh = R.get<const Mesh>(active_mesh_entity);
        TextUnformatted(
            std::format("Vertices | Edges | Faces: {:L} | {:L} | {:L}", active_mesh.VertexCount(), active_mesh.EdgeCount(), active_mesh.FaceCount()).c_str()
        );
    } else if (const auto *armature = R.try_get<ArmatureObject>(active_entity)) {
        const auto &armature_data = R.get<const ArmatureData>(armature->DataEntity);
        Text("Bones: %zu", armature_data.Bones.size());
    }
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
            UpdateWorldTransform(R, active_entity);
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
        if (TreeNode("World transform")) {
            const auto &wt = R.get<WorldTransform>(active_entity);
            Text("Position: %.3f, %.3f, %.3f", wt.Position.x, wt.Position.y, wt.Position.z);
            Text("Rotation: %.3f, %.3f, %.3f, %.3f", wt.Rotation.x, wt.Rotation.y, wt.Rotation.z, wt.Rotation.w);
            Text("Scale: %.3f, %.3f, %.3f", wt.Scale.x, wt.Scale.y, wt.Scale.z);
            TreePop();
        }
    }
    if (active_mesh_instance) {
        const auto active_mesh_entity = active_mesh_instance->MeshEntity;
        if (const auto *primitive_type = R.try_get<PrimitiveType>(active_mesh_entity)) {
            const bool frozen = HasFrozenInstance(R, active_mesh_entity);
            if (frozen) BeginDisabled();
            const auto update_label = std::format("Update primitive{}", frozen ? " (frozen)" : "");
            if (CollapsingHeader(update_label.c_str()) && !frozen) {
                if (auto mesh_data = PrimitiveEditor(*primitive_type, false)) {
                    SetMeshPositions(active_mesh_entity, std::move(mesh_data->Positions));
                }
            }
            if (frozen) EndDisabled();
        }
    }
    if (auto *cd = R.try_get<CameraData>(active_entity)) {
        if (CollapsingHeader("Camera")) {
            if (RenderCameraLensEditor(*cd)) R.patch<CameraData>(active_entity, [](auto &) {});
            Separator();
            if (SavedViewCamera) {
                if (Button("Exit camera view")) ExitLookThroughCamera();
            } else {
                if (Button("Look through")) {
                    SavedViewCamera = R.get<ViewCamera>(SceneEntity);
                    AnimateToCamera(active_entity);
                }
            }
        }
    }
    if (R.all_of<LightIndex>(active_entity) &&
        CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto light = GetLight(*Buffers, R.get<const LightIndex>(active_entity).Value);
        bool changed = false;
        bool wireframe_changed = false;
        constexpr float MaxLightIntensity = 1000.f;
        constexpr float MaxLightRange = 1000.f;

        const char *type_names[]{"Directional", "Point", "Spot"};
        int type_i = int(std::min(light.Type, LightTypeSpot));
        if (Combo("Type", &type_i, type_names, IM_ARRAYSIZE(type_names))) {
            auto next = MakeDefaultLight(uint32_t(type_i));
            next.TransformSlotOffset = light.TransformSlotOffset;
            next.Color = light.Color;
            next.Intensity = light.Intensity;
            light = next;
            changed = true;
            wireframe_changed = true;
        }

        changed |= ColorEdit3("Color", &light.Color.x);
        changed |= SliderFloat("Intensity", &light.Intensity, 0.f, MaxLightIntensity, "%.2f");

        if (light.Type == LightTypePoint || light.Type == LightTypeSpot) {
            bool infinite_range = light.Range <= 0.f;
            if (Checkbox("Infinite range", &infinite_range)) {
                light.Range = infinite_range ? 0.f : std::max(light.Range, DefaultPointRange);
                changed = true;
                wireframe_changed = true;
            }
            if (!infinite_range) {
                if (SliderFloat("Range", &light.Range, 0.01f, MaxLightRange, "%.2f")) {
                    changed = true;
                    wireframe_changed = true;
                }
            }
        }

        if (light.Type == LightTypeSpot) {
            float outer_deg = glm::degrees(AngleFromCos(light.OuterConeCos));
            outer_deg = std::clamp(outer_deg, 0.f, 90.f);
            float inner_deg = glm::degrees(AngleFromCos(light.InnerConeCos));
            inner_deg = std::clamp(inner_deg, 0.f, outer_deg);
            float blend = outer_deg > 1e-4f ? std::clamp(1.f - inner_deg / outer_deg, 0.f, 1.f) : 0.f;

            if (SliderFloat("Size", &outer_deg, 0.f, 90.f, "%.1f deg")) {
                outer_deg = std::clamp(outer_deg, 0.f, 90.f);
                const float outer_rad = glm::radians(outer_deg);
                const float inner_rad = outer_rad * (1.f - blend);
                light.OuterConeCos = std::cos(outer_rad);
                light.InnerConeCos = std::cos(inner_rad);
                changed = true;
                wireframe_changed = true;
            }
            if (SliderFloat("Blend", &blend, 0.f, 1.f, "%.2f")) {
                blend = std::clamp(blend, 0.f, 1.f);
                const float outer_rad = glm::radians(std::clamp(outer_deg, 0.f, 90.f));
                const float inner_rad = outer_rad * (1.f - blend);
                light.OuterConeCos = std::cos(outer_rad);
                light.InnerConeCos = std::cos(inner_rad);
                changed = true;
                wireframe_changed = true;
            }
        }

        if (changed) {
            SetLight(*Buffers, R.get<const LightIndex>(active_entity).Value, light);
            R.emplace_or_replace<LightDataDirty>(active_entity);
        }
        if (wireframe_changed) R.emplace_or_replace<LightWireframeDirty>(active_entity);
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
                const bool edit_allowed = AllSelectedAreMeshes(R);
                for (const auto mode : InteractionModes) {
                    if (mode == InteractionMode::Edit && !edit_allowed) continue;
                    SameLine();
                    interaction_mode_changed |= RadioButton(to_string(mode).c_str(), &interaction_mode_value, int(mode));
                }
                if (interaction_mode_changed) SetInteractionMode(InteractionMode(interaction_mode_value));
                if (interaction_mode == InteractionMode::Edit || interaction_mode == InteractionMode::Excite) {
                    Checkbox("Orbit to active", &OrbitToActive);
                }
                if (interaction_mode == InteractionMode::Edit) {
                    Checkbox("X-ray selection", &SelectionXRay);
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
                        if (const auto *mesh_instance = R.try_get<MeshInstance>(active_entity)) {
                            const auto mesh_entity = mesh_instance->MeshEntity;
                            const auto &selection = R.get<MeshSelection>(mesh_entity);
                            Text("Editing %s: %zu selected", label(edit_mode).data(), selection.Handles.size());
                            if (edit_mode == Element::Vertex && !selection.Handles.empty()) {
                                const auto &mesh = R.get<Mesh>(mesh_entity);
                                for (const auto vh : selection.Handles) {
                                    const auto pos = mesh.GetPosition(VH{vh});
                                    Text("Vertex %u: (%.4f, %.4f, %.4f)", vh, pos.x, pos.y, pos.z);
                                }
                            }
                        } else {
                            TextUnformatted("Edit mode requires an active mesh object.");
                        }
                    }
                }
                PopID();
            }
            if (!R.storage<Selected>().empty()) {
                SeparatorText("Selection actions");
                std::vector<entt::entity> selected_mesh_instances;
                for (const auto entity : R.view<const Selected, const MeshInstance>()) selected_mesh_instances.emplace_back(entity);

                if (!selected_mesh_instances.empty()) {
                    const bool any_visible = any_of(selected_mesh_instances, [&](entt::entity e) { return R.all_of<RenderInstance>(e); });
                    const bool any_hidden = any_of(selected_mesh_instances, [&](entt::entity e) { return !R.all_of<RenderInstance>(e); });
                    const bool mixed_visible = any_visible && any_hidden;
                    if (mixed_visible) ImGui::PushItemFlag(ImGuiItemFlags_MixedValue, true);
                    if (bool set_visible = any_visible && !any_hidden; Checkbox("Visible", &set_visible)) {
                        for (const auto e : selected_mesh_instances) SetVisible(e, set_visible);
                    }
                    if (mixed_visible) ImGui::PopItemFlag();
                }
                if (Button("Duplicate")) Duplicate();
                SameLine();
                if (Button("Duplicate linked")) DuplicateLinked();
                if (Button("Delete")) Delete();
            }
            RenderEntityControls(FindActiveEntity(R));

            if (CollapsingHeader("Add object")) {
                TextDisabled("Shortcuts: Ctrl+Shift+E (Empty), Ctrl+Shift+A (Armature), Ctrl+Shift+C (Camera)");
                if (Button("Add Empty")) {
                    AddEmpty({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
                    StartScreenTransform = TransformGizmo::TransformType::Translate;
                }
                SameLine();
                if (Button("Add Armature")) {
                    AddArmature({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
                    StartScreenTransform = TransformGizmo::TransformType::Translate;
                }
                SameLine();
                if (Button("Add Camera")) {
                    AddCamera({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
                    StartScreenTransform = TransformGizmo::TransformType::Translate;
                }
                SameLine();
                if (Button("Add Light")) {
                    AddLight({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
                    StartScreenTransform = TransformGizmo::TransformType::Translate;
                }
            }

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
            viewport_shading_changed |= RadioButton("Rendered", &viewport_shading, int(ViewportShadingMode::Rendered));
            PopID();
            TextDisabled("Shortcut: Z");

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
                changed |= ColorEdit3("Transform", &theme.Colors.Transform.x);
                changed |= SliderUInt("Silhouette edge width", &theme.SilhouetteEdgeWidth, 1, 4);
                if (changed) R.patch<ViewportTheme>(SceneEntity, [](auto &) {});
            }
            if (settings_changed) R.patch<SceneSettings>(SceneEntity, [](auto &) {});
            EndTabItem();
        }

        if (BeginTabItem("Camera")) {
            auto &camera = R.get<ViewCamera>(SceneEntity);
            bool changed = false;
            const auto extent = R.get<const ViewportExtent>(SceneEntity).Value;
            const float viewport_aspect = extent.width == 0 || extent.height == 0 ? 1.f : float(extent.width) / float(extent.height);
            if (Button("Reset##Camera")) {
                camera = Defaults.ViewCamera;
                changed = true;
            }
            changed |= SliderFloat3("Target", &camera.Target.x, -10, 10);
            changed |= RenderCameraLensEditor(camera.Data, ViewportContext{.Distance = camera.Distance, .AspectRatio = viewport_aspect});
            if (changed) R.patch<ViewCamera>(SceneEntity, [](auto &camera) { camera.StopMoving(); });
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
    if (MeshEditor::BeginTable(name.c_str(), 4)) {
        static const float CharWidth = CalcTextSize("A").x;
        TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, CharWidth * 10);
        TableSetupColumn("Name");
        TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, CharWidth * 10);
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
            const auto type = R.all_of<ObjectKind>(e) ?
                R.get<const ObjectKind>(e).Value :
                R.all_of<MeshInstance>(e) ? ObjectType::Mesh :
                                            ObjectType::Empty;
            TextUnformatted(ObjectTypeName(type).data());
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
