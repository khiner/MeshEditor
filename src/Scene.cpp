#include "Scene.h"
#include "SceneDefaults.h"
#include "SceneMaterials.h"
#include "SceneTextures.h"
#include "SceneTree.h"

#include "Armature.h"
#include "BBox.h"
#include "Bindless.h"
#include "Excitable.h"
#include "File.h"
#include "MeshComponents.h"
#include "MeshInstance.h"
#include "ScenePipelines.h"
#include "SceneSelection.h"
#include "Shader.h"
#include "SvgResource.h"
#include "Timer.h"
#include "gpu/BoxSelectPushConstants.h"
#include "gpu/ElementPickPushConstants.h"
#include "gpu/ObjectPickPushConstants.h"
#include "gpu/SelectionElementFlags.h"
#include "gpu/SelectionElementPushConstants.h"
#include "gpu/SilhouetteEdgeColorPushConstants.h"
#include "gpu/SilhouetteEdgeDepthObjectPushConstants.h"
#include "mesh/MeshData.h"
#include "mesh/MeshStore.h"
#include "mesh/PrimitiveType.h"
#include "scene_impl/SceneInternalTypes.h"

#include "imgui.h"
#include <entt/entity/registry.hpp>

#include "Variant.h"
#include <iostream>

using std::ranges::any_of, std::ranges::all_of, std::ranges::find, std::ranges::fold_left, std::ranges::to;
using std::views::filter, std::views::iota, std::views::transform;

namespace {
constexpr SelectionElementFlags operator|(SelectionElementFlags a, SelectionElementFlags b) {
    return static_cast<SelectionElementFlags>(uint32_t(a) | uint32_t(b));
}

constexpr vk::Extent2D ToExtent2D(vk::Extent3D extent) { return {extent.width, extent.height}; }
const vk::ClearColorValue Transparent{0, 0, 0, 0};

using namespace he;

void WaitFor(vk::Fence fence, vk::Device device) {
    if (auto wait_result = device.waitForFences(fence, VK_TRUE, UINT64_MAX); wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    device.resetFences(fence);
}
} // namespace

#include "scene_impl/SceneComponents.h"

#include "scene_impl/SceneBuffers.h"
#include "scene_impl/SceneDrawing.h"
#include "scene_impl/SceneTransformUtils.h"

namespace {
std::vector<uint> CreateVertexIndices(const Mesh &mesh) { return iota(0u, mesh.VertexCount()) | to<std::vector>(); }
std::vector<uint> CreateNormalIndices(const Mesh &mesh, Element element) {
    if (element == Element::None || element == Element::Edge) return {};
    const auto n = element == Element::Face ? mesh.FaceCount() : mesh.VertexCount();
    return iota(0u, n * 2) | to<std::vector<uint>>();
}
std::vector<Vertex> CreateNormalVertices(const Mesh &mesh, Element element) {
    constexpr float NormalIndicatorLengthScale{0.25};
    std::vector<Vertex> vertices;
    if (element == Element::Vertex) {
        vertices.reserve(mesh.VertexCount() * 2);
        for (const auto vh : mesh.vertices()) {
            const auto vn = mesh.GetNormal(vh);
            const auto &voh_range = mesh.voh_range(vh);
            const float total_edge_length = std::reduce(voh_range.begin(), voh_range.end(), 0.f, [&](float total, const auto &heh) {
                return total + mesh.CalcEdgeLength(heh);
            });
            const float avg_edge_length = total_edge_length / mesh.GetValence(vh);
            const auto p = mesh.GetPosition(vh);
            vertices.emplace_back(p, vn);
            vertices.emplace_back(p + NormalIndicatorLengthScale * avg_edge_length * vn, vn);
        }
    } else if (element == Element::Face) {
        vertices.reserve(mesh.FaceCount() * 2);
        for (const auto fh : mesh.faces()) {
            const auto fn = mesh.GetNormal(fh);
            const auto p = mesh.CalcFaceCentroid(fh);
            vertices.emplace_back(p, fn);
            vertices.emplace_back(p + NormalIndicatorLengthScale * std::sqrt(mesh.CalcFaceArea(fh)) * fn, fn);
        }
    }
    return vertices;
}

// Returns `std::nullopt` if the entity does not have a RenderInstance (i.e., is not visible).
std::optional<uint32_t> GetModelBufferIndex(const entt::registry &r, entt::entity e) {
    if (const auto *ri = r.try_get<RenderInstance>(e)) return ri->BufferIndex;
    return {};
}
} // namespace

void UpdateModelBuffer(entt::registry &r, entt::entity e, const WorldTransform &wt) {
    if (const auto i = GetModelBufferIndex(r, e)) {
        const auto mesh_entity = r.get<MeshInstance>(e).MeshEntity;
        r.patch<ModelsBuffer>(mesh_entity, [&wt, i](auto &mb) { mb.Buffer.Update(as_bytes(wt), *i * sizeof(WorldTransform)); });
    }
}

struct ExtrasWireframe {
    MeshData Data;
    std::vector<uint8_t> VertexClasses{}; // Empty means all VCLASS_NONE (no buffer needed).

    uint32_t AddVertex(vec3 pos, uint8_t vclass) {
        const uint32_t i = Data.Positions.size();
        Data.Positions.push_back(pos);
        VertexClasses.push_back(vclass);
        return i;
    }
    void AddEdge(uint32_t a, uint32_t b) { Data.Edges.push_back({a, b}); }

    void AddCircle(float radius, uint32_t segments, float z, uint8_t vclass, uint32_t edge_stride = 1) {
        const uint32_t base = Data.Positions.size();
        for (uint32_t i = 0; i < segments; ++i) {
            const float angle = float(i) * 2.f * Pi / float(segments);
            AddVertex({radius * std::cos(angle), radius * std::sin(angle), z}, vclass);
        }
        for (uint32_t i = 0; i < segments; i += edge_stride) AddEdge(base + i, base + (i + 1) % segments);
    }

    void AddDiamond(float radius, uint8_t vclass, vec3 axis1, vec3 axis2, vec3 center = {}) {
        const uint32_t base = Data.Positions.size();
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
MeshData BuildCameraFrustumMesh(const Camera &camera) {
    float display_near{0.01f}, display_far{5.f};
    float near_half_w{0.f}, near_half_h{0.f}, far_half_w{0.f}, far_half_h{0.f};
    if (const auto *perspective = std::get_if<Perspective>(&camera)) {
        // Clamp far plane for display so wireframe doesn't extend to infinity.
        display_near = perspective->NearClip;
        display_far = std::min(perspective->FarClip.value_or(5.f), 5.f);

        const float aspect = AspectRatio(camera);
        near_half_h = display_near * std::tan(perspective->FieldOfViewRad * 0.5f);
        near_half_w = near_half_h * aspect;
        far_half_h = display_far * std::tan(perspective->FieldOfViewRad * 0.5f);
        far_half_w = far_half_h * aspect;
    } else if (const auto *orthographic = std::get_if<Orthographic>(&camera)) {
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

float AngleFromCos(float cos_theta) { return std::acos(std::clamp(cos_theta, -1.f, 1.f)); }

ExtrasWireframe BuildLightMesh(const PunctualLight &light) {
    ExtrasWireframe wf;

    const auto add_range_circle = [&](float range) {
        if (range > 0.f) wf.AddCircle(range, 32, 0.f, VClassBillboard);
    };

    if (light.Type == PunctualLightType::Point) {
        add_range_circle(light.Range);
    } else if (light.Type == PunctualLightType::Directional) {
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
    } else if (light.Type == PunctualLightType::Spot) {
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

void ResetObjectPickKeys(SceneBuffers &buffers) {
    auto bytes = buffers.ObjectPickKeyBuffer.GetMappedData();
    std::fill_n(reinterpret_cast<uint32_t *>(bytes.data()), SceneBuffers::MaxSelectableObjects, std::numeric_limits<uint32_t>::max());
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
    MeshMaterial = "mesh_material_changes"_hs,
    Excitable = "excitable_changes"_hs,
    ExcitedVertex = "excited_vertex_changes"_hs,
    ModelsBuffer = "models_buffer_changes"_hs,
    SceneSettings = "scene_settings_changes"_hs,
    InteractionMode = "interaction_mode_changes"_hs,
    Submit = "submit_changes"_hs,
    ViewportTheme = "viewport_theme_changes"_hs,
    Materials = "materials_changes"_hs,
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
        if (const auto *pose_state = r.try_get<const ArmaturePoseState>(modifier.ArmatureEntity)) {
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

bool HasSelectedInstance(const entt::registry &r, entt::entity mesh_entity) {
    return any_of(r.view<const MeshInstance, const Selected>().each(), [mesh_entity](const auto &t) { return std::get<1>(t).MeshEntity == mesh_entity; });
}

void RunSelectionCompute(
    vk::CommandBuffer cb, vk::Queue queue, vk::Fence fence, vk::Device device,
    const auto &compute, const auto &pc, auto &&dispatch, vk::Semaphore wait_semaphore = {}
) {
    const Timer timer{"RunSelectionCompute"};
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    cb.bindPipeline(vk::PipelineBindPoint::eCompute, *compute.Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *compute.PipelineLayout, 0, compute.GetDescriptorSet(), {});
    cb.pushConstants(*compute.PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
    dispatch(cb);

    const vk::MemoryBarrier barrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead};
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, barrier, {}, {});
    cb.end();

    vk::SubmitInfo submit;
    submit.setCommandBuffers(cb);
    static constexpr vk::PipelineStageFlags wait_stage{vk::PipelineStageFlagBits::eComputeShader};
    if (wait_semaphore) {
        submit.setWaitSemaphores(wait_semaphore);
        submit.setWaitDstStageMask(wait_stage);
    }
    queue.submit(submit, fence);
    WaitFor(fence, device);
}

// After rendering elements to selection buffer, dispatch compute shader to find the nearest element to mouse_px.
// Returns 0-based element index, or nullopt if no element found.
std::optional<uint32_t> FindNearestPickedElement(
    const SceneBuffers &buffers, const ComputePipeline &compute, vk::CommandBuffer cb,
    vk::Queue queue, vk::Fence fence, vk::Device device,
    uint32_t head_image_index, uint32_t selection_nodes_slot, uint32_t element_candidate_buffer_slot,
    uvec2 mouse_px, uint32_t max_element_id, Element element,
    vk::Semaphore wait_semaphore
) {
    const uint32_t radius = element == Element::Face ? 0u : ElementSelectRadiusPx;
    const uint32_t group_count = element == Element::Face ? 1u : SceneBuffers::ElementPickGroupCount;
    RunSelectionCompute(
        cb, queue, fence, device, compute,
        ElementPickPushConstants{
            .TargetPx = mouse_px,
            .Radius = radius,
            .HeadImageIndex = head_image_index,
            .SelectionNodesIndex = selection_nodes_slot,
            .ElementCandidateBufferIndex = element_candidate_buffer_slot,
        },
        [group_count](vk::CommandBuffer dispatch_cb) { dispatch_cb.dispatch(group_count, 1, 1); },
        wait_semaphore
    );

    const auto *candidates = reinterpret_cast<const ElementPickCandidate *>(buffers.ElementPickCandidateBuffer.GetData().data());
    ElementPickCandidate best{.Id = 0, .Depth = 1.0f, .DistanceSq = std::numeric_limits<uint32_t>::max()};
    for (uint32_t i = 0; i < group_count; ++i) {
        const auto &candidate = candidates[i];
        if (candidate.Id == 0) continue;
        if (candidate.DistanceSq < best.DistanceSq || (candidate.DistanceSq == best.DistanceSq && candidate.Depth < best.Depth)) {
            best = candidate;
        }
    }

    if (best.Id == 0 || best.Id > max_element_id) return {};
    return best.Id - 1;
}
} // namespace

uint32_t GetMaterialCount(const SceneBuffers &buffers) { return buffers.MaterialBuffer.UsedSize / sizeof(PBRMaterial); }
PBRMaterial GetMaterial(const SceneBuffers &buffers, uint32_t index) { return reinterpret_cast<const PBRMaterial *>(buffers.MaterialBuffer.GetMappedData().data())[index]; }
void SetMaterial(SceneBuffers &buffers, uint32_t index, const PBRMaterial &material) { buffers.MaterialBuffer.Update(as_bytes(material), vk::DeviceSize(index) * sizeof(PBRMaterial)); }
uint32_t AppendMaterial(SceneBuffers &buffers, const PBRMaterial &material) {
    const auto index = GetMaterialCount(buffers);
    SetMaterial(buffers, index, material);
    return index;
}
void SetMaterialCount(SceneBuffers &buffers, uint32_t count) {
    buffers.MaterialBuffer.Reserve(vk::DeviceSize(count) * sizeof(PBRMaterial));
    buffers.MaterialBuffer.UsedSize = vk::DeviceSize(count) * sizeof(PBRMaterial);
}

mvk::BufferContext &GetBufferCtx(SceneBuffers *b) { return b->Ctx; }
Range AllocateArmatureDeform(SceneBuffers *b, uint32_t count) { return b->ArmatureDeformBuffer.Allocate(count); }
std::span<mat4> GetArmatureDeformMutable(SceneBuffers *b, Range r) { return b->ArmatureDeformBuffer.GetMutable(r); }
Range AllocateMorphWeights(SceneBuffers *b, uint32_t count) { return b->MorphWeightBuffer.Allocate(count); }
std::span<float> GetMorphWeightsMutable(SceneBuffers *b, Range r) { return b->MorphWeightBuffer.GetMutable(r); }

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
      Meshes{std::make_unique<MeshStore>(Buffers->Ctx)},
      Textures{std::make_unique<TextureStore>()},
      Environments{std::make_unique<EnvironmentStore>()} {
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
    R.storage<entt::reactive>(changes::MeshMaterial)
        .on_construct<MeshMaterialAssignment>()
        .on_update<MeshMaterialAssignment>();
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
    R.storage<entt::reactive>(changes::Submit)
        .on_construct<SubmitDirty>();
    R.storage<entt::reactive>(changes::ViewportTheme)
        .on_construct<ViewportTheme>()
        .on_update<ViewportTheme>();
    R.storage<entt::reactive>(changes::Materials)
        .on_construct<MaterialEdit>()
        .on_update<MaterialEdit>();
    R.storage<entt::reactive>(changes::SceneView)
        .on_construct<ViewCamera>()
        .on_update<ViewCamera>()
        .on_construct<MaterialPreviewLighting>()
        .on_update<MaterialPreviewLighting>()
        .on_construct<RenderedLighting>()
        .on_update<RenderedLighting>()
        .on_construct<ViewportExtent>()
        .on_update<ViewportExtent>()
        .on_construct<SceneEditMode>()
        .on_update<SceneEditMode>();
    R.storage<entt::reactive>(changes::CameraLens)
        .on_construct<Camera>()
        .on_update<Camera>();
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
    R.emplace<ViewportTheme>(SceneEntity, SceneDefaults::ViewportTheme);
    R.emplace<ViewCamera>(SceneEntity, SceneDefaults::ViewCamera);
    R.emplace<MaterialPreviewLighting>(SceneEntity, false, false, 1.f, 0.f);
    R.emplace<RenderedLighting>(SceneEntity, true, true, 1.f, 0.f);
    R.emplace<ViewportExtent>(SceneEntity);
    R.emplace<AnimationTimeline>(SceneEntity);
    R.emplace<MaterialStore>(SceneEntity);
    Buffers->WorkspaceLightsUBO.Update(as_bytes(SceneDefaults::WorkspaceLights));

    BoxSelectZeroBits.assign(SceneBuffers::BoxSelectBitsetWords, 0);
    ResetObjectPickKeys(*Buffers);

    auto &texture_store = *Textures;
    constexpr std::array<std::byte, 4> white_pixels{std::byte{0xff}, std::byte{0xff}, std::byte{0xff}, std::byte{0xff}};
    auto white_texture = CreateTextureEntry(
        Vk,
        Buffers->Ctx,
        *CommandPool,
        *OneShotFence,
        *Slots,
        white_pixels,
        1,
        1,
        "DefaultWhite",
        TextureColorSpace::Srgb,
        vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode::eRepeat,
        SamplerConfig{}
    );
    texture_store.WhiteTextureSlot = white_texture.SamplerSlot;
    texture_store.Textures.emplace_back(std::move(white_texture));

    auto &environments = *Environments;
    environments.BrdfLut = CreateDefaultLutTexture(Vk, Buffers->Ctx, *CommandPool, *OneShotFence, *Slots, "res/images/lut_ggx.png", "DefaultGGXBRDFLUT");
    environments.SheenELut = CreateDefaultLutTexture(Vk, Buffers->Ctx, *CommandPool, *OneShotFence, *Slots, "res/images/lut_sheen_E.png", "DefaultSheenELUT");
    environments.CharlieLut = CreateDefaultLutTexture(Vk, Buffers->Ctx, *CommandPool, *OneShotFence, *Slots, "res/images/lut_charlie.png", "DefaultCharlieLUT");

    // Discover HDR environment files, sorted by name for stable ordering.
    static constexpr std::string_view HdriDir{"res/images/studiolights/world"};
    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator{HdriDir, ec}) {
        if (entry.path().extension() == ".hdr") {
            const auto stem = entry.path().stem().string();
            environments.Hdris.push_back({.Name = stem, .Path = entry.path(), .Prefiltered = {}});
        }
    }
    std::ranges::sort(environments.Hdris, {}, &HdriEntry::Name);

    // Set the default HDRI: prefer "forest", otherwise use the first entry.
    const auto forest_it = find(environments.Hdris, "forest", &HdriEntry::Name);
    environments.ActiveHdriIndex = forest_it != environments.Hdris.end() ? std::distance(environments.Hdris.begin(), forest_it) : 0;

    SetStudioEnvironment(environments.ActiveHdriIndex);
    environments.SceneWorld = environments.StudioWorld;

    AppendMaterial(
        *Buffers,
        {
            .BaseColorFactor = vec4{1.f},
            .MetallicFactor = 0.f,
            .RoughnessFactor = 1.f,
            .AlphaMode = MaterialAlphaMode::Opaque,
            .AlphaCutoff = 0.5f,
            .DoubleSided = 0u,
            .BaseColorTexture = {.Slot = texture_store.WhiteTextureSlot},
        }
    );
    R.patch<MaterialStore>(SceneEntity, [](auto &material_store) { material_store.Names.emplace_back("Default"); });

    Pipelines->CompileShaders();
}

Scene::~Scene() {
    if (Environments) {
        ReleaseEnvironmentSamplerSlots(*Slots, *Environments);
    }
    if (R.valid(SceneEntity)) {
        auto sampler_slots = CollectSamplerSlots(Textures->Textures);
        ReleaseSamplerSlots(*Slots, sampler_slots);
        R.remove<MaterialStore>(SceneEntity);
    }
    R.clear<Mesh>();
}

World Scene::GetWorld() const { return SceneDefaults::World; }

void Scene::SetStudioEnvironment(uint32_t index) {
    auto &environments = *Environments;
    auto &hdri = environments.Hdris[index];
    if (!hdri.Prefiltered) {
        hdri.Prefiltered = CreateIblFromHdri(
            Vk, Buffers->Ctx, *CommandPool, *OneShotFence, *Slots,
            Pipelines->IblPrefilter, hdri.Path, hdri.Name
        );
    }
    const auto &pre = *hdri.Prefiltered;
    environments.ActiveHdriIndex = index;
    environments.StudioWorld = {.Ibl = MakeIblSamplers(pre, environments), .Name = hdri.Name};
}

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

void Scene::CreateSvgResource(std::unique_ptr<SvgResource> &svg, std::filesystem::path path) {
    const auto RenderBitmap = [this](std::span<const std::byte> data, uint32_t width, uint32_t height) {
        return RenderBitmapToImage(Vk, Buffers->Ctx, *CommandPool, *OneShotFence, data, width, height, Format::Color, ColorSubresourceRange);
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
            if (SavedViewCamera && R.all_of<Active>(instance_entity) && R.all_of<Camera>(instance_entity)) {
                SnapToCamera(instance_entity);
            }
        }
    }
    if (!R.storage<entt::reactive>(changes::Rerecord).empty() || !DestroyTracker->Storage.empty()) {
        request(RenderRequest::ReRecord);
    }

    const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
    const bool is_edit_mode = interaction_mode == InteractionMode::Edit;

    const auto edit_transform_context = is_edit_mode ? EditTransformContext{scene_selection::ComputePrimaryEditInstances(R, false)} : EditTransformContext{};
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
        if (const auto *cd = R.try_get<Camera>(camera_entity)) {
            SetMeshPositions(R.get<MeshInstance>(camera_entity).MeshEntity, BuildCameraFrustumMesh(*cd).Positions);
            // If looking through this camera, trigger a ViewCamera update so the SceneView
            // handler re-derives the widened FOV from the updated camera.
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
    if (!R.storage<entt::reactive>(changes::Submit).empty() || light_count_changed) request(RenderRequest::Submit);
    for (auto light_entity : R.view<LightWireframeDirty, LightIndex, MeshInstance>()) {
        const auto light = Buffers->GetLight(R.get<const LightIndex>(light_entity).Value);
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
    if (auto &tracker = R.storage<entt::reactive>(changes::MeshMaterial); !tracker.empty()) {
        for (auto mesh_entity : tracker) {
            const auto *assignment = R.try_get<const MeshMaterialAssignment>(mesh_entity);
            const auto *mesh = R.try_get<const Mesh>(mesh_entity);
            if (!assignment || !mesh) continue;
            const auto material_count = GetMaterialCount(*Buffers);
            if (material_count == 0u) continue;
            auto primitive_materials = Meshes->GetPrimitiveMaterialIndices(mesh->GetStoreId());
            if (assignment->PrimitiveIndex >= primitive_materials.size()) continue;
            primitive_materials[assignment->PrimitiveIndex] = std::min(assignment->MaterialIndex, material_count - 1u);
        }
        request(RenderRequest::Submit);
    }
    if (!R.storage<entt::reactive>(changes::ModelsBuffer).empty()) request(RenderRequest::Submit);
    if (!R.storage<entt::reactive>(changes::ViewportTheme).empty()) {
        Buffers->ViewportThemeUBO.Update(as_bytes(R.get<const ViewportTheme>(SceneEntity)));
        request(RenderRequest::Submit);
    }
    if (!R.storage<entt::reactive>(changes::Materials).empty()) {
        if (const auto *edit = R.try_get<const MaterialEdit>(SceneEntity);
            edit && edit->Index < GetMaterialCount(*Buffers)) {
            SetMaterial(*Buffers, edit->Index, edit->Value);
        }
        request(RenderRequest::Submit);
    }
    if (!R.storage<entt::reactive>(changes::SceneSettings).empty()) {
        request(RenderRequest::ReRecord);
        dirty_overlay_meshes.merge(scene_selection::GetSelectedMeshEntities(R));
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
                if (scene_selection::HasFrozenInstance(R, mesh_entity)) continue;
                const auto &mesh = R.get<const Mesh>(mesh_entity);
                const auto vertices = mesh.GetVerticesSpan();
                const auto &selection = R.get<const MeshSelection>(mesh_entity);
                if (selection.Handles.empty()) continue;
                const auto &wt = R.get<const WorldTransform>(instance_entity);
                const auto wt_rot = Vec4ToQuat(wt.Rotation);
                const auto inv_rot = glm::conjugate(wt_rot);
                const auto inv_scale = 1.f / wt.Scale;
                const auto vertex_handles = scene_selection::ConvertSelectionElement(selection, mesh, edit_mode, Element::Vertex);
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
            if (const auto *cd = active_entity != entt::null ? R.try_get<Camera>(active_entity) : nullptr) {
                const auto extent = R.get<const ViewportExtent>(SceneEntity).Value;
                const float viewport_aspect = extent.width == 0 || extent.height == 0 ? 1.f : float(extent.width) / float(extent.height);
                R.get<ViewCamera>(SceneEntity).Data = WidenForLookThrough(*cd, viewport_aspect);
            }
        }
        const auto &camera = R.get<const ViewCamera>(SceneEntity);
        const auto &settings = R.get<const SceneSettings>(SceneEntity);
        const auto &mat_preview_lighting = R.get<const MaterialPreviewLighting>(SceneEntity);
        const auto &rendered_lighting = R.get<const RenderedLighting>(SceneEntity);
        const bool is_pbr_mode = settings.ViewportShading == ViewportShadingMode::MaterialPreview || settings.ViewportShading == ViewportShadingMode::Rendered;
        const auto &active_lighting = (settings.ViewportShading == ViewportShadingMode::Rendered) ? static_cast<const PBRViewportLighting &>(rendered_lighting) : static_cast<const PBRViewportLighting &>(mat_preview_lighting);
        const bool use_scene_lights = is_pbr_mode && active_lighting.UseSceneLights;
        const bool use_scene_world = is_pbr_mode && active_lighting.UseSceneWorld;
        const auto &active_environment = use_scene_world ? Environments->SceneWorld : Environments->StudioWorld;
        const float env_intensity = use_scene_world ? 1.f : active_lighting.EnvIntensity;
        const float env_rotation_radians = use_scene_world ? 0.f : active_lighting.EnvRotationDegrees * (Pi / 180.f);
        const float world_opacity = is_pbr_mode ? (use_scene_world ? 1.f : active_lighting.WorldOpacity) : 0.f;
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
            .LightCount = uint32_t(Buffers->LightBuffer.UsedSize / sizeof(PunctualLight)),
            .LightSlot = Buffers->LightBuffer.Slot,
            .UseSceneLightsRender = use_scene_lights ? 1u : 0u,
            .EnvIntensity = env_intensity,
            .EnvRotationRadians = env_rotation_radians,
            .WorldOpacity = world_opacity,
            .Ibl = active_environment.Ibl,
            .InteractionMode = interaction_mode,
            .EditElement = R.get<const SceneEditMode>(SceneEntity).Value,
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
            .MaterialSlot = Buffers->MaterialBuffer.Slot,
            .PrimitiveMaterialSlot = Meshes->PrimitiveMaterialBuffer.Buffer.Slot,
            .FacePrimitiveSlot = Meshes->FacePrimitiveBuffer.Buffer.Slot,
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
                const auto vertex_handles = scene_selection::ConvertSelectionElement(*selection, mesh, edit_mode, Element::Vertex);
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
            for (auto [entity, anim, pose_state, armature] :
                 R.view<const ArmatureAnimation, ArmaturePoseState, Armature>().each()) {
                if (anim.Clips.empty() || anim.ActiveClipIndex >= anim.Clips.size()) continue;
                const auto &clip = anim.Clips[anim.ActiveClipIndex];
                const float clip_time = clip.DurationSeconds > 0 ? std::fmod(eval_seconds, clip.DurationSeconds) : 0.0f;
                for (uint32_t i = 0; i < armature.Bones.size(); ++i) pose_state.BonePoseLocal[i] = armature.Bones[i].RestLocal;
                EvaluateAnimation(clip, clip_time, pose_state.BonePoseLocal);

                auto gpu_span = Buffers->ArmatureDeformBuffer.GetMutable(pose_state.GpuDeformRange);
                ComputeDeformMatrices(armature, pose_state.BonePoseLocal, armature.ImportedSkin->InverseBindMatrices, gpu_span);
                request(RenderRequest::ReRecord);
            }

            // Evaluate morph weight animations
            for (auto [entity, morph_anim, morph_state, mi] :
                 R.view<const MorphWeightAnimation, MorphWeightState, const MeshInstance>().each()) {
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
    R.clear<MeshMaterialAssignment>();
    R.clear<MaterialEdit>();
    R.clear<SubmitDirty>();
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
        iota(0u, scene_selection::GetElementCount(mesh, R.get<const SceneEditMode>(SceneEntity).Value)) | to<std::unordered_set>()
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

void Scene::ApplySelectBehavior(entt::entity entity, MeshInstanceCreateInfo::SelectBehavior behavior) {
    switch (behavior) {
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
    ApplySelectBehavior(instance_entity, info.Select);
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
    R.emplace<Armature>(data_entity);

    const auto entity = R.create();
    R.emplace<ObjectKind>(entity, ObjectType::Armature);
    R.emplace<ArmatureObject>(entity, data_entity);
    SetTransform(R, entity, info.Transform);
    R.emplace<Name>(entity, CreateName(R, info.Name.empty() ? "Armature" : info.Name));

    ApplySelectBehavior(entity, info.Select);
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
    ApplySelectBehavior(entity, info.Select);
    return entity;
}

entt::entity Scene::AddCamera(ObjectCreateInfo info) {
    Camera camera{Perspective{.FieldOfViewRad = glm::radians(60.f), .FarClip = 100.f, .NearClip = 0.01f}};
    const auto entity = CreateExtrasObject({.Data = BuildCameraFrustumMesh(camera)}, ObjectType::Camera, info, "Camera");
    R.emplace<Camera>(entity, camera);
    return entity;
}

entt::entity Scene::AddLight(ObjectCreateInfo info, std::optional<PunctualLight> props) {
    auto light = props.value_or(SceneDefaults::MakePunctualLight(PunctualLightType::Point));
    auto wireframe = BuildLightMesh(light);
    const auto entity = CreateExtrasObject(std::move(wireframe), ObjectType::Light, info, "Light");
    const uint32_t light_index = R.storage<LightIndex>().size();
    R.emplace<LightIndex>(entity, light_index);
    R.emplace<SubmitDirty>(entity);
    R.emplace<LightWireframeDirty>(entity);
    const auto mesh_entity = R.get<MeshInstance>(entity).MeshEntity;
    const auto &ri = R.get<const RenderInstance>(entity);
    light.TransformSlotOffset = {R.get<const ModelsBuffer>(mesh_entity).Buffer.Slot, ri.BufferIndex};
    Buffers->SetLight(light_index, light);
    return entity;
}

std::pair<entt::entity, entt::entity> Scene::AddMesh(MeshData &&data, std::optional<MeshInstanceCreateInfo> info) {
    return AddMesh(Meshes->CreateMesh(std::move(data)), std::move(info));
}

std::pair<entt::entity, entt::entity> Scene::AddMesh(const std::filesystem::path &path, std::optional<MeshInstanceCreateInfo> info) {
    auto result = Meshes->LoadMesh(path);
    if (!result) throw std::runtime_error(result.error());

    if (!result->Materials.empty()) {
        auto &texture_store = *Textures;
        std::unordered_map<std::string, uint32_t> texture_slot_cache;
        const auto resolve_texture_slot = [&](
                                              const std::optional<std::filesystem::path> &source_texture_path,
                                              TextureColorSpace color_space,
                                              std::string_view material_name,
                                              std::string_view texture_label
                                          ) -> uint32_t {
            if (!source_texture_path) return InvalidSlot;
            auto texture_path = *source_texture_path;
            if (texture_path.is_relative()) texture_path = path.parent_path() / texture_path;
            texture_path = texture_path.lexically_normal();

            const auto cache_key = std::format("{}|{}", texture_path.generic_string(), color_space == TextureColorSpace::Srgb ? "sRGB" : "Linear");
            if (const auto it = texture_slot_cache.find(cache_key); it != texture_slot_cache.end()) return it->second;

            std::string encoded;
            try {
                encoded = File::Read(texture_path);
            } catch (const std::exception &e) {
                std::cerr << std::format(
                    "Warning: Failed to read OBJ texture '{}' for material '{}' ({}) in '{}': {}\n",
                    texture_path.string(), material_name, texture_label, path.string(), e.what()
                );
                return InvalidSlot;
            }

            auto texture = CreateTextureEntryFromEncoded(
                Vk,
                Buffers->Ctx,
                *CommandPool,
                *OneShotFence,
                *Slots,
                std::as_bytes(std::span{encoded}),
                texture_path.filename().string(),
                std::format(
                    "{} ({})",
                    texture_path.filename().string(),
                    color_space == TextureColorSpace::Srgb ? "sRGB" : "Linear"
                ),
                color_space,
                vk::SamplerAddressMode::eRepeat,
                vk::SamplerAddressMode::eRepeat,
                SamplerConfig{}
            );
            if (!texture) {
                std::cerr << std::format(
                    "Warning: Failed to decode OBJ texture '{}' for material '{}' ({}) in '{}': {}\n",
                    texture_path.string(), material_name, texture_label, path.string(), texture.error()
                );
                return InvalidSlot;
            }

            const auto sampler_slot = texture->SamplerSlot;
            texture_store.Textures.emplace_back(std::move(*texture));
            texture_slot_cache.emplace(cache_key, sampler_slot);
            return sampler_slot;
        };

        std::vector<uint32_t> scene_material_indices(result->Materials.size(), 0u);
        std::vector<std::string> names;
        names.reserve(result->Materials.size());
        for (uint32_t material_index = 0; material_index < result->Materials.size(); ++material_index) {
            const auto &source = result->Materials[material_index];
            const auto material_name = source.Name.empty() ? std::format("Material{}", material_index) : source.Name;
            const auto base_color_texture = resolve_texture_slot(source.BaseColorTexturePath, TextureColorSpace::Srgb, material_name, "baseColor");
            const auto normal_texture = resolve_texture_slot(source.NormalTexturePath, TextureColorSpace::Linear, material_name, "normal");
            scene_material_indices[material_index] = AppendMaterial(
                *Buffers,
                {
                    .BaseColorFactor = source.BaseColorFactor,
                    .MetallicFactor = std::clamp(source.MetallicFactor, 0.f, 1.f),
                    .RoughnessFactor = std::clamp(source.RoughnessFactor, 0.f, 1.f),
                    .AlphaMode = (source.BaseColorFactor.w < 1.f || source.HasAlphaTexture) ?
                        MaterialAlphaMode::Blend :
                        MaterialAlphaMode::Opaque,
                    .BaseColorTexture = {.Slot = base_color_texture != InvalidSlot ? base_color_texture : Textures->WhiteTextureSlot},
                    .NormalTexture = {.Slot = normal_texture},
                }
            );
            names.emplace_back(material_name);
        }
        R.patch<MaterialStore>(
            SceneEntity,
            [&](auto &material_store) {
                material_store.Names.insert(material_store.Names.end(), std::make_move_iterator(names.begin()), std::make_move_iterator(names.end()));
            }
        );

        auto primitive_materials = Meshes->GetPrimitiveMaterialIndices(result->Mesh.GetStoreId());
        if (!primitive_materials.empty()) {
            const auto fallback = scene_material_indices.front();
            for (auto &primitive_material : primitive_materials) {
                primitive_material = primitive_material < scene_material_indices.size() ?
                    scene_material_indices[primitive_material] :
                    fallback;
            }
        }
    }

    const auto e = AddMesh(std::move(result->Mesh), std::move(info));
    R.emplace<Path>(e.first, path);
    return e;
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
                auto &dst = R.get<Armature>(R.get<ArmatureObject>(copy_entity).Entity);
                dst = R.get<const Armature>(src_armature->Entity);
            }
            return copy_entity;
        }
        return AddEmpty(create_info);
    }

    // Object extras (Camera, Empty, Light) have MeshInstance but create their own wireframe mesh.
    if (R.all_of<ObjectExtrasTag>(R.get<MeshInstance>(e).MeshEntity)) {
        if (const auto *src_cd = R.try_get<Camera>(e)) {
            const auto copy_entity = AddCamera(create_info);
            R.replace<Camera>(copy_entity, *src_cd);
            return copy_entity;
        }
        if (R.all_of<LightIndex>(e)) {
            const auto copy_entity = AddLight(create_info, Buffers->GetLight(R.get<const LightIndex>(e).Value));
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
            R.emplace<ArmatureObject>(e_new, armature->Entity);
            SetTransform(R, e_new, info ? info->Transform : GetTransform(R, e));

            ApplySelectBehavior(e_new, select_behavior);
            return e_new;
        }

        return AddEmpty({
            .Name = info && !info->Name.empty() ? info->Name : std::format("{}_copy", GetName(R, e)),
            .Transform = info ? info->Transform : GetTransform(R, e),
            .Select = select_behavior,
        });
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

    auto &texture_store = *Textures;
    if (texture_store.Textures.size() > 1) {
        const auto imported_textures = std::span<const TextureEntry>{texture_store.Textures}.subspan(1);
        const auto imported_sampler_slots = CollectSamplerSlots(imported_textures);
        ReleaseSamplerSlots(*Slots, imported_sampler_slots);
        texture_store.Textures.erase(texture_store.Textures.begin() + 1, texture_store.Textures.end());
    }
    texture_store.WhiteTextureSlot = texture_store.Textures.empty() ? InvalidSlot : texture_store.Textures.front().SamplerSlot;

    if (GetMaterialCount(*Buffers) > 1) SetMaterialCount(*Buffers, 1u);
    R.patch<MaterialStore>(
        SceneEntity,
        [](auto &material_store) {
            if (material_store.Names.size() > 1) material_store.Names.erase(material_store.Names.begin() + 1, material_store.Names.end());
        }
    );
}

void Scene::SetMeshPositions(entt::entity e, std::span<const vec3> positions) {
    if (scene_selection::HasFrozenInstance(R, e)) return;

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
    if (const auto *armature = R.try_get<ArmatureObject>(e); armature && R.valid(armature->Entity)) {
        armature_data_entities.emplace_back(armature->Entity);
    }
    if (const auto *armature_modifier = R.try_get<ArmatureModifier>(e); armature_modifier && R.valid(armature_modifier->ArmatureEntity)) {
        armature_data_entities.emplace_back(armature_modifier->ArmatureEntity);
    }
    if (const auto *bone_attachment = R.try_get<BoneAttachment>(e); bone_attachment && R.valid(bone_attachment->ArmatureEntity)) {
        armature_data_entities.emplace_back(bone_attachment->ArmatureEntity);
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
        const uint32_t last_index = R.storage<LightIndex>().size() - 1u;
        if (remove_index != last_index) {
            Buffers->SetLight(remove_index, Buffers->GetLight(last_index));
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
            [armature_data_entity](const auto &entry) { return std::get<1>(entry).Entity == armature_data_entity; }
        );
        const auto used_by_armature_modifier = any_of(
            R.view<ArmatureModifier>().each(),
            [armature_data_entity](const auto &entry) { return std::get<1>(entry).ArmatureEntity == armature_data_entity; }
        );
        const auto used_by_bone_attachment = any_of(
            R.view<BoneAttachment>().each(),
            [armature_data_entity](const auto &entry) { return std::get<1>(entry).ArmatureEntity == armature_data_entity; }
        );

        if (!(used_by_armature_object || used_by_armature_modifier || used_by_bone_attachment)) {
            R.destroy(armature_data_entity);
        }
    }
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
    const bool is_wireframe_mode = settings.ViewportShading == ViewportShadingMode::Wireframe;
    const bool show_rendered = settings.ViewportShading == ViewportShadingMode::MaterialPreview || settings.ViewportShading == ViewportShadingMode::Rendered;
    const bool show_fill = !is_wireframe_mode;
    const SPT fill_pipeline = settings.FaceColorMode == FaceColorMode::Mesh ? SPT::Fill : SPT::DebugNormals;
    const auto primary_edit_instances = is_edit_mode ? scene_selection::ComputePrimaryEditInstances(R) : std::unordered_map<entt::entity, entt::entity>{};
    const bool has_pending_transform = is_edit_mode && R.all_of<PendingTransform>(SceneEntity);
    const auto edit_transform_context = is_edit_mode ? EditTransformContext{scene_selection::ComputePrimaryEditInstances(R, false)} : EditTransformContext{};
    const auto is_silhouette_eligible = [&](entt::entity e) {
        if (!R.all_of<MeshInstance, RenderInstance>(e)) return false;
        const auto mesh_entity = R.get<const MeshInstance>(e).MeshEntity;
        if (!R.valid(mesh_entity) || R.all_of<ObjectExtrasTag>(mesh_entity)) return false;
        const auto *mesh_buffers = R.try_get<const MeshBuffers>(mesh_entity);
        return mesh_buffers && mesh_buffers->FaceIndices.Count > 0;
    };

    std::unordered_set<entt::entity> silhouette_instances;
    if (is_edit_mode) {
        for (const auto [e, mi, ri] : R.view<const MeshInstance, const Selected, const RenderInstance>().each()) {
            if (!is_silhouette_eligible(e)) continue;
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

    std::vector<entt::entity> blend_mesh_order;
    if (show_rendered) {
        // Transparent pass ordering: sort mesh draws back-to-front by camera distance.
        // This is a mesh-level approximation; interpenetrating transparent geometry may still require
        // per-primitive sorting or OIT for fully correct compositing.
        const auto camera_position = R.get<const ViewCamera>(SceneEntity).Position();
        std::unordered_map<entt::entity, float> farthest_distance2_by_mesh;
        farthest_distance2_by_mesh.reserve(R.storage<RenderInstance>().size());
        for (const auto [entity, _, wt] : R.view<const RenderInstance, const WorldTransform>().each()) {
            entt::entity mesh_entity = entity;
            if (const auto *mesh_instance = R.try_get<const MeshInstance>(entity)) mesh_entity = mesh_instance->MeshEntity;
            if (!R.valid(mesh_entity) || !R.all_of<Mesh>(mesh_entity)) continue;
            const auto delta = wt.Position - camera_position;
            const auto distance2 = dot(delta, delta);
            if (const auto it = farthest_distance2_by_mesh.find(mesh_entity); it != farthest_distance2_by_mesh.end()) {
                it->second = std::max(it->second, distance2);
            } else {
                farthest_distance2_by_mesh.emplace(mesh_entity, distance2);
            }
        }
        blend_mesh_order.reserve(farthest_distance2_by_mesh.size());
        for (const auto &[mesh_entity, _] : farthest_distance2_by_mesh) blend_mesh_order.emplace_back(mesh_entity);
        std::ranges::sort(
            blend_mesh_order,
            [&](const auto a, const auto b) {
                return farthest_distance2_by_mesh.at(a) > farthest_distance2_by_mesh.at(b);
            }
        );
    }

    // Build mesh_entity -> deform slots mapping for skinned meshes (edit mode shows rest pose)
    const auto mesh_deform_slots = is_edit_mode ? std::unordered_map<entt::entity, DeformSlots>{} : BuildDeformSlots(R, *Meshes);
    const auto get_deform_slots = [&](entt::entity mesh_entity) -> DeformSlots {
        if (auto it = mesh_deform_slots.find(mesh_entity); it != mesh_deform_slots.end()) return it->second;
        return {};
    };

    const bool has_object_silhouette_selection = any_of(
        R.view<const Selected, const MeshInstance, const RenderInstance>().each(),
        [&](const auto &entry) { return is_silhouette_eligible(std::get<0>(entry)); }
    );
    const bool render_silhouette = interaction_mode == InteractionMode::Object ? has_object_silhouette_selection : !silhouette_instances.empty();

    DrawListBuilder draw_list;
    DrawBatchInfo fill_batch_opaque{}, fill_batch_blend{}, line_batch{}, point_batch{};
    DrawBatchInfo extras_line_batch{}, silhouette_batch{};
    DrawBatchInfo overlay_face_normals_batch{}, overlay_vertex_normals_batch{}, overlay_bbox_batch{};
    const auto patch_edit_pending_local_transform = [&](size_t draws_before, entt::entity mesh_entity) {
        if (!has_pending_transform) return;
        const auto context_it = edit_transform_context.TransformInstances.find(mesh_entity);
        if (context_it == edit_transform_context.TransformInstances.end()) return;
        const auto *primary_ri = R.try_get<const RenderInstance>(context_it->second);
        if (!primary_ri) return;
        for (size_t i = draws_before; i < draw_list.Draws.size(); ++i) {
            draw_list.Draws[i].HasPendingVertexTransform = 1u;
            draw_list.Draws[i].PrimaryEditInstanceIndex = primary_ri->BufferIndex;
        }
    };

    if (render_silhouette) {
        silhouette_batch = draw_list.BeginBatch();
        auto append_silhouette = [&](entt::entity e) {
            if (!is_silhouette_eligible(e)) return;
            const auto mesh_entity = R.get<MeshInstance>(e).MeshEntity;
            const auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
            const auto &models = R.get<ModelsBuffer>(mesh_entity);
            const auto deform = get_deform_slots(mesh_entity);
            auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models, deform.BoneDeformOffset, deform.ArmatureDeformOffset, deform.MorphDeformOffset, deform.MorphTargetCount);
            draw.ObjectIdSlot = models.ObjectIds.Slot;
            const auto draws_before = draw_list.Draws.size();
            AppendDraw(draw_list, silhouette_batch, mesh_buffers.FaceIndices, models, draw, R.get<RenderInstance>(e).BufferIndex);
            PatchMorphWeights(draw_list, draws_before, deform);
            patch_edit_pending_local_transform(draws_before, mesh_entity);
        };
        if (is_edit_mode) {
            for (const auto e : silhouette_instances) append_silhouette(e);
        } else {
            for (const auto e : R.view<Selected, RenderInstance>()) append_silhouette(e);
        }
    }

    if (show_fill) {
        const auto append_fill_phase = [&](DrawBatchInfo &batch, std::optional<bool> blend_target) {
            batch = draw_list.BeginBatch();
            const auto append_fill_mesh = [&](entt::entity entity, const MeshBuffers &mesh_buffers, const ModelsBuffer &models, const Mesh &mesh) {
                if (mesh.FaceCount() == 0) return;
                const auto deform = get_deform_slots(entity);
                auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models, deform.BoneDeformOffset, deform.ArmatureDeformOffset, deform.MorphDeformOffset, deform.MorphTargetCount);
                const auto face_id_buffer = Meshes->GetFaceIdRange(mesh.GetStoreId());
                const auto face_first_tri = Meshes->GetFaceFirstTriRange(mesh.GetStoreId());
                const auto face_state_buffer = Meshes->GetFaceStateRange(mesh.GetStoreId());
                const auto face_primitive_buffer = Meshes->GetFacePrimitiveRange(mesh.GetStoreId());
                const auto primitive_material_buffer = Meshes->GetPrimitiveMaterialRange(mesh.GetStoreId());
                draw.ObjectIdSlot = face_id_buffer.Slot;
                draw.FaceIdOffset = face_id_buffer.Offset;
                draw.FaceFirstTriOffset = settings.SmoothShading ? InvalidOffset : face_first_tri.Offset;
                draw.FacePrimitiveOffset = face_primitive_buffer.Count > 0 ? face_primitive_buffer.Offset : InvalidOffset;
                draw.PrimitiveMaterialOffset = primitive_material_buffer.Count > 0 ? primitive_material_buffer.Offset : InvalidOffset;
                const auto append_fill_draw = [&](const DrawData &draw, uint32_t index_count, std::optional<uint32_t> model_index) {
                    const auto db = draw_list.Draws.size();
                    AppendDraw(draw_list, batch, index_count, models, draw, model_index);
                    PatchMorphWeights(draw_list, db, deform);
                    patch_edit_pending_local_transform(db, entity);
                };
                const auto append_fill_for_instances = [&](const DrawData &draw, uint32_t index_count) {
                    if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                        // Draw primary with element state first, then all without (depth LESS won't overwrite)
                        auto primary_draw = draw;
                        primary_draw.ElementStateSlotOffset = face_state_buffer;
                        append_fill_draw(primary_draw, index_count, R.get<RenderInstance>(it->second).BufferIndex);
                        auto other_draw = draw;
                        other_draw.ElementStateSlotOffset = {};
                        append_fill_draw(other_draw, index_count, std::nullopt);
                    } else {
                        auto all_draw = draw;
                        all_draw.ElementStateSlotOffset = face_state_buffer;
                        append_fill_draw(all_draw, index_count, std::nullopt);
                    }
                };

                if (show_rendered) {
                    const auto primitive_materials = Meshes->GetPrimitiveMaterialIndices(mesh.GetStoreId());
                    const auto face_primitives = Meshes->GetFacePrimitiveIndices(mesh.GetStoreId());
                    const auto triangle_face_ids = Meshes->GetTriangleFaceIds(mesh.GetStoreId());
                    const auto material_count = GetMaterialCount(*Buffers);
                    const auto material_is_blend = [&](uint32_t material_index) {
                        return material_index < material_count &&
                            GetMaterial(*Buffers, material_index).AlphaMode == MaterialAlphaMode::Blend;
                    };
                    const uint32_t triangle_count = mesh_buffers.FaceIndices.Count / 3u;
                    if (!primitive_materials.empty() &&
                        face_primitives.size() == mesh.FaceCount() &&
                        triangle_face_ids.size() == triangle_count &&
                        triangle_count > 0u) {
                        const auto triangle_is_blend = [&](uint32_t triangle_index) {
                            const auto face_id = triangle_face_ids[triangle_index];
                            if (face_id == 0u || face_id > face_primitives.size()) return false;
                            auto primitive_index = face_primitives[face_id - 1u];
                            if (primitive_index >= primitive_materials.size()) primitive_index = primitive_materials.size() - 1u;
                            return material_is_blend(primitive_materials[primitive_index]);
                        };

                        struct BlendDrawRange {
                            bool Blend{false};
                            uint32_t FirstTriangle{0};
                            uint32_t TriangleCount{0};
                        };
                        std::vector<BlendDrawRange> blend_ranges;
                        blend_ranges.reserve(16);

                        auto active_blend = triangle_is_blend(0u);
                        auto first_triangle = 0u;
                        for (uint32_t tri = 1u; tri < triangle_count; ++tri) {
                            const auto tri_blend = triangle_is_blend(tri);
                            if (tri_blend == active_blend) continue;
                            blend_ranges.emplace_back(
                                BlendDrawRange{
                                    .Blend = active_blend,
                                    .FirstTriangle = first_triangle,
                                    .TriangleCount = tri - first_triangle,
                                }
                            );
                            active_blend = tri_blend;
                            first_triangle = tri;
                        }
                        blend_ranges.emplace_back(
                            BlendDrawRange{
                                .Blend = active_blend,
                                .FirstTriangle = first_triangle,
                                .TriangleCount = triangle_count - first_triangle,
                            }
                        );

                        for (const auto &range : blend_ranges) {
                            if (blend_target && range.Blend != *blend_target) continue;
                            if (range.TriangleCount == 0u) continue;
                            auto range_draw = draw;
                            range_draw.IndexSlotOffset.Offset += range.FirstTriangle * 3u;
                            range_draw.FaceIdOffset += range.FirstTriangle;
                            append_fill_for_instances(range_draw, range.TriangleCount * 3u);
                        }
                        return;
                    }
                }

                if (!blend_target || !*blend_target) append_fill_for_instances(draw, mesh_buffers.FaceIndices.Count);
            };
            if (blend_target && *blend_target && !blend_mesh_order.empty()) {
                for (const auto entity : blend_mesh_order) {
                    if (!R.valid(entity) || !R.all_of<MeshBuffers, ModelsBuffer, Mesh>(entity)) continue;
                    append_fill_mesh(entity, R.get<const MeshBuffers>(entity), R.get<const ModelsBuffer>(entity), R.get<const Mesh>(entity));
                }
            } else {
                for (const auto [entity, mesh_buffers, models, mesh] : R.view<const MeshBuffers, const ModelsBuffer, const Mesh>().each()) {
                    append_fill_mesh(entity, mesh_buffers, models, mesh);
                }
            }
        };

        if (show_rendered) {
            append_fill_phase(fill_batch_opaque, false);
            append_fill_phase(fill_batch_blend, true);
        } else {
            append_fill_phase(fill_batch_opaque, std::nullopt);
        }
    }

    {
        line_batch = draw_list.BeginBatch();
        for (auto [entity, mesh_buffers, models, mesh] : R.view<MeshBuffers, ModelsBuffer, Mesh>().each()) {
            if (R.all_of<ObjectExtrasTag>(entity)) continue;
            if (mesh_buffers.EdgeIndices.Count == 0) continue;
            const bool is_line_mesh = mesh.FaceCount() == 0 && mesh.EdgeCount() > 0;
            if (!is_line_mesh && !is_edit_mode && !is_excite_mode && !is_wireframe_mode) continue;
            const auto deform = get_deform_slots(entity);
            auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.EdgeIndices, models, deform.BoneDeformOffset, deform.ArmatureDeformOffset, deform.MorphDeformOffset, deform.MorphTargetCount);
            const auto edge_state_buffer = Meshes->GetEdgeStateRange(mesh.GetStoreId());
            draw.ElementStateSlotOffset = edge_state_buffer;
            const auto db = draw_list.Draws.size();
            if (is_line_mesh || is_wireframe_mode) {
                AppendDraw(draw_list, line_batch, mesh_buffers.EdgeIndices, models, draw);
            } else if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                AppendDraw(draw_list, line_batch, mesh_buffers.EdgeIndices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
            } else if (excitable_mesh_entities.contains(entity)) {
                AppendDraw(draw_list, line_batch, mesh_buffers.EdgeIndices, models, draw);
            }
            PatchMorphWeights(draw_list, db, deform);
            patch_edit_pending_local_transform(db, entity);
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
            const auto db = draw_list.Draws.size();
            if (is_point_mesh) {
                AppendDraw(draw_list, point_batch, mesh_buffers.VertexIndices, models, draw);
            } else if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                AppendDraw(draw_list, point_batch, mesh_buffers.VertexIndices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
            } else if (excitable_mesh_entities.contains(entity)) {
                AppendDraw(draw_list, point_batch, mesh_buffers.VertexIndices, models, draw);
            }
            PatchMorphWeights(draw_list, db, deform);
            patch_edit_pending_local_transform(db, entity);
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

    const bool has_silhouette = render_silhouette && silhouette_batch.DrawCount > 0;
    if (has_silhouette) { // Silhouette depth/object pass
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

    // Background environment (PBR modes only; shader discards when WorldOpacity == 0 or no env slot)
    if (show_rendered) main.Renderer.ShaderPipelines.at(SPT::Background).RenderQuad(cb);

    // Silhouette edge depth (not color! we render it before mesh depth to avoid overwriting closer depths with further ones)
    if (has_silhouette) {
        const auto &silhouette_depth = main.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeDepth);
        const uint32_t depth_sampler_index = SelectionHandles->DepthSampler;
        cb.pushConstants(*silhouette_depth.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(depth_sampler_index), &depth_sampler_index);
        silhouette_depth.RenderQuad(cb);
    }

    { // Meshes
        cb.bindIndexBuffer(*Buffers->IdentityIndexBuffer, 0, vk::IndexType::eUint32);
        // Solid faces
        if (show_fill) {
            if (show_rendered) {
                record_draw_batch(main.Renderer, SPT::PBRFill, fill_batch_opaque);
                record_draw_batch(main.Renderer, SPT::PBRFillBlend, fill_batch_blend);
            } else {
                record_draw_batch(main.Renderer, fill_pipeline, fill_batch_opaque);
            }
        }
        // Wireframe edges (always recorded — batch is empty when nothing qualifies)
        record_draw_batch(main.Renderer, SPT::Line, line_batch);
        // Vertex points (always recorded — batch is empty when nothing qualifies)
        record_draw_batch(main.Renderer, SPT::Point, point_batch);
        // Object extras (cameras, lights, empties)
        record_draw_batch(main.Renderer, SPT::ObjectExtrasLine, extras_line_batch);
    }

    // Silhouette edge color (rendered ontop of meshes)
    if (has_silhouette) {
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
        scene_selection::ComputePrimaryEditInstances(R) :
        std::unordered_map<entt::entity, entt::entity>{};
    const bool is_edit_mode = R.get<const SceneInteraction>(SceneEntity).Mode == InteractionMode::Edit;
    const auto selection_deform_slots = is_edit_mode ? std::unordered_map<entt::entity, DeformSlots>{} : BuildDeformSlots(R, *Meshes);
    const auto get_deform_slots = [&](entt::entity mesh_entity) -> DeformSlots {
        if (auto it = selection_deform_slots.find(mesh_entity); it != selection_deform_slots.end()) return it->second;
        return {};
    };

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
                    const auto deform = get_deform_slots(mesh_entity);
                    auto draw = MakeDrawData(mesh_buffers.Vertices, *indices, models, deform.BoneDeformOffset, deform.ArmatureDeformOffset, deform.MorphDeformOffset, deform.MorphTargetCount);
                    draw.ObjectIdSlot = models.ObjectIds.Slot;
                    const auto db = draw_list.Draws.size();
                    if (auto it = primary_edit_instances.find(mesh_entity); it != primary_edit_instances.end()) {
                        AppendDraw(draw_list, batch, *indices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
                    } else {
                        AppendDraw(draw_list, batch, *indices, models, draw);
                    }
                    PatchMorphWeights(draw_list, db, deform);
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
    const Timer timer{"RenderEditSelectionPass"};
    RenderElementSelectionPass(ranges, element, false, {}, {}, signal_semaphore);
}

void Scene::RenderElementSelectionPass(
    std::span<const ElementRange> ranges, Element element, bool write_bitset,
    uvec2 box_min, uvec2 box_max, vk::Semaphore signal_semaphore
) {
    if (ranges.empty() || element == Element::None) return;

    const auto primary_edit_instances = scene_selection::ComputePrimaryEditInstances(R);
    const bool xray_selection = SelectionXRay;
    const auto element_pipeline = [xray_selection](Element el) -> SPT {
        if (el == Element::Vertex) return xray_selection ? SPT::SelectionElementVertexXRay : SPT::SelectionElementVertex;
        if (el == Element::Edge) return xray_selection ? SPT::SelectionElementEdgeXRay : SPT::SelectionElementEdge;
        return xray_selection ? SPT::SelectionElementFaceXRay : SPT::SelectionElementFace;
    };

    DrawListBuilder draw_list;
    DrawBatchInfo silhouette_batch{};
    const bool render_depth = !xray_selection;
    if (render_depth) {
        silhouette_batch = draw_list.BeginBatch();
        // For face selection, faces self-sort by depth (eLess + depthWrite against cleared depth 1.0).
        // Drawing the silhouette would fill depth with mesh surface depth, causing face fragments at equal
        // depth to fail the eLess test. For vertex/edge selection, we do render silhouette geometry so that
        // elements at the surface pass (eLessOrEqual) while occluded elements behind it fail.
        if (element != Element::Face) {
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
    auto element_batch = draw_list.BeginBatch();
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
            AppendDraw(draw_list, element_batch, indices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
        } else {
            AppendDraw(draw_list, element_batch, indices, models, draw);
        }
    }

    if (!draw_list.Draws.empty()) Buffers->SelectionDrawData.Update(as_bytes(draw_list.Draws));
    if (!draw_list.IndirectCommands.empty()) Buffers->SelectionIndirect.Update(as_bytes(draw_list.IndirectCommands));
    Buffers->EnsureIdentityIndexBuffer(draw_list.MaxIndexCount);
    if (auto descriptor_updates = Buffers->Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
        Vk.Device.updateDescriptorSets(std::move(descriptor_updates), {});
        Buffers->Ctx.ClearDeferredDescriptorUpdates();
    }
    Buffers->SceneViewUBO.Update(as_bytes(Buffers->SelectionDrawData.Slot), offsetof(SceneViewUBO, DrawDataSlot));

    auto cb = *ClickCommandBuffer;
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    if (!write_bitset) {
        // Reset linked-list state before writing selection fragments.
        Buffers->SelectionCounterBuffer.Write(as_bytes(SelectionCounters{}));

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
    }

    const auto extent = R.get<const ViewportExtent>(SceneEntity).Value;
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(extent.width), float(extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, extent});

    if (render_depth) {
        const auto &silhouette = Pipelines->Silhouette;
        static const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {Transparent}};
        const vk::Rect2D sil_rect{{0, 0}, ToExtent2D(silhouette.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*silhouette.Renderer.RenderPass, *silhouette.Resources->Framebuffer, sil_rect, clear_values}, vk::SubpassContents::eInline);
        cb.bindIndexBuffer(*Buffers->IdentityIndexBuffer, 0, vk::IndexType::eUint32);
        if (silhouette_batch.DrawCount > 0) {
            const auto &pipeline = silhouette.Renderer.Bind(cb, SPT::SilhouetteDepthObject);
            const DrawPassPushConstants sil_pc{silhouette_batch.DrawDataSlotOffset, InvalidSlot, SelectionHandles->HeadImage, Buffers->SelectionNodeBuffer.Slot, SelectionHandles->SelectionCounter};
            cb.pushConstants(*pipeline.PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(sil_pc), &sil_pc);
            cb.drawIndexedIndirect(*Buffers->SelectionIndirect, silhouette_batch.IndirectOffset, silhouette_batch.DrawCount, sizeof(vk::DrawIndexedIndirectCommand));
        }
        cb.endRenderPass();
    }

    const auto &selection = Pipelines->SelectionFragment;
    const vk::Rect2D sel_rect{{0, 0}, ToExtent2D(Pipelines->Silhouette.Resources->DepthImage.Extent)};
    cb.beginRenderPass({*selection.Renderer.RenderPass, *selection.Resources->Framebuffer, sel_rect, {}}, vk::SubpassContents::eInline);
    cb.bindIndexBuffer(*Buffers->IdentityIndexBuffer, 0, vk::IndexType::eUint32);
    if (element_batch.DrawCount > 0) {
        const SelectionElementPushConstants element_pc{
            element_batch.DrawDataSlotOffset,
            InvalidSlot,
            SelectionHandles->HeadImage,
            Buffers->SelectionNodeBuffer.Slot,
            SelectionHandles->SelectionCounter,
            box_min.x,
            box_min.y,
            box_max.x,
            box_max.y,
            SelectionHandles->SelectionBitset,
            write_bitset ? (SelectionElementFlags::OutputBitset | SelectionElementFlags::ClipToBox) : SelectionElementFlags::None,
        };
        auto draw_with = [&](SPT spt) {
            const auto &pipeline = selection.Renderer.Bind(cb, spt);
            cb.pushConstants(*pipeline.PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(element_pc), &element_pc);
            cb.drawIndexedIndirect(*Buffers->SelectionIndirect, element_batch.IndirectOffset, element_batch.DrawCount, sizeof(vk::DrawIndexedIndirectCommand));
        };
        draw_with(element_pipeline(element));
        if (write_bitset && xray_selection) {
            // X-Ray face: point pass catches edge-on faces (zero projected triangle area).
            if (element == Element::Face) draw_with(SPT::SelectionElementFaceXRayVerts);
            // X-Ray edge: point pass catches near/zero-length projected edges.
            if (element == Element::Edge) draw_with(SPT::SelectionElementEdgeXRayVerts);
        }
    }
    cb.endRenderPass();

    if (write_bitset) {
        // Ensure fragment shader writes to the bitset are visible to the host after the fence.
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eHost, {}, {},
            vk::BufferMemoryBarrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead, {}, {}, *Buffers->BoxSelectBitsetBuffer, 0, VK_WHOLE_SIZE},
            {}
        );
    }

    cb.end();
    vk::SubmitInfo submit;
    submit.setCommandBuffers(cb);
    if (signal_semaphore) submit.setSignalSemaphores(signal_semaphore);
    Vk.Queue.submit(submit, *OneShotFence);
    WaitFor(*OneShotFence, Vk.Device);

    // Element selection pass overwrites the shared head image used for object selection.
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

    // Box-select writes element IDs directly from the selection fragment shader.
    // This avoids linked-list overflow in heavy X-Ray overlap while keeping the same
    // depth semantics as the element render pipeline in non-X-Ray mode.
    const std::span<const uint32_t> zero_bits{BoxSelectZeroBits.data(), bitset_words};
    Buffers->BoxSelectBitsetBuffer.Write(std::as_bytes(zero_bits));
    RenderElementSelectionPass(ranges, element, true, box_min, box_max);

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

void Scene::Render(vk::Fence viewportConsumerFence) {
    auto &dl = *ImGui::GetWindowDrawList();
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
        const auto p = ImGui::GetCursorScreenPos();
        const auto extent = R.get<const ViewportExtent>(SceneEntity).Value;
        const auto &t = *ViewportTexture;
        dl.AddImage(ImTextureID(VkDescriptorSet(t.DescriptorSet)), p, {p.x + float(extent.width), p.y + float(extent.height)}, {t.Uv0.x, t.Uv0.y}, {t.Uv1.x, t.Uv1.y});
    }
    dl.ChannelsMerge();
}

bool Scene::SubmitViewport(vk::Fence viewportConsumerFence) {
    auto &extent = R.get<ViewportExtent>(SceneEntity).Value;
    const auto content_region = ImGui::GetContentRegionAvail();
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

void Scene::SnapToCamera(entt::entity camera_entity) {
    const auto &wt = R.get<WorldTransform>(camera_entity);
    const vec3 pos = wt.Position;
    const vec3 fwd = -glm::normalize(glm::rotate(Vec4ToQuat(wt.Rotation), vec3{0.f, 0.f, 1.f}));
    R.replace<ViewCamera>(SceneEntity, ViewCamera{pos, pos + fwd, R.get<Camera>(camera_entity)});
}
