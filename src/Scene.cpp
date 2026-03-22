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
#include "Instance.h"
#include "MeshComponents.h"
#include "NodeTransformAnimation.h"
#include "ScenePipelines.h"
#include "SceneSelection.h"
#include "Shader.h"
#include "SvgResource.h"
#include "Timer.h"
#include "gpu/BoxSelectPushConstants.h"
#include "gpu/ElementPickPushConstants.h"
#include "gpu/ObjectPickPushConstants.h"
#include "gpu/SelectionElementPushConstants.h"
#include "gpu/SilhouetteEdgeColorPushConstants.h"
#include "gpu/SilhouetteEdgeDepthObjectPushConstants.h"
#include "gpu/UpdateSelectionStatePushConstants.h"
#include "mesh/MeshAttributes.h"
#include "mesh/MeshData.h"
#include "mesh/MeshStore.h"
#include "mesh/PrimitiveType.h"
#include "mesh/Primitives.h"
#include "scene_impl/SceneInternalTypes.h"

#include "imgui.h"

#include "AxisColors.h" // Must be after imgui.h
#include <entt/entity/registry.hpp>

#include "Variant.h"
#include <bit>
#include <iostream>

using std::ranges::any_of, std::ranges::all_of, std::ranges::find, std::ranges::fold_left, std::ranges::to;
using std::views::filter, std::views::iota, std::views::transform;

entt::entity FindArmatureObject(const entt::registry &r, entt::entity e) {
    if (e == entt::null) return entt::null;
    if (r.all_of<ArmatureObject>(e)) return e;
    if (const auto *sub = r.try_get<SubElementOf>(e); sub && r.all_of<ArmatureObject>(sub->Parent)) return sub->Parent;
    return entt::null;
}

entt::entity FindActiveBone(const entt::registry &r) {
    entt::entity result = entt::null;
    for (const auto e : r.view<BoneActive>()) {
        assert(result == entt::null && "Multiple BoneActive entities");
        result = e;
    }
    return result;
}

namespace {
mat4 RestLocalToMatrix(const Transform &t) { return glm::translate(I4, t.P) * glm::mat4_cast(glm::normalize(t.R)) * glm::scale(I4, t.S); }
constexpr vk::Extent2D ToExtent2D(vk::Extent3D extent) { return {extent.width, extent.height}; }
const vk::ClearColorValue Transparent{0, 0, 0, 0};

using namespace he;

void WaitFor(vk::Fence fence, vk::Device device) {
    if (auto wait_result = device.waitForFences(fence, VK_TRUE, UINT64_MAX); wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    device.resetFences(fence);
}

vk::Extent2D ComputeRenderExtentPx(vk::Extent2D logical_extent) {
    const auto scale = ImGui::GetIO().DisplayFramebufferScale;
    const auto scaled_dim = [](uint32_t logical, float s) -> uint32_t {
        if (logical == 0u) return 0u;
        const float scale_value = s > 0.0f ? s : 1.0f;
        return std::max(1u, uint32_t(float(logical) * scale_value + 0.5f));
    };
    return {scaled_dim(logical_extent.width, scale.x), scaled_dim(logical_extent.height, scale.y)};
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
        const auto buffer_entity = r.get<RenderInstance>(e).Entity;
        auto display_wt = wt;
        if (const auto *ds = r.try_get<BoneDisplayScale>(e)) display_wt.Scale = vec3{ds->Value};
        r.patch<ModelsBuffer>(buffer_entity, [&display_wt, i](auto &mb) { mb.Buffer.Update(as_bytes(display_wt), *i * sizeof(WorldTransform)); });

        // Update joint sphere transforms for bone entities
        if (const auto *joints = r.try_get<const BoneJointEntities>(e); joints && r.all_of<BoneDisplayScale>(e)) {
            const float bone_length = r.get<BoneDisplayScale>(e).Value;
            const float sphere_scale = bone_length * 0.06f;

            // Head joint: at bone's world position
            if (joints->Head != entt::null) {
                if (const auto *ri = r.try_get<const RenderInstance>(joints->Head)) {
                    const WorldTransform head_wt{wt.Position, {0, 0, 0, 1}, vec3{sphere_scale}};
                    r.patch<ModelsBuffer>(ri->Entity, [&head_wt, ri](auto &mb) {
                        mb.Buffer.Update(as_bytes(head_wt), ri->BufferIndex * sizeof(WorldTransform));
                    });
                }
            }
            // Tail joint: at bone's tip position (head + rotated (0, bone_length, 0))
            if (joints->Tail != entt::null) {
                if (const auto *ri = r.try_get<const RenderInstance>(joints->Tail)) {
                    const vec3 tail_pos = wt.Position + glm::quat(wt.Rotation.w, wt.Rotation.x, wt.Rotation.y, wt.Rotation.z) * vec3{0, bone_length, 0};
                    const WorldTransform tail_wt{tail_pos, {0, 0, 0, 1}, vec3{sphere_scale}};
                    r.patch<ModelsBuffer>(ri->Entity, [&tail_wt, ri](auto &mb) {
                        mb.Buffer.Update(as_bytes(tail_wt), ri->BufferIndex * sizeof(WorldTransform));
                    });
                }
            }
        }
    }
}

struct ExtrasWireframe {
    MeshData Data;
    std::vector<uint8_t> VertexClasses{}; // Empty means all VCLASS_NONE (no buffer needed).

    uint32_t AddVertex(vec3 pos, uint8_t vclass) {
        const uint32_t i = Data.Positions.size();
        Data.Positions.emplace_back(pos);
        VertexClasses.emplace_back(vclass);
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
    const auto &mesh = r.get<Mesh>(r.get<Instance>(instance_entity).Entity);
    const auto local_pos = ComputeElementLocalPosition(mesh, element, handle);
    const auto &wt = r.get<WorldTransform>(instance_entity);
    return {wt.Position + glm::rotate(Vec4ToQuat(wt.Rotation), wt.Scale * local_pos)};
}

Transform ComposeWorldTransform(const Transform &parent, const Transform &local) {
    return {.P = parent.P + glm::rotate(parent.R, parent.S * local.P), .R = glm::normalize(parent.R * local.R), .S = parent.S * local.S};
}

struct EditTransformContext {
    std::unordered_map<entt::entity, entt::entity> TransformInstances; // excludes frozen, for transforms
};

void ResetObjectPickKeys(SceneBuffers &buffers) {
    auto bytes = buffers.ObjectPickKeyBuffer.GetMappedData();
    std::fill_n(reinterpret_cast<uint32_t *>(bytes.data()), SceneBuffers::MaxSelectableObjects, std::numeric_limits<uint32_t>::max());
}

// clang-format off
namespace changes {
struct Selected {}; struct ActiveInstance {}; struct BoneSelection {}; struct Rerecord {};
struct MeshActiveElement {}; struct MeshGeometry {}; struct MeshMaterial {};
struct Excitable {}; struct ExcitedVertex {}; struct ModelsBuffer {};
struct SceneSettings {}; struct InteractionMode {}; struct Submit {}; struct Rotation {};
struct ViewportTheme {}; struct Materials {}; struct PbrSpecialization {};
struct SceneView {}; struct CameraLens {}; struct TransformPending {};
struct TransformEnd {}; struct WorldTransform {};
} // namespace changes
// clang-format on

enum class On : uint8_t {
    Create = 1,
    Update = 2,
    Destroy = 4
};
constexpr On operator|(On a, On b) { return On(uint8_t(a) | uint8_t(b)); }

struct ReactiveTracker {
    entt::storage_for_t<entt::reactive> &s;
    template<typename T> ReactiveTracker &on(On events) {
        if (uint8_t(events) & uint8_t(On::Create)) s.on_construct<T>();
        if (uint8_t(events) & uint8_t(On::Update)) s.on_update<T>();
        if (uint8_t(events) & uint8_t(On::Destroy)) s.on_destroy<T>();
        return *this;
    }
};
template<typename Change>
ReactiveTracker track(entt::registry &r) { return {r.storage<entt::reactive>(entt::type_hash<Change>::value())}; }
template<typename Change>
auto &reactive(entt::registry &r) { return r.storage<entt::reactive>(entt::type_hash<Change>::value()); }

struct DeformSlots {
    uint32_t BoneDeformOffset{InvalidOffset}, ArmatureDeformOffset{InvalidOffset}, MorphDeformOffset{InvalidOffset};
    uint32_t MorphTargetCount{0};
    // Per-instance morph weights: buffer_index -> offset (weights are per-node in glTF)
    std::unordered_map<uint32_t, uint32_t> MorphWeightsByBufferIndex;
};

std::unordered_map<entt::entity, DeformSlots> BuildDeformSlots(const entt::registry &r, const MeshStore &meshes) {
    std::unordered_map<entt::entity, DeformSlots> result;
    for (const auto [_, instance, modifier] : r.view<const Instance, const ArmatureModifier>().each()) {
        if (result.contains(instance.Entity)) continue;
        const auto &mesh = r.get<const Mesh>(instance.Entity);
        const auto bone_deform = meshes.GetBoneDeformRange(mesh.GetStoreId());
        if (bone_deform.Count == 0) continue;
        if (const auto *pose_state = r.try_get<const ArmaturePoseState>(modifier.ArmatureEntity)) {
            result[instance.Entity] = {
                .BoneDeformOffset = bone_deform.Offset,
                .ArmatureDeformOffset = pose_state->GpuDeformRange.Offset,
                .MorphDeformOffset = InvalidOffset,
                .MorphTargetCount = 0,
                .MorphWeightsByBufferIndex = {},
            };
        }
    }
    // Add morph target slots for mesh instances with morph data (per-instance weights)
    for (const auto [instance_entity, instance, morph_state, ri] : r.view<const Instance, const MorphWeightState, const RenderInstance>().each()) {
        const auto mesh_entity = instance.Entity;
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
    return any_of(r.view<const Instance, const Selected>().each(), [mesh_entity](const auto &t) { return std::get<1>(t).Entity == mesh_entity; });
}

ModelsBuffer CreateModelsBuffer(mvk::BufferContext &ctx, uint32_t instance_count = 1) {
    return {
        mvk::Buffer{ctx, instance_count * sizeof(WorldTransform), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ModelBuffer},
        mvk::Buffer{ctx, instance_count * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ObjectIdBuffer},
        mvk::Buffer{ctx, instance_count * sizeof(uint8_t), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::InstanceStateBuffer},
    };
}

void ReserveModelsBufferInstance(ModelsBuffer &mb) {
    mb.Buffer.Reserve(mb.Buffer.UsedSize + sizeof(WorldTransform));
    mb.ObjectIds.Reserve(mb.ObjectIds.UsedSize + sizeof(uint32_t));
    mb.InstanceStates.Reserve(mb.InstanceStates.UsedSize + sizeof(uint8_t));
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
const PBRMaterial &GetMaterial(const SceneBuffers &buffers, uint32_t index) { return reinterpret_cast<const PBRMaterial *>(buffers.MaterialBuffer.GetData().data())[index]; }
PBRMaterial &GetMaterial(SceneBuffers &buffers, uint32_t index) { return reinterpret_cast<PBRMaterial *>(buffers.MaterialBuffer.GetMappedData().data())[index]; }
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
          ObjectPickSeenBits(slots.Allocate(SlotType::Buffer)),
          SelectionBitset(slots.Allocate(SlotType::Buffer)),
          ObjectIdSampler(slots.Allocate(SlotType::Sampler)),
          DepthSampler(slots.Allocate(SlotType::Sampler)),
          SilhouetteSampler(slots.Allocate(SlotType::Sampler)),
          ColorSampler(slots.Allocate(SlotType::Sampler)),
          LineDataSampler(slots.Allocate(SlotType::Sampler)) {}

    ~SelectionSlotHandles() {
        Slots.Release({SlotType::Image, HeadImage});
        Slots.Release({SlotType::Buffer, SelectionCounter});
        Slots.Release({SlotType::Buffer, ObjectPickKey});
        Slots.Release({SlotType::Buffer, ElementPickCandidates});
        Slots.Release({SlotType::Buffer, ObjectPickSeenBits});
        Slots.Release({SlotType::Buffer, SelectionBitset});
        Slots.Release({SlotType::Sampler, ObjectIdSampler});
        Slots.Release({SlotType::Sampler, DepthSampler});
        Slots.Release({SlotType::Sampler, SilhouetteSampler});
        Slots.Release({SlotType::Sampler, ColorSampler});
        Slots.Release({SlotType::Sampler, LineDataSampler});
    }

    DescriptorSlots &Slots;
    uint32_t HeadImage, SelectionCounter, ObjectPickKey, ElementPickCandidates, ObjectPickSeenBits, SelectionBitset, ObjectIdSampler, DepthSampler, SilhouetteSampler, ColorSampler, LineDataSampler;
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
    track<changes::Selected>(R).on<Selected>(On::Create | On::Destroy);
    track<changes::ActiveInstance>(R).on<Active>(On::Create | On::Destroy);
    track<changes::BoneSelection>(R).on<BoneSelection>(On::Create | On::Update | On::Destroy).on<BoneActive>(On::Create | On::Destroy);
    track<changes::Rerecord>(R)
        .on<RenderInstance>(On::Create | On::Destroy)
        .on<Active>(On::Create | On::Destroy)
        .on<StartTransform>(On::Create | On::Destroy)
        .on<SceneEditMode>(On::Create | On::Update);
    track<changes::MeshActiveElement>(R).on<MeshActiveElement>(On::Create | On::Update);
    track<changes::MeshGeometry>(R).on<MeshGeometryDirty>(On::Create);
    track<changes::MeshMaterial>(R).on<MeshMaterialAssignment>(On::Create | On::Update);
    track<changes::Excitable>(R).on<Excitable>(On::Create | On::Destroy);
    track<changes::ExcitedVertex>(R).on<ExcitedVertex>(On::Create | On::Destroy);
    track<changes::ModelsBuffer>(R).on<ModelsBuffer>(On::Update);
    track<changes::SceneSettings>(R).on<SceneSettings>(On::Create | On::Update);
    track<changes::InteractionMode>(R).on<SceneInteraction>(On::Create | On::Update);
    track<changes::Submit>(R).on<SubmitDirty>(On::Create);
    track<changes::ViewportTheme>(R).on<ViewportTheme>(On::Create | On::Update);
    track<changes::Materials>(R).on<MaterialDirty>(On::Create | On::Update);
    track<changes::PbrSpecialization>(R)
        .on<PbrMeshFeatures>(On::Create | On::Update | On::Destroy)
        .on<MaterialPreviewLighting>(On::Create | On::Update)
        .on<RenderedLighting>(On::Create | On::Update);
    track<changes::SceneView>(R)
        .on<ViewCamera>(On::Create | On::Update)
        .on<MaterialPreviewLighting>(On::Create | On::Update)
        .on<RenderedLighting>(On::Create | On::Update)
        .on<LightIndex>(On::Create | On::Destroy)
        .on<ViewportExtent>(On::Create | On::Update)
        .on<SceneEditMode>(On::Create | On::Update);
    track<changes::CameraLens>(R).on<Camera>(On::Create | On::Update);
    track<changes::Rotation>(R).on<::Rotation>(On::Create | On::Update);
    track<changes::WorldTransform>(R).on<WorldTransform>(On::Create | On::Update);
    track<changes::TransformPending>(R).on<PendingTransform>(On::Create | On::Update);
    track<changes::TransformEnd>(R).on<StartTransform>(On::Destroy);

    DestroyTracker->Bind(R);

    SceneEntity = R.create();
    R.emplace<SceneSettings>(SceneEntity);
    R.emplace<SceneInteraction>(SceneEntity);
    R.emplace<SceneEditMode>(SceneEntity);
    R.emplace<ViewportTheme>(SceneEntity, SceneDefaults::ViewportTheme);
    R.emplace<colors::AxesArray>(SceneEntity, colors::MakeAxes(SceneDefaults::ViewportTheme.AxisColors));
    R.emplace<ViewCamera>(SceneEntity, SceneDefaults::ViewCamera);
    R.emplace<MaterialPreviewLighting>(SceneEntity, false, false, 1.f, 0.f);
    R.emplace<RenderedLighting>(SceneEntity, true, true, 1.f, 0.f);
    R.emplace<ViewportExtent>(SceneEntity);
    R.emplace<AnimationTimeline>(SceneEntity);
    R.emplace<MaterialStore>(SceneEntity);
    Buffers->WorkspaceLightsUBO.Update(as_bytes(SceneDefaults::WorkspaceLights));

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
            environments.Hdris.emplace_back(HdriEntry{.Name = stem, .Path = entry.path(), .Prefiltered = {}});
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
    if (const auto *instance = R.try_get<Instance>(e); instance && R.all_of<Mesh>(instance->Entity)) return instance->Entity;
    return entt::null;
}
entt::entity Scene::GetActiveMeshEntity() const {
    if (const auto active = FindActiveEntity(R); active != entt::null) return GetMeshEntity(active);
    return entt::null;
}

void Scene::Select(entt::entity e) {
    R.clear<Selected>();
    if (e != entt::null) {
        R.clear<Active>();
        R.emplace<Active>(e);
        R.emplace<Selected>(e);
    }
}
void Scene::ToggleSelected(entt::entity e) {
    if (e == entt::null) return;

    if (R.all_of<Selected>(e)) R.remove<Selected>(e);
    else R.emplace_or_replace<Selected>(e);
}

void Scene::SelectBone(entt::entity e) {
    R.clear<BoneSelection>();
    if (e != entt::null) {
        R.clear<BoneActive>();
        R.emplace<BoneActive>(e);
        R.emplace<BoneSelection>(e);
    }
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

    CreateSvgResource(ShadingIcons.Wireframe, svg_path / "shading_wire.svg");
    CreateSvgResource(ShadingIcons.Solid, svg_path / "shading_solid.svg");
    CreateSvgResource(ShadingIcons.MaterialPreview, svg_path / "shading_texture.svg");
    CreateSvgResource(ShadingIcons.Rendered, svg_path / "shading_rendered.svg");

    CreateSvgResource(OverlayIcon, svg_path / "overlay.svg");

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
    const bool profile = std::exchange(ProfileNextProcessComponentEvents, false);
    std::optional<Timer> timer;
    if (profile) timer.emplace("ProcessComponentEvents (post-glTF)");

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
        } else if (!reactive<changes::Excitable>(R).empty()) {
            InteractionModes.insert(InteractionMode::Excite);
            if (interaction_mode == InteractionMode::Excite) request(RenderRequest::ReRecord);
            else SetInteractionMode(InteractionMode::Excite); // Switch to excite mode
        }
    }
    std::unordered_set<entt::entity> dirty_overlay_meshes, dirty_element_state_meshes;
    std::unordered_set<entt::entity> dirty_bone_state_armatures; // Armature obj entities needing bone GPU instance state sync.
    // Helper: write instance_entity's own Selected/Active state to its GPU slot.
    const auto update_instance_state = [&](entt::entity instance_entity) {
        if (const auto *ri = R.try_get<RenderInstance>(instance_entity)) {
            uint8_t state = 0;
            if (R.all_of<Selected>(instance_entity)) state |= ElementStateSelected;
            if (R.all_of<Active>(instance_entity)) state |= ElementStateActive;
            R.patch<ModelsBuffer>(ri->Entity, [&](auto &mb) {
                mb.InstanceStates.Update(as_bytes(state), ri->BufferIndex * sizeof(uint8_t));
            });
        }
    };

    { // Selected/Active instance changes - update instance state buffer
        auto &selected_tracker = reactive<changes::Selected>(R);
        auto &active_tracker = reactive<changes::ActiveInstance>(R);
        if (!selected_tracker.empty()) request(RenderRequest::ReRecord);

        for (auto instance_entity : selected_tracker) {
            update_instance_state(instance_entity);
            if (const auto arm = FindArmatureObject(R, instance_entity); arm != entt::null) dirty_bone_state_armatures.insert(arm);
            if (const auto *instance = R.try_get<Instance>(instance_entity); instance && R.all_of<Mesh>(instance->Entity)) {
                const auto mesh_entity = instance->Entity;
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
            if (const auto arm = FindArmatureObject(R, instance_entity); arm != entt::null) dirty_bone_state_armatures.insert(arm);
            // If looking through a camera and a different camera becomes active, snap to it.
            if (SavedViewCamera && R.all_of<Camera>(instance_entity) && R.all_of<Active>(instance_entity)) {
                SnapToCamera(instance_entity);
            }
        }
    }
    { // Bone selection changes — route to dirty_bone_state_armatures for GPU state sync.
        auto &bone_sel_tracker = reactive<changes::BoneSelection>(R);
        if (!bone_sel_tracker.empty()) {
            request(RenderRequest::ReRecord);
            for (auto bone_entity : bone_sel_tracker) {
                if (const auto arm = FindArmatureObject(R, bone_entity); arm != entt::null) dirty_bone_state_armatures.insert(arm);
            }
        }
    }
    if (!reactive<changes::Rerecord>(R).empty() || !DestroyTracker->Storage.empty()) {
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

    if (const auto &tracker = reactive<changes::MeshActiveElement>(R); !tracker.empty()) {
        const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
        const auto active_entity = FindActiveEntity(R);
        const auto *active_instance = R.try_get<Instance>(active_entity);
        for (auto mesh_entity : tracker) {
            if (const auto *active_element = R.try_get<MeshActiveElement>(mesh_entity);
                active_element && edit_mode != Element::None && active_instance && active_instance->Entity == mesh_entity) {
                orbit_to_active(active_entity, edit_mode, active_element->Handle);
            }
            dirty_element_state_meshes.insert(mesh_entity); // for Excite mode
        }
    }
    if (SelectionBitsDirty) {
        SelectionBitsDirty = false;
        if (is_edit_mode) ApplySelectionStateUpdate(GetBitsetRangesForSelected(), R.get<const SceneEditMode>(SceneEntity).Value);
    }
    for (auto instance_entity : reactive<changes::ExcitedVertex>(R)) {
        if (const auto *inst = R.try_get<Instance>(instance_entity)) dirty_element_state_meshes.insert(inst->Entity);
        if (const auto *ev = R.try_get<ExcitedVertex>(instance_entity)) orbit_to_active(instance_entity, Element::Vertex, ev->Vertex);
    }
    for (auto camera_entity : reactive<changes::CameraLens>(R)) {
        if (const auto *cd = R.try_get<Camera>(camera_entity)) {
            const auto buffer_entity = R.get<Instance>(camera_entity).Entity;
            Meshes->SetPositions(R.get<const VertexStoreId>(buffer_entity).StoreId, BuildCameraFrustumMesh(*cd).Positions);
            R.emplace_or_replace<SubmitDirty>(buffer_entity);
            // If looking through this camera, trigger a ViewCamera update so the SceneView
            // handler re-derives the widened FOV from the updated camera.
            if (SavedViewCamera && R.all_of<Active>(camera_entity)) {
                R.patch<ViewCamera>(SceneEntity, [](auto &) {});
            }
        }
    }
    { // Sync RotationUiVariant from Rotation, but skip entities where the UI is driving the change.
        for (auto e : reactive<changes::Rotation>(R)) {
            if (!R.all_of<::Rotation>(e)) continue;
            if (R.all_of<RotationUiDriving>(e)) {
                R.remove<RotationUiDriving>(e);
                continue;
            }
            const auto v = R.get<const ::Rotation>(e).Value;
            if (auto *ui = R.try_get<RotationUiVariant>(e)) {
                std::visit(
                    overloaded{
                        [&](RotationQuat &u) { u.Value = v; },
                        [&](RotationEuler &u) {
                            float x, y, z;
                            glm::extractEulerAngleXYZ(glm::mat4_cast(v), x, y, z);
                            u.Value = glm::degrees(vec3{x, y, z});
                        },
                        [&](RotationAxisAngle &u) {
                            const auto q = glm::normalize(v);
                            u.Value = {glm::axis(q), glm::degrees(glm::angle(q))};
                        },
                    },
                    *ui
                );
            } else {
                R.emplace<RotationUiVariant>(e, RotationQuat{v});
            }
        }
    }

    bool light_count_changed = false;
    if (const auto required_size = vk::DeviceSize(R.storage<LightIndex>().size()) * sizeof(PunctualLight);
        Buffers->LightBuffer.UsedSize != required_size) {
        Buffers->LightBuffer.Reserve(required_size);
        Buffers->LightBuffer.UsedSize = required_size;
        light_count_changed = true;
    }
    if (!reactive<changes::Submit>(R).empty() || light_count_changed) request(RenderRequest::Submit);
    for (auto light_entity : R.view<LightWireframeDirty, LightIndex, Instance>()) {
        const auto light = Buffers->GetLight(R.get<const LightIndex>(light_entity).Value);
        auto wireframe = BuildLightMesh(light);
        const auto buffer_entity = R.get<Instance>(light_entity).Entity;

        // Release old vertex classes (need old vertex count from MeshBuffers before releasing)
        if (const auto *old_vcr = R.try_get<VertexClass>(buffer_entity)) {
            Buffers->VertexClassBuffer.Release({old_vcr->Offset, R.get<const MeshBuffers>(buffer_entity).Vertices.Count});
            R.remove<VertexClass>(buffer_entity);
        }

        // Release old store entry and mesh buffers
        auto &vs = R.get<VertexStoreId>(buffer_entity);
        Meshes->Release(vs.StoreId);
        if (auto *mb = R.try_get<MeshBuffers>(buffer_entity)) Buffers->Release(*mb);
        R.erase<MeshBuffers>(buffer_entity);

        // Allocate new
        const auto [store_id, vertices] = Meshes->AllocateVertexBuffer(wireframe.Data.Positions, {});
        vs.StoreId = store_id;

        R.emplace<MeshBuffers>(
            buffer_entity, Meshes->GetVerticesRange(store_id),
            SlottedRange{},
            Buffers->CreateIndices(wireframe.Data.CreateEdgeIndices(), IndexKind::Edge),
            SlottedRange{}
        );
        if (!wireframe.VertexClasses.empty()) {
            const auto range = Buffers->VertexClassBuffer.Allocate(std::span<const uint8_t>(wireframe.VertexClasses));
            R.emplace<VertexClass>(buffer_entity, range.Offset);
        }
        request(RenderRequest::ReRecord);
    }
    if (auto &tracker = reactive<changes::MeshGeometry>(R); !tracker.empty()) {
        const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
        std::vector<ElementRange> geometry_ranges;
        for (auto mesh_entity : tracker) {
            if (HasSelectedInstance(R, mesh_entity)) dirty_overlay_meshes.insert(mesh_entity);
            if (auto *br = R.try_get<MeshSelectionBitsetRange>(mesh_entity); br && edit_mode != Element::None) {
                // Topology changed: zero stale selection bits and update count.
                const auto &mesh = R.get<const Mesh>(mesh_entity);
                const uint32_t new_count = scene_selection::GetElementCount(mesh, edit_mode);
                const uint32_t max_words = (std::max(br->Count, new_count) + 31) / 32;
                auto *mapped = Buffers->SelectionBitsetBuffer.GetMappedData().data();
                memset(mapped + (br->Offset / 32) * sizeof(uint32_t), 0, max_words * sizeof(uint32_t));
                br->Count = new_count;
                if (new_count > 0) geometry_ranges.emplace_back(mesh_entity, br->Offset, br->Count);
            }
        }
        if (!geometry_ranges.empty()) ApplySelectionStateUpdate(geometry_ranges, edit_mode);
        request(RenderRequest::Submit);
    }
    if (auto &tracker = reactive<changes::MeshMaterial>(R); !tracker.empty()) {
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
    if (!reactive<changes::ModelsBuffer>(R).empty()) request(RenderRequest::Submit);
    if (!reactive<changes::ViewportTheme>(R).empty()) {
        UpdateDerivedColors(R.get<ViewportTheme>(SceneEntity));
        R.get<colors::AxesArray>(SceneEntity) = colors::MakeAxes(R.get<const ViewportTheme>(SceneEntity).AxisColors);
        auto theme = R.get<const ViewportTheme>(SceneEntity);
        theme.EdgeWidth *= ImGui::GetIO().DisplayFramebufferScale.x;
        Buffers->ViewportThemeUBO.Update(as_bytes(theme));
        request(RenderRequest::Submit);
    }
    if (!reactive<changes::Materials>(R).empty()) {
        if (const auto *dirty = R.try_get<const MaterialDirty>(SceneEntity);
            dirty && dirty->Index < GetMaterialCount(*Buffers)) {
            SetMaterial(*Buffers, dirty->Index, GetMaterial(*Buffers, dirty->Index));
        }
        request(RenderRequest::Submit);
    }
    if (!reactive<changes::SceneSettings>(R).empty()) {
        request(RenderRequest::ReRecord);
        dirty_overlay_meshes.merge(scene_selection::GetSelectedMeshEntities(R));
    }
    if (!reactive<changes::InteractionMode>(R).empty()) {
        request(RenderRequest::ReRecord);
        // Dispatch UpdateSelectionState for all meshes entering Edit mode (MeshSelectionBitsetRange assigned in SetInteractionMode).
        const auto new_interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
        if (new_interaction_mode == InteractionMode::Edit) {
            const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
            if (edit_mode != Element::None) ApplySelectionStateUpdate(GetBitsetRangesForSelected(), edit_mode);
        }
        for (const auto [_, instance, __] : R.view<const Instance, const Excitable>().each()) {
            dirty_element_state_meshes.insert(instance.Entity);
        }
        // Mark all armatures dirty for bone state + pose sync on mode change.
        for (const auto arm : R.view<ArmatureObject>()) dirty_bone_state_armatures.insert(arm);
    }
    // Handle mesh Edit mode transform commit when StartTransform is cleared.
    // Bone Edit mode commits are handled in the bone pose transform section below.
    if (!reactive<changes::TransformEnd>(R).empty()) {
        if (is_edit_mode && FindArmatureObject(R, FindActiveEntity(R)) == entt::null) {
            const auto &pending = R.get<const PendingTransform>(SceneEntity);
            // Apply edit transform once per selected mesh via a representative selected instance.
            // This keeps linked instances from receiving duplicate per-instance edits.
            for (const auto &[mesh_entity, instance_entity] : edit_transform_context.TransformInstances) {
                if (scene_selection::HasFrozenInstance(R, mesh_entity)) continue;
                const auto &mesh = R.get<const Mesh>(mesh_entity);
                const auto vertex_states = Meshes->GetVertexStates(mesh.GetStoreId());
                const bool any_selected = std::ranges::any_of(vertex_states, [](const auto s) { return (s & ElementStateSelected) != 0u; });
                if (!any_selected) continue;
                const auto vertices = mesh.GetVerticesSpan();
                const auto &wt = R.get<const WorldTransform>(instance_entity);
                const auto wt_rot = Vec4ToQuat(wt.Rotation);
                const auto inv_rot = glm::conjugate(wt_rot);
                const auto inv_scale = 1.f / wt.Scale;
                for (uint32_t vi = 0; vi < vertex_states.size(); ++vi) {
                    if ((vertex_states[vi] & ElementStateSelected) == 0u) continue;
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
            R.remove<PendingTransform>(SceneEntity);
        }
    }
    const bool mode_changed = !reactive<changes::InteractionMode>(R).empty();
    bool anim_advanced;
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
        anim_advanced = tl.CurrentFrame != LastEvaluatedFrame;
        if (anim_advanced) LastEvaluatedFrame = tl.CurrentFrame;
        // Timeline frames are displayed 1-based, but animation time starts at t=0 on frame 1.
        const float eval_seconds = float(std::max(0, tl.CurrentFrame - 1)) / tl.Fps;

        // Evaluate animation deltas (data only — no bone entity iteration).
        bool request_rerecord = false;
        if (anim_advanced) {
            for (const auto [arm_obj_entity, arm_obj_comp] : R.view<const ArmatureObject>().each()) {
                auto *pose_state = R.try_get<ArmaturePoseState>(arm_obj_comp.Entity);
                if (!pose_state) continue;
                const auto &armature = R.get<const Armature>(arm_obj_comp.Entity);
                if (!armature.ImportedSkin) continue;
                if (const auto *anim = R.try_get<const ArmatureAnimation>(arm_obj_comp.Entity);
                    anim && !anim->Clips.empty() && anim->ActiveClipIndex < anim->Clips.size()) {
                    const auto &clip = anim->Clips[anim->ActiveClipIndex];
                    const float clip_time = clip.DurationSeconds > 0 ? std::fmod(eval_seconds, clip.DurationSeconds) : 0.0f;
                    EvaluateAnimationDeltas(clip, clip_time, armature.Bones, pose_state->BonePoseDelta);
                }
            }
        }

        if (anim_advanced) {
            // Evaluate morph weight animations
            for (auto [entity, morph_anim, morph_state, instance] :
                 R.view<const MorphWeightAnimation, MorphWeightState, const Instance>().each()) {
                if (morph_anim.Clips.empty() || morph_anim.ActiveClipIndex >= morph_anim.Clips.size()) continue;
                const auto &clip = morph_anim.Clips[morph_anim.ActiveClipIndex];
                const float clip_time = clip.DurationSeconds > 0 ? std::fmod(eval_seconds, clip.DurationSeconds) : 0.0f;
                // Reset to default weights from mesh-level data
                const auto &mesh = R.get<const Mesh>(instance.Entity);
                const auto default_weights = Meshes->GetDefaultMorphWeights(mesh.GetStoreId());
                std::copy(default_weights.begin(), default_weights.end(), morph_state.Weights.begin());
                EvaluateMorphWeights(clip, clip_time, morph_state.Weights);
                // Write to GPU
                auto gpu_weights = Buffers->MorphWeightBuffer.GetMutable(morph_state.GpuWeightRange);
                std::copy(morph_state.Weights.begin(), morph_state.Weights.end(), gpu_weights.begin());
                request_rerecord = true;
            }
            // Evaluate glTF node transform animations (empties/meshes/cameras/lights).
            for (auto [entity, node_anim] : R.view<const NodeTransformAnimation>().each()) {
                if (node_anim.Clips.empty() || node_anim.ActiveClipIndex >= node_anim.Clips.size()) continue;
                const auto &clip = node_anim.Clips[node_anim.ActiveClipIndex];
                const float clip_time = clip.DurationSeconds > 0 ? std::fmod(eval_seconds, clip.DurationSeconds) : 0.0f;
                std::array<Transform, 1> local_pose{node_anim.RestLocal};
                EvaluateAnimation(clip, clip_time, local_pose);
                SetTransform(R, entity, ComposeWorldTransform(node_anim.ParentBindWorld, local_pose.front()));
                request_rerecord = true;
            }
        }
        if (request_rerecord) request(RenderRequest::ReRecord);
    }
    { // Bones
        // GPU instance state
        const bool is_object_mode = interaction_mode == InteractionMode::Object;
        for (const auto arm_obj_entity : dirty_bone_state_armatures) {
            if (!R.all_of<MeshBuffers>(arm_obj_entity)) continue;
            const auto &arm_obj = R.get<const ArmatureObject>(arm_obj_entity);
            const auto &bone_entities = arm_obj.BoneEntities;
            // Object mode: all bone/joint slots get the armature's object-level state (no per-bone color).
            // Edit/Pose: each bone/joint gets its own state based on BoneActive/BoneSelection.
            uint8_t max_state = 0;
            if (is_object_mode) {
                // Only show active/selected bone colors when the armature is selected.
                // In solid mode, unselected armatures don't draw wires at all.
                // In wireframe mode, they draw with neutral Wire color.
                if (R.all_of<Selected>(arm_obj_entity)) {
                    max_state |= ElementStateSelected;
                    if (R.all_of<Active>(arm_obj_entity)) max_state |= ElementStateActive;
                }
            }
            const bool is_edit = interaction_mode == InteractionMode::Edit;
            auto compute_state = [&](entt::entity b, BoneSel part) -> uint8_t {
                if (is_object_mode) return max_state;
                uint8_t s = 0;
                if (R.all_of<BoneActive>(b)) s |= ElementStateActive;
                const auto *parts = R.try_get<const BoneSelection>(b);
                bool selected;
                if (is_edit) {
                    selected = parts && (part == BoneSel::Body ? parts->Body : part == BoneSel::Root ? parts->Root :
                                                                                                       parts->Tip);
                } else {
                    selected = R.all_of<BoneSelection>(b);
                }
                if (selected) s |= ElementStateSelected;
                return s;
            };

            R.patch<ModelsBuffer>(arm_obj_entity, [&](auto &mb) {
                for (const auto b : bone_entities) {
                    if (const auto bi = GetModelBufferIndex(R, b)) {
                        const auto state = compute_state(b, BoneSel::Body);
                        mb.InstanceStates.Update(as_bytes(state), *bi * sizeof(uint8_t));
                    }
                }
            });
            if (arm_obj.JointEntity != entt::null && R.valid(arm_obj.JointEntity)) {
                R.patch<ModelsBuffer>(arm_obj.JointEntity, [&](auto &mb) {
                    for (const auto b : bone_entities) {
                        const auto *joints = R.try_get<const BoneJointEntities>(b);
                        if (!joints) continue;
                        for (const auto &[je, part] : {std::pair{joints->Head, BoneSel::Root}, {joints->Tail, BoneSel::Tip}}) {
                            if (je != entt::null) {
                                if (const auto *ri = R.try_get<const RenderInstance>(je)) {
                                    const auto state = compute_state(b, part);
                                    mb.InstanceStates.Update(as_bytes(state), ri->BufferIndex * sizeof(uint8_t));
                                }
                            }
                        }
                    }
                });
            }
        }
        // Bone pose transforms + deform matrices
        const bool bones_need_refresh = anim_advanced || mode_changed;
        if (bones_need_refresh || !reactive<changes::WorldTransform>(R).empty() || !reactive<changes::TransformEnd>(R).empty()) {
            const auto &wt_changes = reactive<changes::WorldTransform>(R);
            const auto &transform_end = reactive<changes::TransformEnd>(R);
            for (const auto [arm_obj_entity, arm_obj_comp] : R.view<const ArmatureObject>().each()) {
                auto *pose_state = R.try_get<ArmaturePoseState>(arm_obj_comp.Entity);
                if (!pose_state) continue;
                auto &armature = R.get<Armature>(arm_obj_comp.Entity);
                if (!armature.ImportedSkin) continue;

                bool need_sync = false;
                bool rest_pose_edited = false;
                // Track which bones were explicitly transformed and save old rest world for cascade fix.
                std::vector<bool> was_transformed(armature.Bones.size(), false);
                std::vector<mat4> old_rest_world(armature.Bones.size());
                for (uint32_t i = 0; i < armature.Bones.size(); ++i) old_rest_world[i] = armature.Bones[i].RestWorld;
                for (uint32_t i = 0; i < arm_obj_comp.BoneEntities.size(); ++i) {
                    const auto b = arm_obj_comp.BoneEntities[i];
                    if (i >= pose_state->BonePoseDelta.size()) continue;
                    const auto &rest = armature.Bones[i].RestLocal;
                    if (is_edit_mode) {
                        if (mode_changed) {
                            // Entering Edit mode: snap to rest pose.
                            R.replace<Position>(b, rest.P);
                            R.replace<Rotation>(b, rest.R);
                            need_sync = true;
                        } else if (transform_end.contains(b) || (wt_changes.contains(b) && !R.all_of<StartTransform>(b))) {
                            // Commit Edit mode transform (gizmo drag end or UI slider edit).
                            armature.Bones[i].RestLocal.P = R.get<Position>(b).Value;
                            armature.Bones[i].RestLocal.R = R.get<Rotation>(b).Value;
                            was_transformed[i] = true;
                            rest_pose_edited = true;
                            need_sync = true;
                        }
                    } else if (const auto *st = R.try_get<const StartTransform>(b)) {
                        // Active drag: compute user offset into BoneUserOffset (additive on top of animation).
                        const auto &pd = st->ParentDelta;
                        const auto grab_delta = AbsoluteToDelta(
                            rest,
                            {
                                .P = glm::conjugate(pd.R) * ((st->T.P - pd.P) / pd.S),
                                .R = glm::conjugate(pd.R) * st->T.R,
                                .S = st->T.S / pd.S,
                            }
                        );
                        const Transform gizmo_local{R.get<Position>(b).Value, R.get<Rotation>(b).Value, rest.S};
                        pose_state->BoneUserOffset[i] = AbsoluteToDelta(grab_delta, AbsoluteToDelta(rest, gizmo_local));
                        const auto local = ComposeWithDelta(rest, ComposeWithDelta(pose_state->BonePoseDelta[i], pose_state->BoneUserOffset[i]));
                        R.replace<Position>(b, local.P);
                        R.replace<Rotation>(b, local.R);
                        need_sync = true;
                    } else if (transform_end.contains(b)) {
                        // Commit drag: bake current P/R into delta, clear offset.
                        pose_state->BonePoseDelta[i] = AbsoluteToDelta(rest, {R.get<Position>(b).Value, R.get<Rotation>(b).Value, rest.S});
                        pose_state->BoneUserOffset[i] = {};
                        need_sync = true;
                    } else if (bones_need_refresh) {
                        // Animation advanced or leaving Edit mode: recompute entity P/R from deltas.
                        const auto local = ComposeWithDelta(rest, ComposeWithDelta(pose_state->BonePoseDelta[i], pose_state->BoneUserOffset[i]));
                        R.replace<Position>(b, local.P);
                        R.replace<Rotation>(b, local.R);
                        need_sync = true;
                    } else if (wt_changes.contains(b)) {
                        // Manual transform: bake if position actually changed.
                        const auto expected = ComposeWithDelta(rest, ComposeWithDelta(pose_state->BonePoseDelta[i], pose_state->BoneUserOffset[i]));
                        const auto pos = R.get<Position>(b).Value;
                        const auto rot = R.get<Rotation>(b).Value;
                        if (pos == expected.P && rot == expected.R) continue;
                        pose_state->BonePoseDelta[i] = AbsoluteToDelta(rest, {pos, rot, rest.S});
                        pose_state->BoneUserOffset[i] = {};
                        need_sync = true;
                    }
                }
                if (rest_pose_edited) {
                    // Recompute RestWorld in topological order, preserving world positions of untransformed bones.
                    for (uint32_t i = 0; i < armature.Bones.size(); ++i) {
                        const auto parent = armature.Bones[i].ParentIndex;
                        const mat4 parent_world = (parent == InvalidBoneIndex) ? I4 : armature.Bones[parent].RestWorld;
                        if (was_transformed[i]) {
                            armature.Bones[i].RestWorld = parent_world * RestLocalToMatrix(armature.Bones[i].RestLocal);
                        } else {
                            // Preserve old world position, adjust RestLocal to compensate for parent change.
                            const mat4 new_local_mat = glm::inverse(parent_world) * old_rest_world[i];
                            armature.Bones[i].RestLocal.P = vec3(new_local_mat[3]);
                            armature.Bones[i].RestLocal.R = glm::normalize(glm::quat_cast(mat3(new_local_mat)));
                            armature.Bones[i].RestWorld = old_rest_world[i];
                            // Update ECS to match adjusted RestLocal.
                            const auto b = arm_obj_comp.BoneEntities[i];
                            R.replace<Position>(b, armature.Bones[i].RestLocal.P);
                            R.replace<Rotation>(b, armature.Bones[i].RestLocal.R);
                        }
                        armature.Bones[i].InvRestWorld = glm::inverse(armature.Bones[i].RestWorld);
                    }
                    armature.RecomputeInverseBindMatrices();
                }
                if (need_sync) {
                    UpdateWorldTransform(R, arm_obj_entity);
                    if (!is_edit_mode) {
                        ComputeDeformMatrices(
                            armature, pose_state->BonePoseDelta, pose_state->BoneUserOffset,
                            armature.ImportedSkin->InverseBindMatrices,
                            Buffers->ArmatureDeformBuffer.GetMutable(pose_state->GpuDeformRange)
                        );
                    }
                    request(RenderRequest::ReRecord);
                }
            }
        }
    }
    if (SavedViewCamera) {
        // If looking through a camera and it moved (animation or manual edit), snap the ViewCamera.
        // This must run before the SceneView handler so SnapToCamera's ViewCamera replacement is picked up.
        if (const auto active_entity = FindActiveEntity(R);
            active_entity != entt::null && R.all_of<Camera>(active_entity) &&
            reactive<changes::WorldTransform>(R).contains(active_entity)) {
            SnapToCamera(active_entity);
        }
    }
    if (!reactive<changes::SceneView>(R).empty() ||
        !reactive<changes::TransformPending>(R).empty() ||
        !reactive<changes::SceneSettings>(R).empty() ||
        !reactive<changes::InteractionMode>(R).empty() ||
        !reactive<changes::TransformEnd>(R).empty() ||
        light_count_changed) {
        // When looking through a scene camera, keep the ViewCamera's widened FOV in sync
        // with the current viewport aspect ratio (handles viewport resize).
        if (SavedViewCamera) {
            const auto active_entity = FindActiveEntity(R);
            if (const auto *cd = active_entity != entt::null ? R.try_get<Camera>(active_entity) : nullptr) {
                const auto logical_extent = R.get<const ViewportExtent>(SceneEntity).Value;
                const auto render_extent = ComputeRenderExtentPx(logical_extent);
                const float viewport_aspect = render_extent.width == 0 || render_extent.height == 0 ? 1.f : float(render_extent.width) / float(render_extent.height);
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
        const float background_blur = use_scene_world ? 0.f : active_lighting.BackgroundBlur;
        const float world_opacity = is_pbr_mode ? (use_scene_world ? 1.f : active_lighting.WorldOpacity) : 0.f;
        const auto *pending = R.try_get<const PendingTransform>(SceneEntity);
        const auto logical_extent = R.get<const ViewportExtent>(SceneEntity).Value;
        const auto render_extent = ComputeRenderExtentPx(logical_extent);
        const float viewport_height = render_extent.height > 0 ? float(render_extent.height) : 1.f;
        // ScreenPixelScale: world-space size per pixel at unit distance (perspective) or absolute (ortho).
        // Sign encodes camera type: positive = perspective (shader multiplies by distance), negative = orthographic.
        const float screen_pixel_scale = ScreenPixelScale(camera.Data, viewport_height);
        const float aspect = render_extent.width == 0 || render_extent.height == 0 ? 1.f : float(render_extent.width) / float(render_extent.height);
        const auto proj = camera.Projection(aspect);
        Buffers->SceneViewUBO.Update(as_bytes(SceneViewUBO{
            .ViewProj = proj * camera.View(),
            .ViewRotation = mat3(camera.View()),
            .CameraPosition = camera.Position(),
            .CameraNear = camera.NearClip(),
            .CameraFar = camera.FarClip(),
            .LightCount = uint32_t(Buffers->LightBuffer.UsedSize / sizeof(PunctualLight)),
            .LightSlot = Buffers->LightBuffer.Slot,
            .UseSceneLightsRender = use_scene_lights ? 1u : 0u,
            .EnvIntensity = env_intensity,
            .EnvRotationRadians = env_rotation_radians,
            .BackgroundBlur = background_blur,
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
            .ViewportSize = {float(render_extent.width), float(render_extent.height)},
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
            .BoneXRay = settings.ViewportShading == ViewportShadingMode::Wireframe ? 1u : 0u,
            // Polygon offset factor matching Blender's GPU_polygon_offset_calc (viewdist = max ortho extent)
            .NdcOffsetFactor = std::holds_alternative<Perspective>(camera.Data) ? proj[3][2] * -0.00125f : 0.000005f * std::max(std::abs(1.f / proj[0][0]), std::abs(1.f / proj[1][1])),
        }));
        request(RenderRequest::Submit);
    }

    { // Keep targeted PBR specialization mask in sync when one of its inputs changes.
        const auto shading = R.get<const SceneSettings>(SceneEntity).ViewportShading;
        if (!reactive<changes::SceneSettings>(R).empty() || !reactive<changes::PbrSpecialization>(R).empty()) {
            if (shading == ViewportShadingMode::MaterialPreview || shading == ViewportShadingMode::Rendered) {
                PbrFeatureMask pbr_mask{0};
                const bool use_scene_lights = shading == ViewportShadingMode::Rendered ?
                    R.get<const RenderedLighting>(SceneEntity).UseSceneLights :
                    R.get<const MaterialPreviewLighting>(SceneEntity).UseSceneLights;
                if (use_scene_lights) pbr_mask |= PbrFeature::Punctual;
                for (const auto [_, feat] : R.view<const PbrMeshFeatures>().each()) pbr_mask |= feat.Mask;
                if (Pipelines->Main.Compiler.CompilePipelines(pbr_mask)) request(RenderRequest::ReRecord);
            }
        }
    }

    const auto &settings = R.get<const SceneSettings>(SceneEntity);
    // Update selection overlays
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
    // Update mesh element state buffers (Excite mode only; Edit mode handled by GPU compute)
    for (const auto mesh_entity : dirty_element_state_meshes) {
        if (interaction_mode != InteractionMode::Excite) continue;
        const auto &mesh = R.get<const Mesh>(mesh_entity);
        std::unordered_set<VH> selected_vertices;
        std::unordered_set<EH> selected_edges, active_edges;
        std::unordered_set<FH> selected_faces;
        std::optional<uint32_t> active_handle;
        for (auto [entity, instance, excitable] : R.view<const Instance, const Excitable>().each()) {
            if (instance.Entity != mesh_entity) continue;
            selected_vertices.insert(excitable.ExcitableVertices.begin(), excitable.ExcitableVertices.end());
            if (const auto *excited_vertex = R.try_get<ExcitedVertex>(entity)) active_handle = excited_vertex->Vertex;
            break;
        }
        Meshes->UpdateElementStates(mesh, Element::Vertex, selected_vertices, selected_edges, active_edges, selected_faces, active_handle);
        SelectionStale = true;
    }
    if (!dirty_element_state_meshes.empty()) request(RenderRequest::Submit);
    if (ElementStatesDirty) {
        ElementStatesDirty = false;
        request(RenderRequest::Submit);
    }
    for (auto &&[id, storage] : R.storage()) {
        if (storage.info() == entt::type_id<entt::reactive>()) storage.clear();
    }
    DestroyTracker->Storage.clear();
    R.clear<MeshGeometryDirty, MeshMaterialAssignment, MaterialDirty, SubmitDirty, LightWireframeDirty>();

    return render_request;
}

void Scene::SetVisible(entt::entity entity, bool visible) {
    const bool already_visible = R.all_of<RenderInstance>(entity);
    if ((visible && already_visible) || (!visible && !already_visible)) return;

    if (visible) {
        const auto buffer_entity = R.get<Instance>(entity).Entity;

        const auto buffer_index = R.get<const ModelsBuffer>(buffer_entity).Buffer.UsedSize / sizeof(WorldTransform);
        const uint32_t object_id = NextObjectId++;
        R.emplace<RenderInstance>(entity, buffer_entity, buffer_index, object_id);
        const uint8_t initial_state = R.all_of<Selected>(entity) ? static_cast<uint8_t>(ElementStateSelected) : uint8_t{0};
        R.patch<ModelsBuffer>(buffer_entity, [&](auto &mb) {
            mb.Buffer.Insert(as_bytes(R.get<WorldTransform>(entity)), mb.Buffer.UsedSize);
            mb.ObjectIds.Insert(as_bytes(object_id), mb.ObjectIds.UsedSize);
            mb.InstanceStates.Insert(as_bytes(initial_state), mb.InstanceStates.UsedSize);
        });
    } else {
        const auto &ri = R.get<const RenderInstance>(entity);
        const auto buffer_entity = ri.Entity;
        const uint old_model_index = ri.BufferIndex;
        R.remove<RenderInstance>(entity);
        R.patch<ModelsBuffer>(buffer_entity, [old_model_index](auto &mb) {
            mb.Buffer.Erase(old_model_index * sizeof(WorldTransform), sizeof(WorldTransform));
            mb.ObjectIds.Erase(old_model_index * sizeof(uint32_t), sizeof(uint32_t));
            mb.InstanceStates.Erase(old_model_index * sizeof(uint8_t), sizeof(uint8_t));
        });
        // Update buffer indices for all instances of this buffer entity that have higher indices
        for (const auto [other_entity, other_ri] : R.view<const RenderInstance>().each()) {
            if (other_ri.Entity == buffer_entity && other_ri.BufferIndex > old_model_index) {
                R.patch<RenderInstance>(other_entity, [](auto &r) { --r.BufferIndex; });
            }
        }
    }
}

std::pair<entt::entity, entt::entity> Scene::AddMesh(Mesh &&mesh, std::optional<MeshInstanceCreateInfo> info) {
    const auto mesh_entity = R.create();
    R.emplace<ModelsBuffer>(mesh_entity, CreateModelsBuffer(Buffers->Ctx));
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
            }
            break;
    }
}

entt::entity Scene::AddMeshInstance(entt::entity mesh_entity, MeshInstanceCreateInfo info) {
    const auto instance_entity = R.create();
    R.emplace<Instance>(instance_entity, mesh_entity);
    R.emplace<ObjectKind>(instance_entity, ObjectType::Mesh);
    SetTransform(R, instance_entity, info.Transform);
    R.emplace<Name>(instance_entity, CreateName(R, info.Name));

    R.patch<ModelsBuffer>(mesh_entity, [](auto &mb) { ReserveModelsBufferInstance(mb); });
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

void Scene::CreateBoneInstances(entt::entity arm_obj_entity, entt::entity arm_data_entity) {
    const auto &armature = R.get<const Armature>(arm_data_entity);
    const uint32_t n = armature.Bones.size();
    if (n == 0) return;

    auto flatten_tri_indices = [](const std::vector<std::vector<uint32_t>> &faces) {
        std::vector<uint> indices;
        indices.reserve(faces.size() * 3);
        for (const auto &face : faces) {
            for (const auto idx : face) indices.emplace_back(idx);
        }
        return indices;
    };

    auto bone_data = primitive::BoneOctahedron(1.f);
    const auto bone_store_id = Meshes->AllocateVertexBuffer(bone_data.Mesh.Positions, bone_data.Attrs).first;
    const auto bone_tri_indices = flatten_tri_indices(bone_data.Mesh.Faces);
    const auto bone_vertex_count = static_cast<uint32_t>(bone_data.Mesh.Positions.size());
    R.emplace<ModelsBuffer>(arm_obj_entity, CreateModelsBuffer(Buffers->Ctx, n));
    R.emplace<MeshBuffers>(
        arm_obj_entity, Meshes->GetVerticesRange(bone_store_id),
        Buffers->CreateIndices(bone_tri_indices, IndexKind::Face),
        SlottedRange{}, // No edge indices (wire uses adjacency)
        Buffers->CreateIndices(iota(0u, bone_vertex_count) | to<std::vector>(), IndexKind::Vertex)
    );
    // Adjacency indices for silhouette edge detection (references the 6 unique positions at [0..5])
    R.emplace<BoneAdjacencyIndices>(arm_obj_entity, Buffers->CreateIndices(bone_data.AdjacencyIndices, IndexKind::Edge));
    R.emplace<VertexStoreId>(arm_obj_entity, bone_store_id);

    // ECS Scale stays vec3{1} for all bone entities so parent scale never displaces FK child positions.
    // Per-bone bone length is stored in BoneDisplayScale and baked into the mesh scale at GPU write time.
    // BoneOctahedron(1.f) is a unit mesh scaled by bone_length so rendered length = bone_length.
    // Non-leaf bones: length from nearest child distance. Leaf bones: inherit parent's length.
    // Bones are topologically sorted, so parent length is always resolved before children.
    static constexpr float MinBoneLength = 0.004f;
    std::vector<float> bone_scales(n, 0.f);
    // Non-leaf bones: minimum distance to any child (ignoring near-zero distances).
    for (uint32_t i = 0; i < n; ++i) {
        float min_child_dist = std::numeric_limits<float>::max();
        for (uint32_t j = 0; j < n; ++j) {
            if (armature.Bones[j].ParentIndex == i) {
                const float d = glm::length(vec3{armature.Bones[j].RestWorld[3]} - vec3{armature.Bones[i].RestWorld[3]});
                if (d > MinBoneLength) min_child_dist = std::min(min_child_dist, d);
            }
        }
        if (min_child_dist < std::numeric_limits<float>::max()) bone_scales[i] = min_child_dist;
    }
    // Leaf/zero-length bones: inherit parent's scale, or fall back to 1.0.
    for (uint32_t i = 0; i < n; ++i) {
        if (bone_scales[i] == 0.f) {
            bone_scales[i] = armature.Bones[i].ParentIndex != InvalidBoneIndex ? bone_scales[armature.Bones[i].ParentIndex] : 1.f;
        }
    }

    std::vector<entt::entity> bone_entities(n);
    for (uint32_t i = 0; i < n; ++i) {
        const auto &bone = armature.Bones[i];
        const auto bone_entity = R.create();
        R.emplace<BoneIndex>(bone_entity, i);
        R.emplace<SubElementOf>(bone_entity, arm_obj_entity);
        R.emplace<Instance>(bone_entity, arm_obj_entity);
        R.emplace<Name>(bone_entity, bone.Name);
        R.emplace<BoneDisplayScale>(bone_entity, bone_scales[i]);
        SetTransform(R, bone_entity, {bone.RestLocal.P, bone.RestLocal.R, vec3{1}});
        bone_entities[i] = bone_entity;
    }

    // Set up SceneNode hierarchy (bones are topologically sorted: parents before children).
    // Use "parent without inverse" so bone Position.Value represents local-frame offsets
    // (FK cascade: WorldMatrix = parent.WorldMatrix * LocalMatrix, no ParentInverse baked in).
    for (uint32_t i = 0; i < n; ++i) {
        const auto parent = armature.Bones[i].ParentIndex == InvalidBoneIndex ? arm_obj_entity : bone_entities[armature.Bones[i].ParentIndex];
        SetParent(R, bone_entities[i], parent);
        R.emplace_or_replace<ParentInverse>(bone_entities[i], I4); // Parent without inverse
        SetVisible(bone_entities[i], true); // Must precede UpdateWorldTransform
        UpdateWorldTransform(R, bone_entities[i]);
    }
    auto &arm_obj = R.get<ArmatureObject>(arm_obj_entity);
    arm_obj.BoneEntities = std::move(bone_entities);

    // Create shared joint sphere disc mesh (once per armature)
    {
        auto sphere_data = primitive::BoneSphereDisc();
        const auto sphere_store_id = Meshes->AllocateVertexBuffer(sphere_data.Mesh.Positions, {}).first;
        const auto sphere_tri_indices = flatten_tri_indices(sphere_data.Mesh.Faces);
        const auto sphere_vertex_count = static_cast<uint32_t>(sphere_data.Mesh.Positions.size());
        const auto joint_entity = R.create();
        R.emplace<BoneJoint>(joint_entity);
        R.emplace<MeshBuffers>(
            joint_entity, Meshes->GetVerticesRange(sphere_store_id),
            Buffers->CreateIndices(sphere_tri_indices, IndexKind::Face),
            Buffers->CreateIndices(sphere_data.OutlineIndices, IndexKind::Edge),
            Buffers->CreateIndices(iota(0u, sphere_vertex_count) | to<std::vector>(), IndexKind::Vertex)
        );
        R.emplace<VertexStoreId>(joint_entity, sphere_store_id);

        // Each bone gets a head + tail joint sphere instance
        const uint32_t joint_count = 2 * n;

        R.emplace<ModelsBuffer>(joint_entity, CreateModelsBuffer(Buffers->Ctx, joint_count));

        // Create joint sphere instance entities
        auto &joint_models = R.get<ModelsBuffer>(joint_entity);
        uint32_t joint_idx = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const auto bone_entity = arm_obj.BoneEntities[i];

            // Head joint
            const auto head_entity = R.create();
            R.emplace<SubElementOf>(head_entity, arm_obj_entity);
            R.emplace<Instance>(head_entity, joint_entity);
            R.emplace<BoneSubPartOf>(head_entity, bone_entity, false);
            const uint32_t head_oid = NextObjectId++;
            R.emplace<RenderInstance>(head_entity, joint_entity, joint_idx, head_oid);
            joint_models.ObjectIds.Update(as_bytes(head_oid), joint_idx * sizeof(uint32_t));
            ++joint_idx;

            // Tail joint
            const auto tail_entity = R.create();
            R.emplace<SubElementOf>(tail_entity, arm_obj_entity);
            R.emplace<Instance>(tail_entity, joint_entity);
            R.emplace<BoneSubPartOf>(tail_entity, bone_entity, true);
            const uint32_t tail_oid = NextObjectId++;
            R.emplace<RenderInstance>(tail_entity, joint_entity, joint_idx, tail_oid);
            joint_models.ObjectIds.Update(as_bytes(tail_oid), joint_idx * sizeof(uint32_t));
            ++joint_idx;

            R.emplace<BoneJointEntities>(bone_entity, head_entity, tail_entity);
        }

        arm_obj.JointEntity = joint_entity;
    }
}

void Scene::DestroyArmatureData(entt::entity arm_obj_entity) {
    auto &arm = R.get<ArmatureObject>(arm_obj_entity);
    if (arm.JointEntity != entt::null) {
        if (auto *mb = R.try_get<MeshBuffers>(arm.JointEntity)) Buffers->Release(*mb);
        if (auto *ref = R.try_get<VertexStoreId>(arm.JointEntity)) Meshes->Release(ref->StoreId);
        R.remove<MeshBuffers, VertexStoreId, ModelsBuffer>(arm.JointEntity);
        R.destroy(arm.JointEntity);
        arm.JointEntity = entt::null;
    }
    if (auto *mb = R.try_get<MeshBuffers>(arm_obj_entity)) Buffers->Release(*mb);
    if (auto *adj = R.try_get<BoneAdjacencyIndices>(arm_obj_entity)) Buffers->EdgeIndexBuffer.Release(adj->Indices);
    if (auto *ref = R.try_get<VertexStoreId>(arm_obj_entity)) Meshes->Release(ref->StoreId);
    R.remove<MeshBuffers, VertexStoreId, ModelsBuffer, BoneAdjacencyIndices>(arm_obj_entity);
}

void Scene::RebuildBoneStructure(entt::entity arm_data_entity) {
    auto &armature = R.get<Armature>(arm_data_entity);
    armature.FinalizeStructure();
    armature.RecomputeRestWorld();

    const uint32_t n = armature.Bones.size();
    const Transform identity_delta{vec3{0}, quat{1, 0, 0, 0}, vec3{1}};

    // Reset pose state to identity. Structural edits happen in Edit mode where pose deltas aren't active;
    // animation will re-evaluate from keyframes when switching back to Pose mode.
    if (auto *pose_state = R.try_get<ArmaturePoseState>(arm_data_entity)) {
        pose_state->BonePoseDelta.assign(n, identity_delta);
        pose_state->BoneUserOffset.assign(n, identity_delta);
        pose_state->Dirty = true;
    }

    if (auto *anim = R.try_get<ArmatureAnimation>(arm_data_entity)) {
        for (auto &clip : anim->Clips) armature.ResolveAnimationIndices(clip);
    }

    // Force animation re-evaluation on next frame so pose is restored when leaving Edit mode.
    LastEvaluatedFrame = -1;
}

namespace {
// Non-leaf: minimum distance to any child (ignoring near-zero). Leaf: inherit parent's scale, or 1.0.
float ComputeSingleBoneDisplayScale(const Armature &armature, uint32_t bone_index) {
    static constexpr float MinBoneLength = 0.004f;
    float min_child_dist = std::numeric_limits<float>::max();
    for (uint32_t j = 0; j < armature.Bones.size(); ++j) {
        if (armature.Bones[j].ParentIndex == bone_index) {
            const float d = glm::length(vec3{armature.Bones[j].RestWorld[3]} - vec3{armature.Bones[bone_index].RestWorld[3]});
            if (d > MinBoneLength) min_child_dist = std::min(min_child_dist, d);
        }
    }
    if (min_child_dist < std::numeric_limits<float>::max()) return min_child_dist;
    if (armature.Bones[bone_index].ParentIndex != InvalidBoneIndex) {
        return ComputeSingleBoneDisplayScale(armature, armature.Bones[bone_index].ParentIndex);
    }
    return 1.f;
}
} // namespace

// Create a single bone ECS entity with model buffer reservation, joint spheres, and scene hierarchy.
// Returns the new bone entity. Caller is responsible for selection.
entt::entity Scene::CreateSingleBoneInstance(entt::entity arm_obj_entity, BoneId bone_id) {
    auto &arm_obj = R.get<ArmatureObject>(arm_obj_entity);
    const auto &armature = R.get<const Armature>(arm_obj.Entity);
    const auto new_index = *armature.FindBoneIndex(bone_id);
    const auto &bone = armature.Bones[new_index];

    const auto bone_entity = R.create();
    R.emplace<BoneIndex>(bone_entity, new_index);
    R.emplace<SubElementOf>(bone_entity, arm_obj_entity);
    R.emplace<Instance>(bone_entity, arm_obj_entity);
    R.emplace<Name>(bone_entity, bone.Name);
    R.emplace<BoneDisplayScale>(bone_entity, ComputeSingleBoneDisplayScale(armature, new_index));
    SetTransform(R, bone_entity, {bone.RestLocal.P, bone.RestLocal.R, vec3{1}});

    const auto parent_entity = bone.ParentIndex == InvalidBoneIndex ? arm_obj_entity : arm_obj.BoneEntities[bone.ParentIndex];
    SetParent(R, bone_entity, parent_entity);
    R.emplace_or_replace<ParentInverse>(bone_entity, I4);

    R.patch<ModelsBuffer>(arm_obj_entity, [](auto &mb) { ReserveModelsBufferInstance(mb); });

    if (arm_obj.JointEntity != null_entity && R.valid(arm_obj.JointEntity)) {
        auto &joint_models = R.get<ModelsBuffer>(arm_obj.JointEntity);
        const uint32_t head_idx = uint32_t(joint_models.Buffer.UsedSize / sizeof(WorldTransform));
        const uint32_t tail_idx = head_idx + 1;

        joint_models.Buffer.Reserve((tail_idx + 1) * sizeof(WorldTransform));
        joint_models.ObjectIds.Reserve((tail_idx + 1) * sizeof(uint32_t));
        joint_models.InstanceStates.Reserve((tail_idx + 1) * sizeof(uint8_t));

        const auto head_entity = R.create();
        R.emplace<SubElementOf>(head_entity, arm_obj_entity);
        R.emplace<Instance>(head_entity, arm_obj.JointEntity);
        R.emplace<BoneSubPartOf>(head_entity, bone_entity, false);
        const uint32_t head_oid = NextObjectId++;
        R.emplace<RenderInstance>(head_entity, arm_obj.JointEntity, head_idx, head_oid);
        joint_models.ObjectIds.Update(as_bytes(head_oid), head_idx * sizeof(uint32_t));

        const auto tail_entity = R.create();
        R.emplace<SubElementOf>(tail_entity, arm_obj_entity);
        R.emplace<Instance>(tail_entity, arm_obj.JointEntity);
        R.emplace<BoneSubPartOf>(tail_entity, bone_entity, true);
        const uint32_t tail_oid = NextObjectId++;
        R.emplace<RenderInstance>(tail_entity, arm_obj.JointEntity, tail_idx, tail_oid);
        joint_models.ObjectIds.Update(as_bytes(tail_oid), tail_idx * sizeof(uint32_t));

        R.emplace<BoneJointEntities>(bone_entity, head_entity, tail_entity);
    }

    SetVisible(bone_entity, true);
    UpdateWorldTransform(R, bone_entity);

    arm_obj.BoneEntities.emplace_back(bone_entity);
    return bone_entity;
}

void Scene::AddBone() {
    const auto active_entity = FindActiveEntity(R);
    const auto arm_obj_entity = FindArmatureObject(R, active_entity);
    if (arm_obj_entity == entt::null) return;

    auto &armature = R.get<Armature>(R.get<ArmatureObject>(arm_obj_entity).Entity);

    // New bone: unparented, unit length, oriented along Y (up), at world origin.
    const auto arm_world_inv = glm::inverse(ToMatrix(R.get<WorldTransform>(arm_obj_entity)));
    const auto new_id = armature.AddBone("Bone", {}, {.P = vec3{arm_world_inv * vec4{0, 0, 0, 1}}});
    RebuildBoneStructure(R.get<ArmatureObject>(arm_obj_entity).Entity);

    const auto bone_entity = CreateSingleBoneInstance(arm_obj_entity, new_id);
    SelectBone(bone_entity);
    R.emplace_or_replace<BoneSelection>(bone_entity, false, true, false);
}

void Scene::ExtrudeBone() {
    const auto active_entity = FindActiveEntity(R);
    const auto arm_obj_entity = FindArmatureObject(R, active_entity);
    if (arm_obj_entity == entt::null) return;

    auto &arm_obj = R.get<ArmatureObject>(arm_obj_entity);
    auto &armature = R.get<Armature>(arm_obj.Entity);

    // Collect selected bones with their extrude type.
    // Blender logic: tip or body selected → extrude from tip (child); root-only → extrude from root (sibling).
    struct ExtrudeInfo {
        entt::entity BoneEntity;
        uint32_t BoneIndex;
        bool FromTip;
    };
    std::vector<ExtrudeInfo> to_extrude;
    for (const auto e : R.view<BoneSelection, BoneIndex>()) {
        if (R.get<SubElementOf>(e).Parent != arm_obj_entity) continue;
        const auto idx = R.get<BoneIndex>(e).Index;
        const auto *parts = R.try_get<const BoneSelection>(e);
        const bool root_only = parts && parts->Root && !parts->Tip && !parts->Body;
        to_extrude.emplace_back(ExtrudeInfo{e, idx, !root_only});
    }
    if (to_extrude.empty()) return;

    // For root extrude, skip if parent bone's tip is also selected (Blender conflict avoidance).
    std::erase_if(to_extrude, [&](const ExtrudeInfo &info) {
        if (info.FromTip) return false;
        const auto &bone = armature.Bones[info.BoneIndex];
        if (bone.ParentIndex == InvalidBoneIndex) return false;
        const auto parent_entity = arm_obj.BoneEntities[bone.ParentIndex];
        const auto *parent_parts = R.try_get<const BoneSelection>(parent_entity);
        return parent_parts && parent_parts->Tip;
    });
    if (to_extrude.empty()) return;

    // Create new bones in the Armature data.
    std::vector<BoneId> new_bone_ids;
    for (const auto &info : to_extrude) {
        const auto &bone = armature.Bones[info.BoneIndex];
        if (info.FromTip) {
            const float bone_length = R.get<BoneDisplayScale>(info.BoneEntity).Value;
            new_bone_ids.emplace_back(armature.AddBone("", bone.Id, {.P = vec3{0, bone_length, 0}}));
        } else {
            auto parent_id = bone.ParentBoneId == InvalidBoneId ? std::optional<BoneId>{} : std::optional<BoneId>{bone.ParentBoneId};
            new_bone_ids.emplace_back(armature.AddBone("", parent_id, bone.RestLocal));
        }
    }

    RebuildBoneStructure(arm_obj.Entity);

    // Deselect all bones — armature keeps its object-level Active/Selected.
    R.clear<BoneSelection, BoneActive>();

    // Create ECS entities for each new bone, select with tip only.
    // Zero display scale so the bone starts as a point (head=tail until moved).
    for (const auto id : new_bone_ids) {
        const auto bone_entity = CreateSingleBoneInstance(arm_obj_entity, id);
        R.get<BoneDisplayScale>(bone_entity).Value = 0.f;
        UpdateModelBuffer(R, bone_entity, R.get<WorldTransform>(bone_entity));
        R.emplace<BoneSelection>(bone_entity, false, true, false);
        R.emplace_or_replace<BoneActive>(bone_entity); // Last one wins as the active bone.
    }
    // Update parent bone display scales (they may have gained a child).
    for (const auto &info : to_extrude) {
        if (info.FromTip) {
            R.get<BoneDisplayScale>(info.BoneEntity).Value = ComputeSingleBoneDisplayScale(armature, info.BoneIndex);
            UpdateModelBuffer(R, info.BoneEntity, R.get<WorldTransform>(info.BoneEntity));
        }
    }
}

void Scene::DuplicateSelectedBones() {
    const auto active_entity = FindActiveEntity(R), arm_obj_entity = FindArmatureObject(R, active_entity);
    if (arm_obj_entity == entt::null) return;

    const auto &arm_obj = R.get<ArmatureObject>(arm_obj_entity);
    auto &armature = R.get<Armature>(arm_obj.Entity);

    struct DupInfo {
        entt::entity Entity;
        BoneId Id, NewId{InvalidBoneId};
    };
    std::vector<DupInfo> to_duplicate;
    for (const auto e : R.view<BoneSelection, BoneIndex>()) {
        if (R.get<SubElementOf>(e).Parent != arm_obj_entity) continue;
        to_duplicate.emplace_back(DupInfo{e, armature.Bones[R.get<BoneIndex>(e).Index].Id});
    }
    if (to_duplicate.empty()) return;

    auto unique_bone_name = [&](std::string_view base) {
        for (uint32_t i = 1;; ++i) {
            auto candidate = std::format("{}.{:03d}", base, i);
            if (!any_of(armature.Bones, [&](const auto &b) { return b.Name == candidate; })) return candidate;
        }
    };

    std::unordered_map<BoneId, BoneId> orig_to_new;
    for (auto &info : to_duplicate) {
        const auto &bone = armature.Bones[*armature.FindBoneIndex(info.Id)];
        auto parent_id = bone.ParentBoneId == InvalidBoneId ? std::optional<BoneId>{} : std::optional<BoneId>{bone.ParentBoneId};
        info.NewId = armature.AddBone(unique_bone_name(bone.Name), parent_id, bone.RestLocal);
        orig_to_new[info.Id] = info.NewId;
    }

    // If both a bone and its parent were duplicated, point the duplicate child to the duplicate parent.
    for (const auto &info : to_duplicate) {
        auto &new_bone = armature.Bones[*armature.FindBoneIndex(info.NewId)];
        if (auto it = orig_to_new.find(new_bone.ParentBoneId); it != orig_to_new.end()) {
            new_bone.ParentBoneId = it->second;
        }
    }

    RebuildBoneStructure(arm_obj.Entity);

    R.clear<BoneSelection, BoneActive>();
    entt::entity last_bone{};
    for (const auto &info : to_duplicate) {
        last_bone = CreateSingleBoneInstance(arm_obj_entity, info.NewId);
        R.get<BoneDisplayScale>(last_bone).Value = R.get<BoneDisplayScale>(info.Entity).Value;
        UpdateModelBuffer(R, last_bone, R.get<WorldTransform>(last_bone));
        R.emplace<BoneSelection>(last_bone);
    }
    R.emplace<BoneActive>(last_bone);

    StartScreenTransform = TransformGizmo::TransformType::Translate;
}

void Scene::DeleteSelectedBones() {
    const auto active_entity = FindActiveEntity(R);
    const auto arm_obj_entity = FindArmatureObject(R, active_entity);
    if (arm_obj_entity == entt::null) return;

    auto &arm_obj = R.get<ArmatureObject>(arm_obj_entity);
    auto &armature = R.get<Armature>(arm_obj.Entity);

    // Destroy bone entities in reverse topological order (children before parents).
    // Reparent scene graph children to their grandparent, adjusting local transforms
    // to preserve world position (same composition as Armature::RemoveBone applies to RestLocal).
    std::vector<uint32_t> to_delete;
    for (const auto e : R.view<BoneSelection, BoneIndex>()) {
        if (R.get<SubElementOf>(e).Parent == arm_obj_entity) to_delete.emplace_back(R.get<BoneIndex>(e).Index);
    }
    if (to_delete.empty()) return;
    std::sort(to_delete.rbegin(), to_delete.rend());

    for (const auto idx : to_delete) {
        const auto bone_entity = arm_obj.BoneEntities[idx];
        const auto &bone = armature.Bones[idx];
        const auto grandparent = bone.ParentIndex == InvalidBoneIndex ? arm_obj_entity : arm_obj.BoneEntities[bone.ParentIndex];

        // Destroy joint sphere entities for this bone.
        if (auto *joints = R.try_get<BoneJointEntities>(bone_entity)) {
            if (joints->Head != null_entity) {
                SetVisible(joints->Head, false);
                R.destroy(joints->Head);
            }
            if (joints->Tail != null_entity) {
                SetVisible(joints->Tail, false);
                R.destroy(joints->Tail);
            }
            R.remove<BoneJointEntities>(bone_entity);
        }

        std::vector<entt::entity> children;
        for (const auto child : Children{&R, bone_entity}) children.emplace_back(child);
        for (const auto child : children) {
            SetTransform(R, child, ComposeLocalTransforms(bone.RestLocal, GetTransform(R, child)));
            ClearParent(R, child);
            SetParent(R, child, grandparent);
            R.emplace_or_replace<ParentInverse>(child, I4);
            UpdateWorldTransform(R, child);
        }

        ClearParent(R, bone_entity);
        SetVisible(bone_entity, false);
        R.destroy(bone_entity);
    }

    for (const auto idx : to_delete) {
        armature.RemoveBone(armature.Bones[idx].Id);
        arm_obj.BoneEntities.erase(arm_obj.BoneEntities.begin() + idx);
    }

    RebuildBoneStructure(arm_obj.Entity);

    // Update surviving bones with new dense indices.
    for (uint32_t i = 0; i < arm_obj.BoneEntities.size(); ++i) R.get<BoneIndex>(arm_obj.BoneEntities[i]).Index = i;

    if (arm_obj.BoneEntities.empty()) DestroyArmatureData(arm_obj_entity);

    Select(arm_obj_entity);
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
    CreateBoneInstances(entity, data_entity);
    return entity;
}

entt::entity Scene::CreateExtrasBufferEntity(ExtrasWireframe &&wireframe) {
    const auto buffer_entity = R.create();
    const auto [store_id, vertices] = Meshes->AllocateVertexBuffer(wireframe.Data.Positions, {});

    R.emplace<ModelsBuffer>(buffer_entity, CreateModelsBuffer(Buffers->Ctx));
    R.emplace<MeshBuffers>(
        buffer_entity, Meshes->GetVerticesRange(store_id),
        SlottedRange{}, // No face indices
        Buffers->CreateIndices(wireframe.Data.CreateEdgeIndices(), IndexKind::Edge),
        SlottedRange{} // No vertex indices
    );
    R.emplace<ObjectExtrasTag>(buffer_entity);
    R.emplace<VertexStoreId>(buffer_entity, store_id);
    if (!wireframe.VertexClasses.empty()) {
        const auto range = Buffers->VertexClassBuffer.Allocate(std::span<const uint8_t>(wireframe.VertexClasses));
        R.emplace<VertexClass>(buffer_entity, range.Offset);
    }
    return buffer_entity;
}

entt::entity Scene::CreateExtrasObject(ExtrasWireframe &&wireframe, ObjectType type, ObjectCreateInfo info, const std::string &default_name) {
    const auto buffer_entity = CreateExtrasBufferEntity(std::move(wireframe));

    const auto entity = R.create();
    R.emplace<ObjectKind>(entity, type);
    R.emplace<Instance>(entity, buffer_entity);
    SetTransform(R, entity, info.Transform);
    R.emplace<Name>(entity, CreateName(R, info.Name.empty() ? default_name : info.Name));

    R.patch<ModelsBuffer>(buffer_entity, [](auto &mb) { ReserveModelsBufferInstance(mb); });
    SetVisible(entity, true);
    ApplySelectBehavior(entity, info.Select);
    return entity;
}

entt::entity Scene::AddCamera(ObjectCreateInfo info) {
    Camera camera{DefaultPerspectiveCamera()};
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
    const auto buffer_entity = R.get<Instance>(entity).Entity;
    const auto &ri = R.get<const RenderInstance>(entity);
    light.TransformSlotOffset = {R.get<const ModelsBuffer>(buffer_entity).Buffer.Slot, ri.BufferIndex};
    Buffers->SetLight(light_index, light);
    return entity;
}

std::pair<entt::entity, entt::entity> Scene::AddMesh(MeshData &&data, std::optional<MeshInstanceCreateInfo> info) {
    return AddMesh(Meshes->CreateMesh(std::move(data), {}, {}), std::move(info));
}

std::pair<entt::entity, entt::entity> Scene::AddMesh(const std::filesystem::path &path, std::optional<MeshInstanceCreateInfo> info) {
    auto result = Meshes->LoadMesh(path);
    if (!result) throw std::runtime_error(result.error());

    if (!result->Materials.empty()) {
        auto &texture_store = *Textures;
        std::unordered_map<std::string, uint32_t> texture_slot_cache;
        const auto resolve_texture_slot =
            [&](
                const std::optional<std::filesystem::path> &source_texture_path,
                TextureColorSpace color_space,
                std::string_view material_name, std::string_view texture_label
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

    if (!R.all_of<Instance>(e)) {
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

    // Bone sub-entities (head/tail joints, bone instances) are not independently duplicable.
    if (R.all_of<BoneSubPartOf>(e)) return entt::null;

    // Object extras (Camera, Empty, Light) have Instance but create their own wireframe mesh.
    if (R.all_of<ObjectExtrasTag>(R.get<Instance>(e).Entity)) {
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

    const auto mesh_entity = R.get<Instance>(e).Entity;
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
    if (R.all_of<BoneSubPartOf>(e)) return entt::null;
    if (!R.all_of<Instance>(e)) {
        const auto select_behavior = info ? info->Select : (R.all_of<Selected>(e) ? MeshInstanceCreateInfo::SelectBehavior::Additive : MeshInstanceCreateInfo::SelectBehavior::None);

        if (const auto *armature = R.try_get<ArmatureObject>(e)) {
            const auto e_new = R.create();
            R.emplace<Name>(e_new, !info || info->Name.empty() ? CreateName(R, std::format("{}_copy", GetName(R, e))) : CreateName(R, info->Name));
            R.emplace<ObjectKind>(e_new, ObjectType::Armature);
            R.emplace<ArmatureObject>(e_new, armature->Entity);
            SetTransform(R, e_new, info ? info->Transform : GetTransform(R, e));

            ApplySelectBehavior(e_new, select_behavior);
            CreateBoneInstances(e_new, armature->Entity);
            return e_new;
        }

        return AddEmpty({
            .Name = info && !info->Name.empty() ? info->Name : std::format("{}_copy", GetName(R, e)),
            .Transform = info ? info->Transform : GetTransform(R, e),
            .Select = select_behavior,
        });
    }

    const auto mesh_entity = R.get<Instance>(e).Entity;
    const auto e_new = R.create();
    {
        uint instance_count = 0; // Count instances for naming (first duplicated instance is _1, etc.)
        for (const auto [_, instance] : R.view<Instance>().each()) {
            if (instance.Entity == mesh_entity) ++instance_count;
        }
        R.emplace<Name>(e_new, !info || info->Name.empty() ? std::format("{}_{}", GetName(R, e), instance_count) : CreateName(R, info->Name));
    }
    R.emplace<Instance>(e_new, mesh_entity);
    R.emplace<ObjectKind>(e_new, ObjectType::Mesh);
    R.patch<ModelsBuffer>(mesh_entity, [](auto &mb) { ReserveModelsBufferInstance(mb); });
    SetTransform(R, e_new, info ? info->Transform : GetTransform(R, e));
    SetVisible(e_new, !info || info->Visible);
    if (const auto *armature_modifier = R.try_get<ArmatureModifier>(e)) R.emplace<ArmatureModifier>(e_new, *armature_modifier);
    if (const auto *bone_attachment = R.try_get<BoneAttachment>(e)) R.emplace<BoneAttachment>(e_new, *bone_attachment);

    if (!info || info->Select == MeshInstanceCreateInfo::SelectBehavior::Additive) R.emplace<Selected>(e_new);
    else if (info->Select == MeshInstanceCreateInfo::SelectBehavior::Exclusive) Select(e_new);

    return e_new;
}

namespace {
bool IsBoneEditMode(const entt::registry &R, entt::entity scene_entity) {
    const auto mode = R.get<const SceneInteraction>(scene_entity).Mode;
    if (mode != InteractionMode::Edit) return false;
    return FindArmatureObject(R, FindActiveEntity(R)) != entt::null;
}
} // namespace

bool Scene::CanDuplicate() const {
    const auto mode = R.get<const SceneInteraction>(SceneEntity).Mode;
    if (mode == InteractionMode::Pose) return false;
    if (IsBoneEditMode(R, SceneEntity)) return !R.view<BoneSelection>().empty();
    return !R.storage<Selected>().empty();
}

bool Scene::CanDuplicateLinked() const {
    const auto mode = R.get<const SceneInteraction>(SceneEntity).Mode;
    if (mode == InteractionMode::Pose) return false;
    if (IsBoneEditMode(R, SceneEntity)) return false;
    return !R.storage<Selected>().empty();
}

bool Scene::CanDelete() const {
    const auto mode = R.get<const SceneInteraction>(SceneEntity).Mode;
    if (mode == InteractionMode::Pose) return false;
    if (IsBoneEditMode(R, SceneEntity)) return !R.view<BoneSelection>().empty();
    return !R.storage<Selected>().empty();
}

void Scene::Delete() {
    if (!CanDelete()) return;
    if (IsBoneEditMode(R, SceneEntity)) {
        DeleteSelectedBones();
        return;
    }
    std::vector<entt::entity> entities;
    for (const auto e : R.view<Selected>()) {
        if (!R.all_of<SubElementOf>(e)) entities.emplace_back(e);
    }
    for (const auto e : entities) Destroy(e);
}

void Scene::Duplicate() {
    if (!CanDuplicate()) return;
    if (IsBoneEditMode(R, SceneEntity)) {
        DuplicateSelectedBones();
        return;
    }
    const Timer timer{"Duplicate"};
    std::vector<entt::entity> entities;
    for (const auto e : R.view<Selected>()) entities.emplace_back(e);
    for (const auto e : entities) {
        const auto new_e = Duplicate(e);
        if (R.all_of<Active>(e)) {
            R.remove<Active>(e);
            R.emplace<Active>(new_e);
        }
        R.remove<Selected>(e);
    }
    StartScreenTransform = TransformGizmo::TransformType::Translate;
}

void Scene::DuplicateLinked() {
    if (!CanDuplicateLinked()) return;
    const Timer timer{"DuplicateLinked"};
    std::vector<entt::entity> entities;
    for (const auto e : R.view<Selected>()) entities.emplace_back(e);
    for (const auto e : entities) {
        const auto new_e = DuplicateLinked(e);
        if (R.all_of<Active>(e)) {
            R.remove<Active>(e);
            R.emplace<Active>(new_e);
        }
        R.remove<Selected>(e);
    }
    StartScreenTransform = TransformGizmo::TransformType::Translate;
}

void Scene::ClearMeshes() {
    std::vector<entt::entity> entities;
    for (const auto e : R.view<Instance>()) {
        if (!R.all_of<SubElementOf>(e)) entities.emplace_back(e);
    }
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

    entt::entity buffer_entity = entt::null;
    if (const auto *instance = R.try_get<Instance>(e)) {
        if (R.all_of<Mesh>(instance->Entity) || R.all_of<ObjectExtrasTag>(instance->Entity)) buffer_entity = instance->Entity;
        SetVisible(e, false);
    }
    std::vector<entt::entity> armature_data_entities;
    auto try_add_armature_data = [&](entt::entity data_entity) {
        if (R.valid(data_entity) && find(armature_data_entities, data_entity) == armature_data_entities.end()) {
            armature_data_entities.emplace_back(data_entity);
        }
    };
    if (const auto *armature = R.try_get<ArmatureObject>(e)) try_add_armature_data(armature->Entity);
    if (const auto *armature_modifier = R.try_get<ArmatureModifier>(e)) try_add_armature_data(armature_modifier->ArmatureEntity);
    if (const auto *bone_attachment = R.try_get<BoneAttachment>(e)) try_add_armature_data(bone_attachment->ArmatureEntity);

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

    if (R.all_of<ArmatureObject>(e)) {
        auto &arm = R.get<ArmatureObject>(e);
        auto destroy_visible = [&](entt::entity entity) {
            SetVisible(entity, false);
            R.destroy(entity);
        };
        for (const auto bone_entity : arm.BoneEntities) {
            if (auto *joints = R.try_get<BoneJointEntities>(bone_entity)) {
                if (joints->Head != entt::null) destroy_visible(joints->Head);
                if (joints->Tail != entt::null) destroy_visible(joints->Tail);
            }
            R.remove<BoneJointEntities>(bone_entity);
        }

        // Destroy children before parents (reverse of topological order) so ClearParent
        // can access the parent's SceneNode to unlink the child.
        for (auto it = arm.BoneEntities.rbegin(); it != arm.BoneEntities.rend(); ++it) {
            ClearParent(R, *it);
            destroy_visible(*it);
        }
        DestroyArmatureData(e);
    }

    R.destroy(e);

    // If this was the last instance, destroy the buffer entity
    if (R.valid(buffer_entity)) {
        const auto has_instances = any_of(
            R.view<Instance>().each(),
            [buffer_entity](const auto &entry) { return std::get<1>(entry).Entity == buffer_entity; }
        );
        if (!has_instances) {
            if (auto *mesh_buffers = R.try_get<MeshBuffers>(buffer_entity)) {
                if (const auto *vcr = R.try_get<VertexClass>(buffer_entity)) {
                    Buffers->VertexClassBuffer.Release({vcr->Offset, mesh_buffers->Vertices.Count});
                }
                Buffers->Release(*mesh_buffers);
            }
            if (const auto *vs = R.try_get<VertexStoreId>(buffer_entity)) Meshes->Release(vs->StoreId);
            R.destroy(buffer_entity);
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
    const bool show_overlays = settings.ShowOverlays;
    const SPT fill_pipeline = settings.FaceColorMode == FaceColorMode::Mesh ? SPT::Fill : SPT::DebugNormals;
    const auto primary_edit_instances = is_edit_mode ? scene_selection::ComputePrimaryEditInstances(R) : std::unordered_map<entt::entity, entt::entity>{};
    const bool has_pending_transform = is_edit_mode && R.all_of<PendingTransform>(SceneEntity);
    const auto edit_transform_context = is_edit_mode ? EditTransformContext{scene_selection::ComputePrimaryEditInstances(R, false)} : EditTransformContext{};
    const auto is_silhouette_eligible = [&](entt::entity e) {
        if (!R.all_of<Instance, RenderInstance>(e)) return false;
        const auto buffer_entity = R.get<const Instance>(e).Entity;
        if (!R.valid(buffer_entity) || R.all_of<ObjectExtrasTag>(buffer_entity)) return false;
        // Bones get outlines from BoneWire/BoneSphereWire, not the screen-space silhouette system.
        if (R.all_of<ArmatureObject>(buffer_entity) || R.all_of<BoneJoint>(buffer_entity)) return false;
        const auto *mesh_buffers = R.try_get<const MeshBuffers>(buffer_entity);
        return mesh_buffers && mesh_buffers->FaceIndices.Count > 0;
    };

    std::unordered_set<entt::entity> silhouette_instances;
    if (is_edit_mode) {
        for (const auto [e, instance, ri] : R.view<const Instance, const Selected, const RenderInstance>().each()) {
            if (!is_silhouette_eligible(e)) continue;
            if (auto it = primary_edit_instances.find(instance.Entity); it == primary_edit_instances.end() || it->second != e) {
                silhouette_instances.insert(e);
            }
        }
    }

    std::unordered_set<entt::entity> excitable_mesh_entities;
    if (is_excite_mode) {
        for (const auto [e, instance, excitable] : R.view<const Instance, const Excitable>().each()) {
            excitable_mesh_entities.emplace(instance.Entity);
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
            if (const auto *instance = R.try_get<const Instance>(entity)) mesh_entity = instance->Entity;
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

    const bool has_object_silhouette_selection =
        any_of(R.view<const Selected, const Instance, const RenderInstance>().each(), [&](const auto &entry) { return is_silhouette_eligible(std::get<0>(entry)); });
    // Pose mode silhouettes work exactly like Object mode: selected/active instances get silhouettes.
    const bool render_silhouette = (show_overlays && settings.ShowOutlineSelected) &&
        (is_edit_mode ? !silhouette_instances.empty() : has_object_silhouette_selection);

    DrawListBuilder draw_list;
    DrawBatchInfo fill_batch_opaque{}, fill_batch_blend{}, edge_quad_batch{}, wire_line_batch{}, point_batch{};
    DrawBatchInfo extras_line_batch{}, silhouette_batch{};
    DrawBatchInfo bone_fill_batch{}, bone_wire_batch{}, bone_sphere_fill_batch{}, bone_sphere_wire_batch{};
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
            const auto mesh_entity = R.get<Instance>(e).Entity;
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
                    if (R.all_of<ArmatureObject>(entity) || R.all_of<BoneJoint>(entity)) continue; // drawn in bone batches after depth clear
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

    // Build bone batches for X-ray rendering (drawn after a mid-pass depth clear so bones are never occluded by scene meshes)
    // Only draw bones for: active armature in Edit/Pose mode, selected armatures in Object mode.
    const auto should_draw_armature_bones = [&](entt::entity arm_obj_entity) {
        if (is_wireframe_mode) return true; // Wireframe mode: always show bone outlines
        const bool is_bone_mode = is_edit_mode || interaction_mode == InteractionMode::Pose;
        if (is_bone_mode) return R.all_of<Active>(arm_obj_entity);
        return R.all_of<Selected>(arm_obj_entity);
    };
    // Map a BoneJoint entity back to its owning armature object entity.
    const auto find_joint_owner = [&](entt::entity joint_entity) -> entt::entity {
        for (const auto [e, arm_obj] : R.view<const ArmatureObject>().each()) {
            if (arm_obj.JointEntity == joint_entity) return e;
        }
        return entt::null;
    };
    if (show_overlays && settings.ShowBones) {
        bone_fill_batch = draw_list.BeginBatch();
        for (const auto [entity, arm_obj, mesh_buffers, models] : R.view<const ArmatureObject, const MeshBuffers, const ModelsBuffer>().each()) {
            if (mesh_buffers.FaceIndices.Count == 0) continue;
            auto fill_draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models);
            fill_draw.InstanceStateSlot = models.InstanceStates.Slot;
            AppendDraw(draw_list, bone_fill_batch, mesh_buffers.FaceIndices, models, fill_draw);
        }
        bone_wire_batch = draw_list.BeginBatch();
        for (const auto [entity, arm_obj, mesh_buffers, models] : R.view<const ArmatureObject, const MeshBuffers, const ModelsBuffer>().each()) {
            if (!should_draw_armature_bones(entity)) continue;
            if (const auto *adj = R.try_get<const BoneAdjacencyIndices>(entity)) {
                auto wire_draw = MakeDrawData(mesh_buffers.Vertices, adj->Indices, models);
                wire_draw.InstanceStateSlot = models.InstanceStates.Slot;
                AppendDraw(draw_list, bone_wire_batch, adj->Indices, models, wire_draw);
            }
        }

        // Joint sphere batches
        bone_sphere_fill_batch = draw_list.BeginBatch();
        for (const auto [entity, mesh_buffers, models] : R.view<const BoneJoint, const MeshBuffers, const ModelsBuffer>().each()) {
            if (mesh_buffers.FaceIndices.Count == 0) continue;
            auto fill_draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models);
            fill_draw.InstanceStateSlot = models.InstanceStates.Slot;
            AppendDraw(draw_list, bone_sphere_fill_batch, mesh_buffers.FaceIndices, models, fill_draw);
        }
        bone_sphere_wire_batch = draw_list.BeginBatch();
        for (const auto [entity, mesh_buffers, models] : R.view<const BoneJoint, const MeshBuffers, const ModelsBuffer>().each()) {
            if (mesh_buffers.EdgeIndices.Count == 0) continue;
            const auto owner = find_joint_owner(entity);
            if (owner != entt::null && !should_draw_armature_bones(owner)) continue;
            auto wire_draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.EdgeIndices, models);
            wire_draw.InstanceStateSlot = models.InstanceStates.Slot;
            AppendDraw(draw_list, bone_sphere_wire_batch, mesh_buffers.EdgeIndices, models, wire_draw);
        }
    }

    // Edit mode edges: triangle quads with self-AA (matches Blender's overlay_edit_mesh_edge).
    // Wireframe/line mesh edges: GPU lines + LineAA composite (matches Blender's wireframe overlay).
    // Two separate loops so each batch's indirect commands are contiguous for drawIndexedIndirect.
    {
        edge_quad_batch = draw_list.BeginBatch();
        for (auto [entity, mesh_buffers, models, mesh] : R.view<MeshBuffers, ModelsBuffer, Mesh>().each()) {
            if (R.all_of<ObjectExtrasTag>(entity)) continue;
            if (R.all_of<ArmatureObject>(entity) || R.all_of<BoneJoint>(entity)) continue;
            if (mesh_buffers.EdgeIndices.Count == 0) continue;
            if (!is_edit_mode && !is_excite_mode) continue;
            const bool is_line_mesh = mesh_buffers.FaceIndices.Count == 0;
            if (is_line_mesh) continue; // Line meshes use wire_line_batch
            const auto deform = get_deform_slots(entity);
            auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.EdgeIndices, models, deform.BoneDeformOffset, deform.ArmatureDeformOffset, deform.MorphDeformOffset, deform.MorphTargetCount);
            draw.ElementStateSlotOffset = Meshes->GetEdgeStateRange(mesh.GetStoreId());
            const auto db = draw_list.Draws.size();
            if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                const uint32_t eq_count = mesh_buffers.EdgeIndices.Count * 3; // 6 verts per edge (2 triangles)
                AppendDraw(draw_list, edge_quad_batch, eq_count, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
            } else if (excitable_mesh_entities.contains(entity)) {
                const uint32_t eq_count = mesh_buffers.EdgeIndices.Count * 3;
                AppendDraw(draw_list, edge_quad_batch, eq_count, models, draw);
            }
            PatchMorphWeights(draw_list, db, deform);
            patch_edit_pending_local_transform(db, entity);
        }
    }
    {
        wire_line_batch = draw_list.BeginBatch();
        for (auto [entity, mesh_buffers, models, mesh] : R.view<MeshBuffers, ModelsBuffer, Mesh>().each()) {
            if (R.all_of<ObjectExtrasTag>(entity)) continue;
            if (R.all_of<ArmatureObject>(entity) || R.all_of<BoneJoint>(entity)) continue;
            if (mesh_buffers.EdgeIndices.Count == 0) continue;
            const bool is_line_mesh = mesh_buffers.FaceIndices.Count == 0;
            if (!is_line_mesh && !is_wireframe_mode) continue;
            const auto deform = get_deform_slots(entity);
            auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.EdgeIndices, models, deform.BoneDeformOffset, deform.ArmatureDeformOffset, deform.MorphDeformOffset, deform.MorphTargetCount);
            draw.ElementStateSlotOffset = Meshes->GetEdgeStateRange(mesh.GetStoreId());
            const auto db = draw_list.Draws.size();
            AppendDraw(draw_list, wire_line_batch, mesh_buffers.EdgeIndices, models, draw);
            PatchMorphWeights(draw_list, db, deform);
            patch_edit_pending_local_transform(db, entity);
        }
    }

    if (show_overlays && settings.ShowExtras) {
        AppendExtrasDraw(R, draw_list, extras_line_batch, [](auto &, const auto &) {});
    }

    {
        point_batch = draw_list.BeginBatch();
        for (auto [entity, mesh_buffers, models] : R.view<MeshBuffers, ModelsBuffer>().each()) {
            if (R.all_of<ArmatureObject>(entity) || R.all_of<BoneJoint>(entity)) continue;
            const bool is_point_mesh = mesh_buffers.FaceIndices.Count == 0 && mesh_buffers.EdgeIndices.Count == 0;
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
    const auto render_extent = ComputeRenderExtentPx(R.get<const ViewportExtent>(SceneEntity).Value);
    const auto &cb = *RenderCommandBuffer;
    cb.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(render_extent.width), float(render_extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, render_extent});
    const uint32_t transform_vertex_state_slot = is_edit_mode ? Meshes->GetVertexStateSlot() : InvalidSlot;
    auto record_draw_batch = [&](const PipelineRenderer &renderer, SPT spt, const DrawBatchInfo &batch) {
        if (batch.DrawCount == 0) return;
        const auto &pipeline = renderer.Bind(cb, spt);
        const DrawPassPushConstants pc{batch.DrawDataSlotOffset, transform_vertex_state_slot, InvalidSlot, InvalidSlot, InvalidSlot};
        cb.pushConstants(*pipeline.PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        cb.drawIndexedIndirect(*Buffers->RenderIndirect, batch.IndirectOffset, batch.DrawCount, sizeof(vk::DrawIndexedIndirectCommand));
    };
    auto record_pbr_batch = [&](const DrawBatchInfo &batch, bool opaque) {
        if (batch.DrawCount == 0) return;
        const auto layout = Pipelines->Main.Compiler.BindTargeted(cb, opaque);
        const DrawPassPushConstants pc{batch.DrawDataSlotOffset, transform_vertex_state_slot, InvalidSlot, InvalidSlot, InvalidSlot};
        cb.pushConstants(layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        cb.drawIndexedIndirect(*Buffers->RenderIndirect, batch.IndirectOffset, batch.DrawCount, sizeof(vk::DrawIndexedIndirectCommand));
    };
    const auto make_shader_read_barrier = [](vk::AccessFlags src_access, vk::ImageLayout layout, vk::Image image, const vk::ImageSubresourceRange &range) {
        return vk::ImageMemoryBarrier{src_access, vk::AccessFlagBits::eShaderRead, layout, layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, range};
    };
    const auto sync_fragment_shader_reads = [&](vk::PipelineStageFlags src_stages, auto &&barriers) {
        cb.pipelineBarrier(src_stages, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barriers);
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

        // Silhouette pass offscreen color writes -> edge pass fragment sampling.
        const std::array silhouette_to_edge_barriers{
            make_shader_read_barrier(vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eShaderReadOnlyOptimal, *silhouette.Resources->OffscreenImage.Image, ColorSubresourceRange),
        };
        sync_fragment_shader_reads(vk::PipelineStageFlagBits::eColorAttachmentOutput, silhouette_to_edge_barriers);

        const auto &silhouette_edge = Pipelines->SilhouetteEdge;
        static const std::vector<vk::ClearValue> edge_clear_values{{vk::ClearDepthStencilValue{1, 0}}, {Transparent}};
        const vk::Rect2D edge_rect{{0, 0}, ToExtent2D(silhouette_edge.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*silhouette_edge.Renderer.RenderPass, *silhouette_edge.Resources->Framebuffer, edge_rect, edge_clear_values}, vk::SubpassContents::eInline);
        const auto &silhouette_edo = silhouette_edge.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeDepthObject);
        const SilhouetteEdgeDepthObjectPushConstants edge_pc{SelectionHandles->SilhouetteSampler};
        cb.pushConstants(*silhouette_edo.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(edge_pc), &edge_pc);
        silhouette_edo.RenderQuad(cb);
        cb.endRenderPass();

        // Edge pass depth/color writes -> main pass silhouette sampling.
        const std::array edge_to_main_barriers{
            make_shader_read_barrier(vk::AccessFlagBits::eDepthStencilAttachmentWrite, vk::ImageLayout::eDepthStencilReadOnlyOptimal, *silhouette_edge.Resources->DepthImage.Image, DepthSubresourceRange),
            make_shader_read_barrier(vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eShaderReadOnlyOptimal, *silhouette_edge.Resources->OffscreenImage.Image, ColorSubresourceRange),
        };
        sync_fragment_shader_reads(vk::PipelineStageFlagBits::eLateFragmentTests | vk::PipelineStageFlagBits::eColorAttachmentOutput, edge_to_main_barriers);
    }

    const auto &main = Pipelines->Main;
    // Main rendering pass
    {
        // Three clear values: depth, color, linedata (linedata cleared to 0 so alpha=0 means "no line")
        const std::vector<vk::ClearValue> clear_values{
            {vk::ClearDepthStencilValue{1, 0}},
            {settings.ClearColor},
            {vk::ClearColorValue{std::array<float, 4>{0, 0, 0, 0}}},
        };
        const vk::Rect2D rect{{0, 0}, ToExtent2D(main.Resources->ColorImage.Extent)};
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
                record_pbr_batch(fill_batch_opaque, true);
                record_pbr_batch(fill_batch_blend, false);
            } else {
                record_draw_batch(main.Renderer, fill_pipeline, fill_batch_opaque);
            }
        }
        // Edit mode edges as triangle quads with self-AA
        record_draw_batch(main.Renderer, SPT::EdgeQuad, edge_quad_batch);
        // Wireframe/line mesh edges as GPU lines (LineAA composite handles AA)
        record_draw_batch(main.Renderer, SPT::Line, wire_line_batch);
        // Vertex points (always recorded — batch is empty when nothing qualifies)
        record_draw_batch(main.Renderer, SPT::Point, point_batch);
        // Object extras (cameras, lights, empties)
        record_draw_batch(main.Renderer, SPT::ObjectExtrasLine, extras_line_batch);
    }

    // Silhouette edge color (rendered ontop of meshes)
    if (has_silhouette) {
        const auto &silhouette_edc = main.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeColor);
        // In mesh Edit mode, suppress active silhouette (element selection drives active state differently).
        // In armature Edit/Pose mode, the active bone gets the active-color silhouette.
        const auto active_entity = FindActiveEntity(R);
        const auto active_bone = FindActiveBone(R);
        const bool armature_mode = FindArmatureObject(R, active_entity) != entt::null;
        uint32_t active_object_id = 0;
        if (armature_mode && active_bone != entt::null) {
            if (R.all_of<RenderInstance>(active_bone)) {
                active_object_id = R.get<RenderInstance>(active_bone).ObjectId;
            }
        } else if (!is_edit_mode) {
            if (active_entity != entt::null && R.all_of<RenderInstance>(active_entity)) {
                active_object_id = R.get<RenderInstance>(active_entity).ObjectId;
            }
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

    // Grid lines texture (drawn before bone depth clear so grid remains depth-tested against scene meshes)
    if (show_overlays && settings.ShowGrid) {
        // MoltenVK/Metal workaround: the grid shader writes gl_FragDepth (disabling early-z),
        // and late fragment tests don't correctly read unresolved fast-cleared depth on tile-based GPUs.
        // Re-clear depth when no triangle draws with depth write have resolved the fast-clear state.
        if (!has_silhouette && fill_batch_opaque.DrawCount == 0) {
            const vk::ClearAttachment grid_depth_resolve{vk::ImageAspectFlagBits::eDepth, 0, vk::ClearDepthStencilValue{1.f, 0}};
            const vk::ClearRect grid_clear_rect{{{0, 0}, ToExtent2D(main.Resources->ColorImage.Extent)}, 0, 1};
            cb.clearAttachments(grid_depth_resolve, grid_clear_rect);
        }
        main.Renderer.ShaderPipelines.at(SPT::Grid).RenderQuad(cb);
    }

    { // Bone X-ray: clear depth so bones are never occluded by scene meshes (only mutually occlude each other)
        if (bone_fill_batch.DrawCount > 0 || bone_sphere_fill_batch.DrawCount > 0) {
            cb.bindIndexBuffer(*Buffers->IdentityIndexBuffer, 0, vk::IndexType::eUint32);
            const vk::ClearAttachment depth_clear{vk::ImageAspectFlagBits::eDepth, 0, vk::ClearDepthStencilValue{1.f, 0}};
            const auto extent = main.Resources->ColorImage.Extent;
            const vk::ClearRect clear_rect{{{0, 0}, ToExtent2D(extent)}, 0, 1};
            cb.clearAttachments(depth_clear, clear_rect);

            // In Object+wireframe mode, show only outlines (no fills).
            // In Edit/Pose+wireframe, fills are semitransparent and write far-plane depth (via shader) so wires are never occluded.
            const bool object_wireframe = is_wireframe_mode && interaction_mode == InteractionMode::Object;
            if (!object_wireframe) {
                record_draw_batch(main.Renderer, SPT::BoneFill, bone_fill_batch);
                record_draw_batch(main.Renderer, SPT::BoneSphereFill, bone_sphere_fill_batch);
            }
            // In non-wireframe Object mode, "Outline selected" off suppresses bone wire outlines.
            // In wireframe+Object mode, wires are the only bone visualization so always show them.
            const bool hide_bone_outlines = !is_wireframe_mode && interaction_mode == InteractionMode::Object &&
                (!show_overlays || !settings.ShowOutlineSelected);
            if (!hide_bone_outlines) {
                record_draw_batch(main.Renderer, SPT::BoneWire, bone_wire_batch);
                record_draw_batch(main.Renderer, SPT::BoneSphereWire, bone_sphere_wire_batch);
            }
        }
    }

    cb.endRenderPass();

    { // Line AA composite pass: blends anti-aliased lines from LineDataImage onto ColorImage → FinalColorImage
        const vk::ClearValue clear_value{vk::ClearColorValue{std::array<float, 4>{0, 0, 0, 1}}};
        const vk::Rect2D rect{{0, 0}, ToExtent2D(main.Resources->FinalColorImage.Extent)};
        cb.beginRenderPass({*main.LineAARenderPass, *main.Resources->LineAAFramebuffer, rect, clear_value}, vk::SubpassContents::eInline);
        const struct {
            uint32_t ColorSamplerSlot, LineDataSamplerSlot;
        } line_aa_pc{SelectionHandles->ColorSampler, SelectionHandles->LineDataSampler};
        cb.pushConstants(*main.LineAAComposite.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(line_aa_pc), &line_aa_pc);
        main.LineAAComposite.RenderQuad(cb);
        cb.endRenderPass();
    }

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
                for (auto [entity, mesh_buffers, models] : R.view<MeshBuffers, ModelsBuffer>().each()) {
                    if (R.all_of<ObjectExtrasTag>(entity) || R.all_of<BoneJoint>(entity)) continue;
                    const auto *indices = filter(mesh_buffers);
                    if (!indices || indices->Count == 0) continue;
                    const auto deform = get_deform_slots(entity);
                    auto draw = MakeDrawData(mesh_buffers.Vertices, *indices, models, deform.BoneDeformOffset, deform.ArmatureDeformOffset, deform.MorphDeformOffset, deform.MorphTargetCount);
                    draw.ObjectIdSlot = models.ObjectIds.Slot;
                    const auto db = draw_list.Draws.size();
                    if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                        AppendDraw(draw_list, batch, *indices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
                    } else {
                        AppendDraw(draw_list, batch, *indices, models, draw);
                    }
                    PatchMorphWeights(draw_list, db, deform);
                }
                return batch;
            };

            auto tri_batch = append_topology_batch([](const MeshBuffers &b) -> const SlottedRange * {
                return b.FaceIndices.Count > 0 ? &b.FaceIndices : nullptr;
            });
            auto line_batch = append_topology_batch([](const MeshBuffers &b) -> const SlottedRange * {
                return b.FaceIndices.Count == 0 && b.EdgeIndices.Count > 0 ? &b.EdgeIndices : nullptr;
            });
            auto point_batch = append_topology_batch([](const MeshBuffers &b) -> const SlottedRange * {
                return b.FaceIndices.Count == 0 && b.EdgeIndices.Count == 0 ? &b.VertexIndices : nullptr;
            });

            auto bone_sphere_batch = draw_list.BeginBatch();
            for (const auto [entity, mesh_buffers, models] : R.view<const BoneJoint, const MeshBuffers, const ModelsBuffer>().each()) {
                if (mesh_buffers.FaceIndices.Count == 0) continue;
                auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models);
                draw.ObjectIdSlot = models.ObjectIds.Slot;
                AppendDraw(draw_list, bone_sphere_batch, mesh_buffers.FaceIndices, models, draw);
            }

            DrawBatchInfo extras_batch;
            AppendExtrasDraw(R, draw_list, extras_batch, [](auto &draw, const auto &models) {
                draw.ObjectIdSlot = models.ObjectIds.Slot;
            });

            return {
                {SPT::SelectionFragmentTriangles, tri_batch},
                {SPT::SelectionFragmentLines, line_batch},
                {SPT::SelectionFragmentPoints, point_batch},
                {SPT::SelectionFragmentBoneSphere, bone_sphere_batch},
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
                const auto *inst = R.try_get<Instance>(e);
                if (!inst) continue;
                const auto buffer_entity = inst->Entity;
                const auto &mesh_buffers = R.get<MeshBuffers>(buffer_entity);
                const auto &models = R.get<ModelsBuffer>(buffer_entity);
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

    const auto render_extent = ComputeRenderExtentPx(R.get<const ViewportExtent>(SceneEntity).Value);
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(render_extent.width), float(render_extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, render_extent});

    auto record_draw_batch = [&](const PipelineRenderer &renderer, SPT spt, const DrawBatchInfo &batch) {
        if (batch.DrawCount == 0) return;
        const auto &pipeline = renderer.Bind(cb, spt);
        const DrawPassPushConstants pc{batch.DrawDataSlotOffset, InvalidSlot, SelectionHandles->HeadImage, Buffers->SelectionNodeBuffer.Slot, SelectionHandles->SelectionCounter};
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
    const auto element_pipeline = [xray_selection, write_bitset](Element el) -> SPT {
        if (el == Element::Vertex) {
            if (xray_selection) return write_bitset ? SPT::SelectionElementVertexXRayBitsetBox : SPT::SelectionElementVertexXRay;
            return write_bitset ? SPT::SelectionElementVertexBitsetBox : SPT::SelectionElementVertex;
        }
        if (el == Element::Edge) {
            if (xray_selection) return write_bitset ? SPT::SelectionElementEdgeXRayBitsetBox : SPT::SelectionElementEdgeXRay;
            return write_bitset ? SPT::SelectionElementEdgeBitsetBox : SPT::SelectionElementEdge;
        }
        if (xray_selection) return write_bitset ? SPT::SelectionElementFaceXRayBitsetBox : SPT::SelectionElementFaceXRay;
        return write_bitset ? SPT::SelectionElementFaceBitsetBox : SPT::SelectionElementFace;
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
                const auto *inst = R.try_get<Instance>(e);
                if (!inst) continue;
                const auto buffer_entity = inst->Entity;
                const auto &mesh_buffers = R.get<MeshBuffers>(buffer_entity);
                const auto &models = R.get<ModelsBuffer>(buffer_entity);
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

    const auto render_extent = ComputeRenderExtentPx(R.get<const ViewportExtent>(SceneEntity).Value);
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(render_extent.width), float(render_extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, render_extent});

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
        };
        auto draw_with = [&](SPT spt) {
            const auto &pipeline = selection.Renderer.Bind(cb, spt);
            cb.pushConstants(*pipeline.PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(element_pc), &element_pc);
            cb.drawIndexedIndirect(*Buffers->SelectionIndirect, element_batch.IndirectOffset, element_batch.DrawCount, sizeof(vk::DrawIndexedIndirectCommand));
        };
        draw_with(element_pipeline(element));
        if (write_bitset && xray_selection) {
            // X-Ray face: point pass catches edge-on faces (zero projected triangle area).
            if (element == Element::Face) draw_with(SPT::SelectionElementFaceXRayVertsBitsetBox);
            // X-Ray edge: point pass catches near/zero-length projected edges.
            if (element == Element::Edge) draw_with(SPT::SelectionElementEdgeXRayVertsBitsetBox);
        }
    }
    cb.endRenderPass();

    if (write_bitset) {
        // Ensure fragment shader writes to the bitset are visible to the host after the fence.
        // Scope the barrier to the written range.
        const auto element_count = fold_left(ranges, uint32_t{0}, [](uint32_t total, const auto &r) { return std::max(total, r.Offset + r.Count); });
        const vk::DeviceSize bitset_bytes = ((element_count + 31) / 32) * sizeof(uint32_t);
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eHost, {}, {},
            vk::BufferMemoryBarrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead, {}, {}, *Buffers->SelectionBitsetBuffer, 0, bitset_bytes},
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

void Scene::DispatchUpdateSelectionStates(std::span<const ElementRange> ranges, Element element) {
    if (ranges.empty() || element == Element::None) return;

    auto cb = *ClickCommandBuffer;
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Ensure bitset writes from previous render/compute are visible to this compute shader.
    const vk::MemoryBarrier input_barrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead};
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader, {}, input_barrier, {}, {});

    const auto &compute = Pipelines->UpdateSelectionState;
    cb.bindPipeline(vk::PipelineBindPoint::eCompute, *compute.Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *compute.PipelineLayout, 0, compute.GetDescriptorSet(), {});

    for (const auto &range : ranges) {
        const auto &mesh = R.get<const Mesh>(range.MeshEntity);
        const auto &mesh_buffers = R.get<const MeshBuffers>(range.MeshEntity);
        const auto *active_element = R.try_get<const MeshActiveElement>(range.MeshEntity);

        uint32_t state_slot, state_offset;
        if (element == Element::Vertex) {
            state_slot = Meshes->GetVertexStateSlot();
            state_offset = mesh_buffers.Vertices.Offset;
        } else if (element == Element::Edge) {
            const auto edge_range = Meshes->GetEdgeStateRange(mesh.GetStoreId());
            state_slot = edge_range.Slot;
            state_offset = edge_range.Offset;
        } else {
            const auto face_range = Meshes->GetFaceStateRange(mesh.GetStoreId());
            state_slot = face_range.Slot;
            state_offset = face_range.Offset;
        }

        const UpdateSelectionStatePushConstants pc{
            .BitsetSlot = SelectionHandles->SelectionBitset,
            .BitsetOffset = range.Offset,
            .StateSlot = state_slot,
            .StateOffset = state_offset,
            .ElementCount = range.Count,
            .ActiveHandle = active_element ? active_element->Handle : InvalidOffset,
            .EdgeMode = element == Element::Edge ? 1u : 0u,
        };
        cb.pushConstants(*compute.PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        cb.dispatch((range.Count + 255) / 256, 1, 1);
    }

    const vk::MemoryBarrier output_barrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead};
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, output_barrier, {}, {});
    cb.end();

    vk::SubmitInfo submit;
    submit.setCommandBuffers(cb);
    Vk.Queue.submit(submit, *OneShotFence);
    WaitFor(*OneShotFence, Vk.Device);
}

void Scene::ApplySelectionStateUpdate(std::span<const ElementRange> ranges, Element element) {
    DispatchUpdateSelectionStates(ranges, element);
    if (element == Element::Vertex) {
        // Vertex states are already updated by the GPU compute above; derive edge/face states from them directly.
        for (const auto &range : ranges) {
            const auto &mesh = R.get<const Mesh>(range.MeshEntity);
            Meshes->UpdateEdgeStatesFromVertices(mesh);
            Meshes->UpdateFaceStatesFromVertices(mesh);
        }
    } else if (element == Element::Face || element == Element::Edge) {
        const auto *bits = reinterpret_cast<const uint32_t *>(Buffers->SelectionBitsetBuffer.GetMappedData().data());
        for (const auto &range : ranges) {
            const auto &mesh = R.get<const Mesh>(range.MeshEntity);
            const auto selected_handles = scene_selection::ScanBitsetRange(bits, range.Offset, range.Count);
            std::optional<uint32_t> active_handle;
            if (const auto *active = R.try_get<const MeshActiveElement>(range.MeshEntity); active && active->Handle < range.Count) {
                active_handle = active->Handle;
            }
            if (element == Element::Face) Meshes->UpdateEdgeStatesFromFaces(mesh, selected_handles, active_handle);
            if (element == Element::Edge) Meshes->UpdateFaceStatesFromEdges(mesh);
            Meshes->UpdateVertexStatesFromElements(mesh, selected_handles, element, active_handle);
        }
    }
    ElementStatesDirty = true;
}

std::vector<Scene::ElementRange> Scene::GetBitsetRangesForSelected() const {
    std::vector<ElementRange> ranges;
    for (const auto mesh_entity : scene_selection::GetSelectedMeshEntities(R)) {
        if (const auto *br = R.try_get<const MeshSelectionBitsetRange>(mesh_entity); br && br->Count > 0) {
            ranges.emplace_back(mesh_entity, br->Offset, br->Count);
        }
    }
    return ranges;
}

void Scene::RunBoxSelectElements(std::span<const ElementRange> ranges, Element element, std::pair<uvec2, uvec2> box_px, bool is_additive) {
    if (ranges.empty()) return;

    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return;

    const Timer timer{"RunBoxSelectElements"};
    const auto element_count = fold_left(
        ranges, uint32_t{0},
        [](uint32_t total, const auto &range) { return std::max(total, range.Offset + range.Count); }
    );
    if (element_count == 0) return;

    const uint32_t bitset_words = (element_count + 31) / 32;
    if (bitset_words > SceneBuffers::SelectionBitsetWords) return;

    // Restore baseline bitset for additive mode, or clear for non-additive.
    auto mapped = Buffers->SelectionBitsetBuffer.GetMappedData();
    if (is_additive) {
        const auto *baseline = R.try_get<const AdditiveBoxSelectBaseline>(SceneEntity);
        if (baseline && !baseline->ElementBitset.empty()) {
            const auto copy_words = std::min(bitset_words, uint32_t(baseline->ElementBitset.size()));
            memcpy(mapped.data(), baseline->ElementBitset.data(), copy_words * sizeof(uint32_t));
            if (copy_words < bitset_words) { // Zero any remaining words beyond the baseline
                memset(mapped.data() + copy_words * sizeof(uint32_t), 0, (bitset_words - copy_words) * sizeof(uint32_t));
            }
        }
    } else {
        memset(mapped.data(), 0, bitset_words * sizeof(uint32_t));
    }

    // Box-select writes element IDs directly from the selection fragment shader.
    RenderElementSelectionPass(ranges, element, true, box_min, box_max);
    // After RenderElementSelectionPass (which waits on fence), SelectionBitsetBuffer is populated.
    // Dispatch UpdateSelectionState compute shader to update element state buffers on GPU.
    ApplySelectionStateUpdate(ranges, element);
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
    const auto *instance = R.try_get<Instance>(instance_entity);
    if (!instance) return {};

    const Timer timer{"RunExcitableVertexPick"};
    const auto mesh_entity = instance->Entity;
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
            .SeenBitsIndex = SelectionHandles->ObjectPickSeenBits,
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

    const auto *bits = reinterpret_cast<const uint32_t *>(Buffers->ObjectPickSeenBitsetBuffer.GetData().data());
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
    auto mapped = Buffers->SelectionBitsetBuffer.GetMappedData();
    memset(mapped.data(), 0, bitset_words * sizeof(uint32_t));

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

    const auto *bits = reinterpret_cast<const uint32_t *>(Buffers->SelectionBitsetBuffer.GetData().data());
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
        // Recreate the ImGui texture wrapper for the line-AA composited image.
        ViewportTexture = std::make_unique<mvk::ImGuiTexture>(Vk.Device, *Pipelines->Main.Resources->FinalColorImage.View, vec2{0, 1}, vec2{1, 0});
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
    auto &logical_extent = R.get<ViewportExtent>(SceneEntity).Value;
    const auto content_region = ImGui::GetContentRegionAvail();
    const vk::Extent2D new_logical_extent{
        uint32_t(std::max(content_region.x, 0.0f)),
        uint32_t(std::max(content_region.y, 0.0f))
    };
    const bool extent_changed = logical_extent.width != new_logical_extent.width || logical_extent.height != new_logical_extent.height;
    if (extent_changed) {
        logical_extent = new_logical_extent;
        R.patch<ViewportExtent>(SceneEntity, [](auto &) {});
    }
    const auto render_extent = ComputeRenderExtentPx(logical_extent);
    const auto current_render_extent = Pipelines->Main.Resources ? ToExtent2D(Pipelines->Main.Resources->ColorImage.Extent) : vk::Extent2D{};
    const bool render_extent_changed = current_render_extent.width != render_extent.width || current_render_extent.height != render_extent.height;
    if (render_extent_changed && !extent_changed) {
        // Trigger SceneView update (projection, screen pixel scale) when DPI scale changes at fixed logical viewport size.
        R.patch<ViewportExtent>(SceneEntity, [](auto &) {});
    }

    const auto render_request = ProcessComponentEvents();

    if (auto descriptor_updates = Buffers->Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
        const Timer timer{"SubmitViewport->UpdateBufferDescriptorSets"};
        Vk.Device.updateDescriptorSets(std::move(descriptor_updates), {});
        Buffers->Ctx.ClearDeferredDescriptorUpdates();
    }
    if (!extent_changed && !render_extent_changed && render_request == RenderRequest::None) return false;

    const Timer timer{"SubmitViewport"};
    if (extent_changed || render_extent_changed) {
        if (viewportConsumerFence) { // Wait for viewport consumer to finish sampling old resources
            std::ignore = Vk.Device.waitForFences(viewportConsumerFence, VK_TRUE, UINT64_MAX);
        }
        Pipelines->SetExtent(render_extent);
        Buffers->ResizeSelectionNodeBuffer(render_extent);
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
            const auto &main = Pipelines->Main;
            const vk::DescriptorImageInfo silhouette_sampler{*sil.Resources->ImageSampler, *sil.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo object_id_sampler{*sil_edge.Resources->ImageSampler, *sil_edge.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo depth_sampler{*sil_edge.Resources->DepthSampler, *sil_edge.Resources->DepthImage.View, vk::ImageLayout::eDepthStencilReadOnlyOptimal};
            const vk::DescriptorImageInfo color_sampler{*main.Resources->NearestSampler, *main.Resources->ColorImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo line_data_sampler{*main.Resources->NearestSampler, *main.Resources->LineDataImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const auto selection_bitset = Buffers->GetSelectionBitsetDescriptor();
            const auto object_pick_seen_bitset = Buffers->GetObjectPickSeenBitsetDescriptor();
            Vk.Device.updateDescriptorSets(
                {
                    Slots->MakeImageWrite(SelectionHandles->HeadImage, head_image_info),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->SelectionCounter}, selection_counter),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->ObjectPickKey}, object_pick_key),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->ElementPickCandidates}, element_pick_candidates),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->ObjectPickSeenBits}, object_pick_seen_bitset),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->SelectionBitset}, selection_bitset),
                    Slots->MakeSamplerWrite(SelectionHandles->ObjectIdSampler, object_id_sampler),
                    Slots->MakeSamplerWrite(SelectionHandles->DepthSampler, depth_sampler),
                    Slots->MakeSamplerWrite(SelectionHandles->SilhouetteSampler, silhouette_sampler),
                    Slots->MakeSamplerWrite(SelectionHandles->ColorSampler, color_sampler),
                    Slots->MakeSamplerWrite(SelectionHandles->LineDataSampler, line_data_sampler),
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

    if (render_request == RenderRequest::ReRecord || extent_changed || render_extent_changed) RecordRenderCommandBuffer();

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
    return extent_changed || render_extent_changed;
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
