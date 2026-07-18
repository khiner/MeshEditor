#include "ProcessEvents.h"

#include "Camera.h"
#include "Changes.h"
#include "File.h"
#include "Reactive.h"
#include "TransformMath.h"
#include "Variant.h"
#include "action/Selection.h"
#include "animation/AnimationData.h"
#include "animation/AnimationTimeline.h"
#include "animation/MorphWeightState.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/SoundVertices.h"
#include "gizmo/GizmoInteraction.h"
#include "gltf/GltfScene.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "mesh/TetMeshData.h"
#include "object/ExtrasComponents.h"
#include "object/ExtrasMesh.h"
#include "object/ObjectOps.h"
#include "object/PendingSync.h"
#include "physics/PhysicsChanges.h"
#include "physics/PhysicsDebugDraw.h"
#include "physics/PhysicsSystem.h"
#include "physics/PhysicsTypes.h"
#include "render/DrawState.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/MaterialComponents.h"
#include "render/OneShotGpu.h"
#include "render/PickConstants.h"
#include "render/Pipelines.h"
#include "render/Profile.h"
#include "render/Textures.h"
#include "scene/Defaults.h"
#include "scene/EntityDestroyTracker.h"
#include "scene/RotationUi.h"
#include "scene/SceneGraph.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionBitset.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionQueries.h"
#include "viewport/FrameState.h"
#include "viewport/GizmoDrag.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderExtent.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/ViewportConsumerFence.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportOps.h"

#include <glm/gtx/euler_angles.hpp>

#include <iostream>

using std::ranges::to;
using std::views::iota;

namespace {
using namespace he;

// Sort collected {BufferIndex, value} writes by index and flush them through the mutable span.
// Returns whether anything was written.
bool FlushIndexedWrites(auto &writes, auto &&span_getter) {
    if (writes.empty()) return false;
    std::sort(writes.begin(), writes.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    auto span = span_getter();
    for (const auto &[index, value] : writes) span[index] = value;
    return true;
}

const std::vector<Element> NormalElements{Element::Vertex, Element::Face};

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

uint8_t InstanceStateBits(const entt::registry &r, entt::entity e) {
    return (r.all_of<Selected>(e) ? ElementStateSelected : 0) | (r.all_of<Active>(e) ? ElementStateActive : 0);
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
    const auto &mesh = GetMesh(r, r.get<Instance>(instance_entity).Entity);
    const auto &wt = r.get<WorldTransform>(instance_entity);
    return {wt.P + glm::rotate(wt.R, wt.S * ComputeElementLocalPosition(mesh, element, handle))};
}

void RederiveCollider(entt::registry &r, entt::entity e) {
    const auto *cs = r.try_get<const ColliderShape>(e);
    const auto *policy = r.try_get<const ColliderPolicy>(e);
    if (!cs || !policy) return;
    const auto mesh_entity = cs->MeshEntity != null_entity ? cs->MeshEntity : FindMeshEntity(r, e);
    const auto mesh = TryGetMesh(r, mesh_entity);
    if (!mesh) return;

    const auto verts = mesh->GetVerticesSpan();
    const bool has_verts = !verts.empty();
    const auto aabb = mesh->CalcAABB();
    const vec3 aabb_center = has_verts ? (aabb.Min + aabb.Max) * 0.5f : vec3{0};
    const vec3 aabb_extents = has_verts ? (aabb.Max - aabb.Min) : vec3{0};

    PhysicsShape shape = cs->Shape;
    // AutoFitDims off -> user owns kind, dims, and offset. Preserve them.
    vec3 local_offset = policy->AutoFitDims ? vec3{0} : cs->LocalOffset;

    if (policy->AutoFitDims && !policy->LockedKind) {
        if (const auto *prim = r.try_get<const PrimitiveShape>(mesh_entity)) {
            shape = std::visit(
                overloaded{
                    [](const primitive::Cuboid &s) -> PhysicsShape { return physics::Box{s.HalfExtents * 2.f}; },
                    [](const primitive::Plane &s) -> PhysicsShape { return physics::Plane{s.HalfExtents.x * 2.f, s.HalfExtents.y * 2.f}; },
                    [](const primitive::IcoSphere &s) -> PhysicsShape { return physics::Sphere{s.Radius}; },
                    [](const primitive::UVSphere &s) -> PhysicsShape { return physics::Sphere{s.Radius}; },
                    [](const primitive::Cylinder &s) -> PhysicsShape { return physics::Cylinder{s.Height, s.Radius, s.Radius}; },
                    [](const primitive::Cone &s) -> PhysicsShape { return physics::Cylinder{s.Height, 0.f, s.Radius}; },
                    [](const auto &) -> PhysicsShape { return physics::ConvexHull{}; },
                },
                *prim
            );
        } else {
            // No primitive → general-purpose ConvexHull (covers de-primitivized + imported meshes).
            shape = physics::ConvexHull{};
        }
    }

    if (policy->AutoFitDims && has_verts) {
        // Ritter's algorithm (Real-Time Collision Detection §4.3.5): near-minimum enclosing sphere
        // via two farthest-point passes plus one expand pass.
        auto ritter = [&]() -> std::pair<vec3, float> {
            const auto farthest_from = [&](vec3 from) {
                vec3 best = from;
                float best_d2 = 0;
                for (const auto &v : verts) {
                    const float d2 = glm::length2(v.Position - from);
                    if (d2 > best_d2) {
                        best_d2 = d2;
                        best = v.Position;
                    }
                }
                return best;
            };
            const vec3 q = farthest_from(verts[0].Position);
            const vec3 ru = farthest_from(q);
            vec3 c = (q + ru) * 0.5f;
            float radius = glm::length(ru - c);
            for (const auto &v : verts) {
                const float d = glm::length(v.Position - c);
                if (d > radius) {
                    const float new_r = (radius + d) * 0.5f;
                    c = c + ((d - radius) / (2.f * d)) * (v.Position - c);
                    radius = new_r;
                }
            }
            return {c, radius};
        };
        // Tightest radius for an axis-aligned (Y-axis) cylinder/capsule centered at aabb_center.
        const auto xz_radius = [&] {
            const vec2 c{aabb_center.x, aabb_center.z};
            float r = 0;
            for (const auto &v : verts) r = glm::max(r, glm::length(vec2{v.Position.x, v.Position.z} - c));
            return r;
        };

        std::visit(
            overloaded{
                [&](physics::Box &s) {
                    s.Size = aabb_extents;
                    local_offset = aabb_center;
                },
                [&](physics::Sphere &s) {
                    const auto [c, radius] = ritter();
                    s.Radius = radius;
                    local_offset = c;
                },
                [&](physics::Cylinder &s) {
                    const float radius = xz_radius();
                    s.RadiusTop = s.RadiusBottom = radius;
                    s.Height = glm::max(physics::MinShapeHeight, aabb_extents.y);
                    local_offset = aabb_center;
                },
                [&](physics::Capsule &s) {
                    const float radius = xz_radius();
                    s.RadiusTop = s.RadiusBottom = radius;
                    // 2r ≥ aabb.y degenerates toward a sphere; clamp height to keep it spec-valid.
                    s.Height = glm::max(physics::MinShapeHeight, aabb_extents.y - 2.f * radius);
                    local_offset = aabb_center;
                },
                [](auto &) {}, // Plane, ConvexHull, TriangleMesh: no fittable dims, offset stays 0
            },
            shape
        );
    }

    r.patch<ColliderShape>(e, [&](ColliderShape &x) {
        x.Shape = std::move(shape);
        x.MeshEntity = IsMeshBackedShape(x.Shape) ? mesh_entity : null_entity;
        x.LocalOffset = local_offset;
    });
}

void SetEditMode(entt::registry &r, entt::entity viewport, Element mode) {
    const auto current_mode = r.get<const EditMode>(viewport).Value;
    if (current_mode == mode) return;

    auto &meshes = r.ctx().get<MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto *bits = buffers.SelectionBitset.Data();

    struct PendingConvert {
        entt::entity MeshEntity;
        uint32_t NewCount;
        std::vector<uint32_t> FromHandles;
    };
    std::vector<PendingConvert> pending;
    std::vector<ElementRange> old_ranges;
    uint32_t old_max_end = 0;
    for (auto [mesh_entity, br, _] : r.view<MeshSelectionBitsetRange, const MeshHandle>().each()) {
        const auto mesh = GetMesh(r, mesh_entity);
        const uint32_t old_count = br.Count, new_count = selection::GetElementCount(mesh, mode);
        auto from_handles = selection::ScanBitsetRange(bits, br.Offset, old_count);
        if (old_count > 0) old_ranges.emplace_back(mesh_entity, br.Offset, old_count);
        old_max_end = std::max(old_max_end, br.Offset + old_count);
        r.remove<MeshActiveElement>(mesh_entity);
        pending.emplace_back(PendingConvert{mesh_entity, new_count, std::move(from_handles)});
    }

    // Clear the superset of old and new packed bit ranges once, to avoid stale/overlap bits.
    uint32_t new_max_end = 0;
    for (const auto &p : pending) new_max_end += (p.NewCount + 31) / 32 * 32;
    const uint32_t clear_words = (std::max(old_max_end, new_max_end) + 31) / 32;
    if (clear_words > 0) memset(bits, 0, clear_words * sizeof(uint32_t));

    if (!old_ranges.empty()) {
        DispatchUpdateSelectionStates(r, old_ranges, current_mode);
        // Face mode also derives edge states via CPU; clear them when exiting face-select.
        if (current_mode == Element::Face) {
            for (const auto &p : pending) meshes.UpdateEdgeStatesFromFaces(GetMesh(r, p.MeshEntity), {}, {});
        }
    }

    std::vector<ElementRange> new_ranges;
    uint32_t next_offset = 0;
    for (const auto &p : pending) {
        auto &br = r.get<MeshSelectionBitsetRange>(p.MeshEntity);
        br.Offset = next_offset;
        br.Count = p.NewCount;
        const auto &mesh = GetMesh(r, p.MeshEntity);
        for (const uint32_t h : selection::ConvertSelectionElement(p.FromHandles, mesh, current_mode, mode)) {
            if (h < p.NewCount) {
                const uint32_t gbit = next_offset + h;
                bits[gbit >> 5] |= 1u << (gbit & 31u);
            }
        }
        if (p.NewCount > 0) new_ranges.emplace_back(p.MeshEntity, next_offset, p.NewCount);
        next_offset = (next_offset + p.NewCount + 31) / 32 * 32;
    }

    r.patch<EditMode>(viewport, [mode](auto &edit_mode) { edit_mode.Value = mode; });
    if (!new_ranges.empty()) ApplySelectionStateUpdate(r, viewport, new_ranges, mode);
    else if (!old_ranges.empty()) r.emplace_or_replace<ElementStatesDirty>(viewport);
}

struct SyncResult {
    std::vector<entt::entity> NewlyInserted; // Entities inserted into GPU buffers — callers must write their WorldTransform before submit.
    std::vector<entt::entity> NewMeshEntities; // Mesh entities needing deferred index buffer creation.
    std::vector<entt::entity> NewExtrasEntities; // Non-mesh buffer entities (extras/bone/joint) needing deferred index creation.
    bool Compacted{false}; // Instances were compact-erased (hides) — caller must request Submit.
};

SyncResult SyncModelsBuffers(entt::registry &r) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    // Consume the new-buffer-entity tracker. Categorize by type for deferred index buffer creation.
    std::vector<entt::entity> new_mesh_entities, new_extras_entities;
    for (auto e : reactive<changes::NewBufferEntity>(r)) {
        if (!r.valid(e) || !r.all_of<MeshBuffers>(e)) continue;
        if (HasMesh(r, e)) new_mesh_entities.emplace_back(e);
        else if (r.all_of<ObjectExtrasTag>(e) || r.all_of<ArmatureObject>(e) || r.all_of<BoneJoint>(e)) new_extras_entities.emplace_back(e);
    }

    // Hides — compact-erase removed instances within their entity's range in the shared InstanceArena.
    bool compacted = false;
    for (auto [buffer_entity, pending] : r.view<PendingHide>().each()) {
        // Sort descending so erasing from back-to-front avoids index shifting within the batch.
        auto &indices = pending.BufferIndices;
        std::sort(indices.begin(), indices.end(), std::greater<>());
        auto &mb = r.get<ModelsBuffer>(buffer_entity);
        for (const auto global_idx : indices) {
            buffers.Instances.CompactErase(global_idx, mb.InstanceRange.Offset + mb.InstanceCount);
            --mb.InstanceCount;
        }
        compacted = true;
        // Fixup BufferIndex on remaining RenderInstances for this buffer entity.
        for (auto [_, ri] : r.view<RenderInstance>().each()) {
            if (ri.Entity != buffer_entity || ri.BufferIndex == UINT32_MAX) continue;
            // Count how many erased indices were below this instance's index.
            uint32_t shift = 0;
            for (const auto erased_idx : indices) {
                if (erased_idx < ri.BufferIndex) ++shift;
            }
            if (shift > 0) ri.BufferIndex -= shift;
        }
        r.remove<PendingHide>(buffer_entity);
    }

    // Shows — batch-insert new instances into GPU buffers. Creates ModelsBuffer on first use.
    // WorldTransform is NOT written here — only ObjectIds, InstanceStates, and Bounds.
    // Callers must ensure WorldTransform is written before submit (see newly_inserted return value).
    std::vector<entt::entity> newly_inserted;
    std::unordered_map<entt::entity, std::vector<entt::entity>> shows_by_buffer;
    for (auto entity : reactive<changes::RenderInstanceCreated>(r)) {
        if (!r.valid(entity) || !r.all_of<RenderInstance>(entity)) continue;

        const auto &ri = r.get<const RenderInstance>(entity);
        if (ri.BufferIndex == UINT32_MAX) shows_by_buffer[ri.Entity].emplace_back(entity);
    }
    // Pre-reserve InstanceArena for all new instances to avoid per-Allocate growth checks.
    {
        uint32_t total_new_instances = 0;
        for (const auto &[_, entities] : shows_by_buffer) total_new_instances += entities.size();
        if (total_new_instances > 0) buffers.Instances.ReserveAdditional(total_new_instances);
    }
    // Shared write buffers — reused across groups to avoid per-group heap allocations.
    std::vector<uint32_t> object_ids;
    std::vector<uint8_t> states;
    for (auto &[buffer_entity, entities] : shows_by_buffer) {
        const uint32_t n = entities.size();
        // Create ModelsBuffer on first show (deferred from MeshBuffers creation so we know the initial capacity).
        if (!r.all_of<ModelsBuffer>(buffer_entity)) {
            r.emplace<ModelsBuffer>(buffer_entity, ModelsBuffer{buffers.Instances.Allocate(n), 0});
        }
        auto &mb = r.get<ModelsBuffer>(buffer_entity);
        // Grow the range if needed (e.g., adding instances to existing entity).
        const auto new_total = mb.InstanceCount + n;
        if (new_total > mb.InstanceRange.Count) {
            auto old_range = mb.InstanceRange;
            const auto new_capacity = std::max(mb.InstanceRange.Count * 2, new_total);
            mb.InstanceRange = buffers.Instances.Allocate(new_capacity);
            buffers.Instances.CopyInstances(old_range.Offset, mb.InstanceRange.Offset, mb.InstanceCount);
            // Fixup existing BufferIndex values for this entity.
            for (auto [other_entity, ri] : r.view<RenderInstance>().each()) {
                if (ri.Entity == buffer_entity && ri.BufferIndex != UINT32_MAX) {
                    ri.BufferIndex = mb.InstanceRange.Offset + (ri.BufferIndex - old_range.Offset);
                }
            }
            buffers.Instances.Free(old_range);
        }
        // Build contiguous arrays for ObjectIds and InstanceStates (reusing shared vectors).
        object_ids.resize(n);
        states.resize(n);
        const auto base_index = mb.InstanceRange.Offset + mb.InstanceCount;
        for (uint32_t j = 0; j < n; ++j) {
            const auto instance_entity = entities[j];
            r.get<RenderInstance>(instance_entity).BufferIndex = base_index + j;
            object_ids[j] = r.get<const RenderInstance>(instance_entity).ObjectId;
            states[j] = InstanceStateBits(r, instance_entity);
        }
        // Write ObjectIds, InstanceStates, and Bounds at the new slots. WorldTransform slots are left unwritten —
        // the WorldTransform reactive pass writes them before submit.
        buffers.Instances.ObjectIdBuffer.Update(as_bytes(object_ids), vk::DeviceSize(base_index) * sizeof(uint32_t));
        buffers.Instances.StateBuffer.Update(as_bytes(states), vk::DeviceSize(base_index) * sizeof(uint8_t));
        // Undeformed instances share the mesh's local AABB. The deformed-bounds pass overwrites
        // deformed objects' slots later this call.
        const auto mesh = TryGetMesh(r, buffer_entity);
        std::ranges::fill(buffers.Instances.GetMutableBounds({base_index, n}), mesh ? mesh->CalcAABB() : AABB{});
        mb.InstanceCount = new_total;
        newly_inserted.append_range(entities);
    }
    return {std::move(newly_inserted), std::move(new_mesh_entities), std::move(new_extras_entities), compacted};
}

void EnsureWireframes(entt::registry &r, entt::entity viewport) {
    auto &meshes = r.ctx().get<MeshStore>();
    auto &shape_buffers = r.ctx().get<ColliderShapeBuffers>();
    const auto &settings = r.get<const ViewportDisplay>(viewport);
    const bool show_aabb = settings.ShowBoundingBoxes;
    const bool show_tets = settings.ShowTetWireframe;
    if (r.view<const ColliderShape>().empty() && r.view<ColliderWireframe>().empty() &&
        r.view<AABBWireframe>().empty() && r.view<TetWireframe>().empty() &&
        !show_aabb && !show_tets) return;

    using enum ColliderShapeBuffer;
    auto buf = [&](ColliderShapeBuffer kind) -> entt::entity & { return shape_buffers.Entities[uint8_t(kind)]; };

    // Lazily create canonical buffer entities (recreated if destroyed by last-instance cleanup).
    auto ensure_buffer = [&](ColliderShapeBuffer kind, auto generator) {
        if (r.valid(buf(kind))) return;
        auto mesh = generator();
        if (!mesh.Positions.empty()) {
            buf(kind) = ::CreateExtrasBufferEntity(r, meshes, mesh.Positions, mesh.EdgeIndices);
            r.emplace<OverlayExtra>(buf(kind));
        }
    };
    ensure_buffer(Box, physics_debug::UnitBox);
    ensure_buffer(Sphere, physics_debug::UnitSphere);
    ensure_buffer(CapsuleCap, physics_debug::UnitCapsuleCap);
    ensure_buffer(Circle, physics_debug::UnitCircle);
    ensure_buffer(Line, physics_debug::UnitLine);

    // Buffer entity that Instances[0] should reference for a given shape kind. null = no wireframe.
    // Cylinder uses a Circle for its top ring; Capsule uses a CapsuleCap for its top hemisphere.
    auto primary_buffer = [&](const PhysicsShape &shape) -> entt::entity {
        return std::visit(
            overloaded{
                [&](const physics::Box &) { return buf(Box); },
                [&](const physics::Sphere &) { return buf(Sphere); },
                [&](const physics::Cylinder &) { return buf(Circle); },
                [&](const physics::Capsule &) { return buf(CapsuleCap); },
                [](const auto &) { return entt::entity{entt::null}; },
            },
            shape
        );
    };

    // Drop wireframe: destroy instances and remove the component (batched so views stay valid).
    auto drop_wireframes = [&](std::span<const entt::entity> entities) {
        for (auto e : entities) {
            const auto &cw = r.get<const ColliderWireframe>(e);
            for (uint8_t i = 0; i < cw.Count; ++i) {
                if (r.valid(cw.Instances[i])) Destroy(r, viewport, cw.Instances[i]);
            }
            r.remove<ColliderWireframe>(e);
        }
    };

    auto make_instance = [&](entt::entity buffer_entity, entt::entity parent) -> entt::entity {
        if (buffer_entity == entt::null) return entt::null;
        const auto inst = r.create();
        r.emplace<Instance>(inst, buffer_entity);
        r.emplace<Transform>(inst);
        r.emplace<WorldTransform>(inst);
        r.emplace<SubElementOf>(inst, parent);
        r.emplace<OverlayExtra>(inst);
        Show(r, inst);
        return inst;
    };

    // Drop wireframes whose backing buffer no longer matches the shape kind (e.g. Box → ConvexHull).
    // The creation loop below recreates them (or leaves none for non-wireframe kinds).
    std::vector<entt::entity> stale;
    for (auto [entity, cs, cw] : r.view<const ColliderShape, const ColliderWireframe>().each()) {
        if (cw.Count > 0 && r.valid(cw.Instances[0]) && r.get<const Instance>(cw.Instances[0]).Entity != primary_buffer(cs.Shape)) {
            stale.emplace_back(entity);
        }
    }
    drop_wireframes(stale);

    for (auto [entity, cs] : r.view<const ColliderShape>().each()) {
        if (r.all_of<ColliderWireframe>(entity)) continue;

        const auto &shape = cs.Shape;
        ColliderWireframe cw{};

        if (std::holds_alternative<physics::Cylinder>(shape) || std::holds_alternative<physics::Capsule>(shape)) {
            const auto cap_buf = buf(std::holds_alternative<physics::Capsule>(shape) ? CapsuleCap : Circle);
            cw.Instances[0] = make_instance(cap_buf, entity); // top
            cw.Instances[1] = make_instance(cap_buf, entity); // bottom
            for (uint8_t i = 0; i < 4; ++i) cw.Instances[2 + i] = make_instance(buf(Line), entity);
            cw.Count = 6;
        } else {
            const auto shape_buf = std::visit(
                overloaded{
                    [&](const physics::Box &) { return buf(Box); },
                    [&](const physics::Sphere &) { return buf(Sphere); },
                    [](const auto &) { return entt::entity{entt::null}; },
                },
                shape
            );
            if (shape_buf != entt::null) {
                cw.Instances[0] = make_instance(shape_buf, entity);
                cw.Count = 1;
            }
        }
        if (cw.Count > 0) r.emplace<ColliderWireframe>(entity, cw);
    }

    // Remove wireframe instances for colliders that no longer exist.
    std::vector<entt::entity> orphans;
    for (auto [entity, cw] : r.view<ColliderWireframe>().each()) {
        if (!r.all_of<ColliderShape>(entity)) orphans.emplace_back(entity);
    }
    drop_wireframes(orphans);

    std::vector<entt::entity> aabb_stale;
    for (auto [entity, bw] : r.view<AABBWireframe>().each()) {
        if (!show_aabb || !r.all_of<Selected>(entity)) aabb_stale.emplace_back(entity);
    }
    for (auto e : aabb_stale) {
        if (auto &bw = r.get<AABBWireframe>(e); r.valid(bw.Instance)) Destroy(r, viewport, bw.Instance);
        r.remove<AABBWireframe>(e);
    }

    if (show_aabb) {
        ensure_buffer(Box, physics_debug::UnitBox); // Removing a deselected aabb above may have freed the shared Box buffer.
        for (auto entity : r.view<Selected>()) {
            if (!r.all_of<AABBWireframe>(entity)) {
                const auto *instance = r.try_get<const Instance>(entity);
                if (instance && HasMesh(r, instance->Entity)) r.emplace<AABBWireframe>(entity, make_instance(buf(Box), entity));
            }
        }
    }

    // Drop tet wireframes whose backing geometry no longer matches (toggle off, deselected,
    // TetMeshData removed, or point count differs after regeneration).
    std::vector<entt::entity> tet_stale;
    for (auto [entity, tw] : r.view<TetWireframe>().each()) {
        const auto *inst = r.try_get<const Instance>(entity);
        const auto *tm = inst ? r.try_get<const TetMeshData>(inst->Entity) : nullptr;
        if (!show_tets || !r.all_of<Selected>(entity) || !tm || tm->Positions.empty()) {
            tet_stale.emplace_back(entity);
            continue;
        }
        const auto *wi = r.try_get<const Instance>(tw.Instance);
        const auto *mb = wi ? r.try_get<const MeshBuffers>(wi->Entity) : nullptr;
        if (!mb || mb->Vertices.Count != tm->Positions.size()) tet_stale.emplace_back(entity);
    }
    for (auto e : tet_stale) {
        if (auto &tw = r.get<TetWireframe>(e); r.valid(tw.Instance)) Destroy(r, viewport, tw.Instance);
        r.remove<TetWireframe>(e);
    }

    if (show_tets) {
        for (auto entity : r.view<Selected>()) {
            if (r.all_of<TetWireframe>(entity)) continue;
            const auto *instance = r.try_get<const Instance>(entity);
            if (!instance) continue;

            const auto *tm = r.try_get<const TetMeshData>(instance->Entity);
            if (!tm || tm->Positions.empty()) continue;

            const auto tet_buf = ::CreateExtrasBufferEntity(r, meshes, tm->Positions, tm->EdgeIndices);
            r.emplace<OverlayExtra>(tet_buf);
            r.emplace<TetWireframe>(entity, make_instance(tet_buf, entity));
        }
    }
}

// Build a camera/light/empty gizmo's wireframe from the object's params, replacing any existing one.
void RebuildGizmoGeometry(entt::registry &r, MeshStore &meshes, GpuBuffers &buffers, entt::entity object, entt::entity buffer, ObjectType type) {
    MeshData data;
    std::vector<uint8_t> vertex_classes;
    if (type == ObjectType::Light) {
        auto wf = BuildLightMesh(r.get<const PunctualLight>(object));
        data = std::move(wf.Data);
        vertex_classes = std::move(wf.VertexClasses);
    } else if (type == ObjectType::Camera) {
        data = BuildCameraFrustumMesh(r.get<const Camera>(object), r.all_of<LookingThrough>(object));
    } else {
        data = BuildEmptyMesh();
    }

    // Release the existing wireframe before rebuilding.
    if (r.all_of<OverlayVertexStoreId>(buffer)) {
        if (const auto *vcr = r.try_get<const VertexClass>(buffer)) {
            buffers.VertexClassBuffer.Release({vcr->Offset, r.get<const MeshBuffers>(buffer).Vertices.Count});
            r.remove<VertexClass>(buffer);
        }
        meshes.ReleaseOverlay(r.get<const OverlayVertexStoreId>(buffer).StoreId);
        if (auto *mb = r.try_get<MeshBuffers>(buffer)) buffers.Release(*mb);
        r.erase<MeshBuffers>(buffer);
        r.remove<OverlayVertexStoreId>(buffer);
    }

    // Emplacing the handle builds MeshBuffers (vertices) via on-construct; add the edges after.
    r.emplace<OverlayVertexStoreId>(buffer, meshes.AllocateOverlayVertexBuffer(data.Positions).first);
    if (const auto edges = data.CreateEdgeIndices(); !edges.empty()) {
        r.patch<MeshBuffers>(buffer, [&](auto &mb) { mb.EdgeIndices = buffers.CreateIndices(edges, IndexKind::Edge); });
    }
    if (!vertex_classes.empty()) {
        r.emplace<VertexClass>(buffer, buffers.VertexClassBuffer.Allocate(std::span<const uint8_t>(vertex_classes)).Offset);
    }
}

// Mesh-local bounds after morph and skin deformation, matching the vertex shader. Nullopt if undeformed.
std::optional<AABB> ComputeDeformedLocalAABB(const entt::registry &r, MeshStore &meshes, GpuBuffers &buffers, entt::entity object, const Mesh &mesh) {
    const uint32_t store_id = mesh.GetStoreId();

    std::span<const BoneDeformVertex> bone;
    std::span<const mat4> deform;
    if (const auto *mod = r.try_get<const ArmatureModifier>(object)) {
        if (const auto *pose = r.try_get<const ArmaturePoseState>(mod->ArmatureEntity); pose && pose->GpuDeformRange.Count > 0) {
            bone = meshes.GetBoneDeform(store_id);
            deform = buffers.ArmatureDeformBuffer.Get(pose->GpuDeformRange);
        }
    }
    std::span<const MorphTargetVertex> morph;
    std::span<const float> weights;
    if (const auto *ms = r.try_get<const MorphWeightState>(object); ms && !ms->Weights.empty()) {
        morph = meshes.GetMorphTargets(store_id);
        weights = ms->Weights;
    }
    if (bone.empty() && weights.empty()) return std::nullopt;

    const auto verts = mesh.GetVerticesSpan();
    const auto vtx_count = uint32_t(verts.size());
    AABB aabb;
    for (uint32_t i = 0; i < vtx_count; ++i) {
        vec3 p = verts[i].Position;
        for (uint32_t t = 0; t < weights.size(); ++t) p += weights[t] * morph[t * vtx_count + i].PositionDelta;
        if (!bone.empty()) {
            const auto &bd = bone[i];
            const mat4 m = bd.Weights.x * deform[bd.Joints.x] + bd.Weights.y * deform[bd.Joints.y] +
                bd.Weights.z * deform[bd.Joints.z] + bd.Weights.w * deform[bd.Joints.w];
            p = vec3{m * vec4{p, 1.f}};
        }
        aabb.Min = glm::min(aabb.Min, p);
        aabb.Max = glm::max(aabb.Max, p);
    }
    return aabb;
}

void UpdateWireframeTransforms(entt::registry &r, const std::unordered_set<entt::entity> &deformed_bounds_changed) {
    if (r.view<ColliderWireframe>().empty() && r.view<AABBWireframe>().empty() && r.view<TetWireframe>().empty()) return;

    const auto &wt_changed = reactive<changes::WorldTransform>(r);
    const auto &shape_changed = reactive<changes::PhysicsShape>(r);
    const auto &mesh_geom_changed = reactive<changes::MeshGeometry>(r);
    for (auto [entity, cs, cw] : r.view<const ColliderShape, const ColliderWireframe>().each()) {
        const auto *wt = r.try_get<const WorldTransform>(entity);
        if (!wt) continue;

        const bool parent_moved = wt_changed.contains(entity);
        const bool shape_resized = shape_changed.contains(entity);
        const bool newly_created = [&] {
            for (uint8_t i = 0; i < cw.Count; ++i) {
                if (r.valid(cw.Instances[i]) && wt_changed.contains(cw.Instances[i])) return true;
            }
            return false;
        }();
        if (!parent_moved && !shape_resized && !newly_created) continue;

        auto set_wt = [&](entt::entity inst, mat4 m) {
            if (!r.valid(inst)) return;
            r.replace<WorldTransform>(inst, ToTransform(m));
        };

        // Maps the unit Y-line (from (0,+0.5,0) to (0,-0.5,0)) onto the segment p1→p2.
        // Line is rotationally symmetric about its axis - any perpendicular X/Z basis works.
        auto line_xform = [](vec3 p1, vec3 p2) -> mat4 {
            const auto mid = (p1 + p2) * 0.5f;
            const auto y_axis = p1 - p2;
            const auto len = glm::length(y_axis);
            // Coincident endpoints: collapse to a point at mid to avoid divide-by-zero.
            if (len < 1e-6f) return glm::scale(glm::translate(mat4{1}, mid), vec3{0});

            const auto y_dir = y_axis / len;
            const auto x_dir = glm::normalize(glm::cross(std::abs(y_dir.y) > 0.9f ? vec3{1, 0, 0} : vec3{0, 1, 0}, y_dir));
            return {{x_dir, 0}, {y_dir * len, 0}, {glm::cross(x_dir, y_dir), 0}, {mid, 1}};
        };

        const auto base = ToMatrix(*wt) * glm::translate(mat4{1}, cs.LocalOffset);
        auto set_side_lines = [&](float rt, float rb, float h) {
            for (uint8_t i = 0; i < 4; ++i) {
                const auto a = Pi * 0.5f * float(i);
                const auto c = std::cos(a), s = std::sin(a);
                set_wt(cw.Instances[2 + i], base * line_xform({rt * c, h * 0.5f, rt * s}, {rb * c, -h * 0.5f, rb * s}));
            }
        };

        std::visit(
            overloaded{
                [&](const physics::Box &s) { set_wt(cw.Instances[0], base * glm::scale(mat4{1}, s.Size)); },
                [&](const physics::Sphere &s) { set_wt(cw.Instances[0], base * glm::scale(mat4{1}, vec3{s.Radius * 2})); },
                [&](const physics::Capsule &s) {
                    const float dt = s.RadiusTop * 2.0f, db = s.RadiusBottom * 2.0f;
                    set_wt(cw.Instances[0], base * glm::translate(mat4{1}, {0, s.Height * 0.5f, 0}) * glm::scale(mat4{1}, {dt, dt, dt}));
                    set_wt(cw.Instances[1], base * glm::translate(mat4{1}, {0, -s.Height * 0.5f, 0}) * glm::scale(mat4{1}, {db, -db, db}));
                    set_side_lines(s.RadiusTop, s.RadiusBottom, s.Height);
                },
                [&](const physics::Cylinder &s) {
                    const float dt = s.RadiusTop * 2.0f, db = s.RadiusBottom * 2.0f;
                    set_wt(cw.Instances[0], base * glm::translate(mat4{1}, {0, s.Height * 0.5f, 0}) * glm::scale(mat4{1}, {dt, 1, dt}));
                    set_wt(cw.Instances[1], base * glm::translate(mat4{1}, {0, -s.Height * 0.5f, 0}) * glm::scale(mat4{1}, {db, 1, db}));
                    set_side_lines(s.RadiusTop, s.RadiusBottom, s.Height);
                },
                [](const auto &) {},
            },
            cs.Shape
        );
    }

    // Reposition the AABB wireframe when the mesh geometry or bounds changed, the parent moved, or the wireframe instance was just created this frame.
    const auto &buffers = r.ctx().get<const GpuBuffers>();
    for (auto [entity, bw] : r.view<const AABBWireframe>().each()) {
        if (!r.valid(bw.Instance)) continue;

        const auto *wt = r.try_get<const WorldTransform>(entity);
        const auto *instance = r.try_get<const Instance>(entity);
        if (!wt || !instance) continue;

        const auto *ri = r.try_get<const RenderInstance>(entity);
        if (!ri || ri->BufferIndex == UINT32_MAX) continue;

        const bool parent_moved = wt_changed.contains(entity);
        const bool mesh_changed = mesh_geom_changed.contains(instance->Entity);
        const bool newly_created = wt_changed.contains(bw.Instance);
        const bool bounds_changed = deformed_bounds_changed.contains(entity);
        if (!parent_moved && !mesh_changed && !newly_created && !bounds_changed) continue;

        const auto &aabb = buffers.Instances.GetBounds(ri->BufferIndex);
        const auto size = aabb.Max - aabb.Min, center = (aabb.Min + aabb.Max) * 0.5f;
        r.replace<WorldTransform>(bw.Instance, ToTransform(ToMatrix(*wt) * glm::translate(mat4{1}, center) * glm::scale(mat4{1}, size)));
    }

    for (auto [entity, tw] : r.view<const TetWireframe>().each()) {
        if (r.valid(tw.Instance)) {
            const auto *wt = r.try_get<const WorldTransform>(entity);
            if (wt && (wt_changed.contains(entity) || wt_changed.contains(tw.Instance))) r.replace<WorldTransform>(tw.Instance, *wt);
        }
    }
}

// Resize the viewport's GPU render resources to match RenderExtentPx(ViewportExtent), recreating images and
// rewriting selection descriptors. Returns true when a resize occurred.
bool SyncViewportRenderResources(entt::registry &r, entt::entity viewport) {
    auto &pipelines = r.ctx().get<Pipelines>();
    const auto render_extent_px = RenderExtentPx(r);
    const vk::Extent2D render_extent{render_extent_px.x, render_extent_px.y};
    const auto built = pipelines.BuiltColorExtent();
    if (built.width == render_extent.width && built.height == render_extent.height) return false;

    const auto &vk = r.ctx().get<const VulkanResources>();
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    // Wait for the live consumer (ImGui) to finish sampling the old resources before recreating them.
    if (const auto fence = r.ctx().get<const ViewportConsumerFence>().Value) {
        std::ignore = vk.Device.waitForFences(fence, VK_TRUE, UINT64_MAX);
    }
    pipelines.SetExtent(render_extent);
    {
        const auto shading = r.get<const ViewportDisplay>(viewport).ViewportShading;
        const bool is_pbr = shading == ViewportShadingMode::MaterialPreview || shading == ViewportShadingMode::Rendered;
        const bool want_transmission = is_pbr && GetActivePbrLighting(r, viewport, shading).RealTransmission && pipelines.Main.Compiler.HasFeature(PbrFeature::Transmission);
        pipelines.Main.EnsureTransmissionResources(render_extent, vk.Device, vk.PhysicalDevice, want_transmission);
    }
    buffers.ResizeSelectionNodeBuffer(render_extent);
    {
        const profile::CpuScope scope{"UpdateSelectionDescriptorSets"};
        const auto head_image_info = vk::DescriptorImageInfo{
            nullptr,
            *pipelines.SelectionFragment.Resources->HeadImage.View,
            vk::ImageLayout::eGeneral
        };
        const auto selection_counter = buffers.SelectionCounter.GetDescriptor();
        const auto object_pick_key = buffers.ObjectPickKeys.GetDescriptor(GpuBuffers::MaxSelectableObjects);
        const auto element_pick_candidates = buffers.ElementPickCandidates.GetDescriptor(GpuBuffers::ElementPickGroupCount);
        const auto &sil = pipelines.Silhouette;
        const auto &sil_edge = pipelines.SilhouetteEdge;
        const auto &main = pipelines.Main;
        const vk::DescriptorImageInfo silhouette_sampler{*sil.Resources->ImageSampler, *sil.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
        const vk::DescriptorImageInfo object_id_sampler{*sil_edge.Resources->ImageSampler, *sil_edge.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
        const vk::DescriptorImageInfo depth_sampler{*sil_edge.Resources->DepthSampler, *sil_edge.Resources->DepthImage.View, vk::ImageLayout::eDepthStencilReadOnlyOptimal};
        const auto scene_color_sampler = main.SceneColorSamplerInfo();
        const auto overlay_color_sampler = main.OverlayColorSamplerInfo();
        const vk::DescriptorImageInfo line_data_sampler{*main.Resources->NearestSampler, *main.Resources->LineDataImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
        const auto transmission_sampler = main.TransmissionSamplerInfo();
        const auto motion_blur_accum_sampler = main.MotionBlurAccumSamplerInfo();
        const auto velocity_sampler = main.VelocitySamplerInfo();
        const auto scene_depth_sampler = main.SceneDepthSamplerInfo();
        const auto selection_bitset = buffers.SelectionBitset.GetDescriptor(GpuBuffers::SelectionBitsetWords);
        const auto object_pick_seen_bitset = buffers.ObjectPickSeenBitset.GetDescriptor(GpuBuffers::ObjectPickBitsetWords);
        vk.Device.updateDescriptorSets(
            {
                slots.MakeImageWrite(sel_slots.HeadImage, head_image_info),
                slots.MakeBufferWrite({SlotType::Buffer, sel_slots.SelectionCounter}, selection_counter),
                slots.MakeBufferWrite({SlotType::Buffer, sel_slots.ObjectPickKey}, object_pick_key),
                slots.MakeBufferWrite({SlotType::Buffer, sel_slots.ElementPickCandidates}, element_pick_candidates),
                slots.MakeBufferWrite({SlotType::Buffer, sel_slots.ObjectPickSeenBits}, object_pick_seen_bitset),
                slots.MakeBufferWrite({SlotType::Buffer, sel_slots.SelectionBitset}, selection_bitset),
                slots.MakeSamplerWrite(sel_slots.ObjectIdSampler, object_id_sampler),
                slots.MakeSamplerWrite(sel_slots.DepthSampler, depth_sampler),
                slots.MakeSamplerWrite(sel_slots.SilhouetteSampler, silhouette_sampler),
                slots.MakeSamplerWrite(sel_slots.SceneColorSampler, scene_color_sampler),
                slots.MakeSamplerWrite(sel_slots.OverlayColorSampler, overlay_color_sampler),
                slots.MakeSamplerWrite(sel_slots.LineDataSampler, line_data_sampler),
                slots.MakeSamplerWrite(sel_slots.TransmissionSampler, transmission_sampler),
                slots.MakeSamplerWrite(sel_slots.MotionBlurAccumSampler, motion_blur_accum_sampler),
                slots.MakeSamplerWrite(sel_slots.VelocitySampler, velocity_sampler),
                slots.MakeSamplerWrite(sel_slots.SceneDepthSampler, scene_depth_sampler),
                slots.MakeSamplerWrite(sel_slots.DepthPyramidSampler, main.DepthPyramidSamplerInfo()),
            },
            {}
        );
    }
    buffers.Ctx.FlushDeferredDescriptorUpdates(vk.Device);
    return true;
}
} // namespace

void ProcessComponentEvents(entt::registry &r, entt::entity viewport) {
    const auto &vk = r.ctx().get<const VulkanResources>();
    const auto &one_shot = r.ctx().get<const OneShotGpu>();
    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &meshes = r.ctx().get<MeshStore>();
    auto &textures = r.ctx().get<TextureStore>();
    auto &environments = r.ctx().get<EnvironmentStore>();
    auto &pipelines = r.ctx().get<Pipelines>();
    const profile::CpuScope profile_scope{"ProcessEvents"};

    auto &pending_render = r.ctx().get<PendingRenderRequest>().Value;
    auto request = [&pending_render](RenderRequest req) { pending_render = std::max(pending_render, req); };

    BuildMissingWorldTransforms(r);

    // Resize render resources before the pick handlers below resolve against the rendered scene.
    const bool resized = SyncViewportRenderResources(r, viewport);
    if (resized) request(RenderRequest::ReRecord);

    if (r.all_of<PendingShaderRecompile>(viewport)) {
        r.remove<PendingShaderRecompile>(viewport);
        pipelines.CompileShaders();
        request(RenderRequest::ReRecord);
    }

    // Re-upload restored textures into their exact recorded slots.
    // A free slot means a restore (re-upload it), an already-live slot means import already materialized it this session (skip).
    if (!reactive<changes::MaterializedTextures>(r).empty()) {
        if (const auto *manifest = r.try_get<const MaterializedTextures>(viewport)) {
            auto *src_assets = r.try_get<gltf::SourceAssets>(viewport);
            auto &pending = r.get_or_emplace<PendingTextureUploads>(viewport);
            for (const auto &t : manifest->Items) {
                if (!slots.Reserve(SlotType::Sampler, t.SamplerSlot)) continue;
                // External-URI images drop their encoded Bytes after the original upload (SourceAbsPath is the persistence), so re-read them from disk before decoding.
                if (src_assets && t.SourceImageIndex < src_assets->Images.size()) {
                    auto &img = src_assets->Images[t.SourceImageIndex];
                    if (img.Bytes.empty() && !img.SourceAbsPath.empty() && std::filesystem::is_regular_file(img.SourceAbsPath)) {
                        const auto encoded = File::ReadAsString(img.SourceAbsPath).value_or(std::string{});
                        img.Bytes.assign(reinterpret_cast<const std::byte *>(encoded.data()), reinterpret_cast<const std::byte *>(encoded.data()) + encoded.size());
                    }
                }
                pending.Items.emplace_back(PendingTextureUpload{
                    .SamplerSlot = t.SamplerSlot,
                    .Source = PendingTextureUpload::GltfImageRef{t.SourceImageIndex},
                    .ColorSpace = t.ColorSpace,
                    .WrapS = t.WrapS,
                    .WrapT = t.WrapT,
                    .Sampler = t.Sampler,
                    .Name = t.Name,
                });
            }
        }
    }
    if (auto *pending_tex = r.try_get<PendingTextureUploads>(viewport); pending_tex && !pending_tex->Items.empty()) {
        const auto *src = r.try_get<const gltf::SourceAssets>(viewport);
        static const std::vector<gltf::Image> empty_images;
        const auto &gltf_images = src ? src->Images : empty_images;
        auto batch = BeginTextureUploadBatch(vk.Device, *one_shot.Pool, buffers.Ctx);
        for (const auto &item : pending_tex->Items) {
            auto entry = MaterializeTextureEntry(vk, batch, slots, item, gltf_images);
            if (!entry) {
                std::cerr << std::format("Warning: Failed to materialize texture '{}': {}\n", item.Name, entry.error());
                ReleaseSamplerSlots(slots, std::span{&item.SamplerSlot, 1});
                continue;
            }
            textures.Textures.emplace_back(std::move(*entry));
        }
        SubmitTextureUploadBatch(batch, vk.Queue, *one_shot.Fence, vk.Device);
        r.remove<PendingTextureUploads>(viewport);
    }
    // Rebuild an imported (EXT-IBL) scene world from restored SourceAssets, whose prefiltered cubemap ClearScene released.
    // Re-emit the import only on restore (import time already emitted it), allocating fresh IBL cube slots since they aren't baked into materials.
    if (!reactive<changes::SceneWorld>(r).empty()) {
        const auto *src = r.try_get<const gltf::SourceAssets>(viewport);
        if (src && src->ImageBasedLight && !environments.ImportedSceneWorld && !r.all_of<PendingEnvironmentImport>(viewport)) {
            const auto [diffuse_slot, specular_slot] = AllocateIblCubeSlots(slots);
            r.emplace_or_replace<PendingEnvironmentImport>(viewport, *src->ImageBasedLight, diffuse_slot, specular_slot);
            r.remove<PendingSceneWorldClear>(viewport);
        }
    }
    // PendingSceneWorldClear takes precedence: a non-EXT load arrived after a previous EXT-IBL load
    // and the import is now stale. Cancel any pending import (release its allocated slots) before reset.
    if (r.all_of<PendingSceneWorldClear>(viewport)) {
        auto &env = environments;
        if (auto *imp = r.try_get<PendingEnvironmentImport>(viewport)) {
            ReleaseCubeSamplerSlot(slots, imp->DiffuseCubeSlot);
            ReleaseCubeSamplerSlot(slots, imp->SpecularCubeSlot);
            r.remove<PendingEnvironmentImport>(viewport);
        }
        if (env.ImportedSceneWorld) {
            ReleaseCubeSamplerSlot(slots, env.ImportedSceneWorld->DiffuseEnv.SamplerSlot);
            ReleaseCubeSamplerSlot(slots, env.ImportedSceneWorld->SpecularEnv.SamplerSlot);
            env.ImportedSceneWorld.reset();
        }
        env.SceneWorldRotation = mat3{1.f};
        env.SceneWorld = {.Ibl = MakeIblSamplers(env.EmptySceneWorld, env), .Name = env.EmptySceneWorld.Name};
        r.remove<PendingSceneWorldClear>(viewport);
    }
    if (auto *pending_env = r.try_get<PendingEnvironmentImport>(viewport)) {
        if (const auto *src = r.try_get<const gltf::SourceAssets>(viewport)) {
            auto batch = BeginTextureUploadBatch(vk.Device, *one_shot.Pool, buffers.Ctx);
            auto pre = MaterializeEnvironmentImport(vk, batch, slots, *pending_env, src->Images);
            SubmitTextureUploadBatch(batch, vk.Queue, *one_shot.Fence, vk.Device);
            if (pre) {
                auto &env = environments;
                if (env.ImportedSceneWorld) {
                    ReleaseCubeSamplerSlot(slots, env.ImportedSceneWorld->DiffuseEnv.SamplerSlot);
                    ReleaseCubeSamplerSlot(slots, env.ImportedSceneWorld->SpecularEnv.SamplerSlot);
                }
                env.ImportedSceneWorld = std::move(*pre);
                env.SceneWorldRotation = glm::mat3_cast(pending_env->Source.Rotation);
                env.SceneWorld = {.Ibl = MakeIblSamplers(*env.ImportedSceneWorld, env), .Name = env.ImportedSceneWorld->Name};
            } else {
                std::cerr << std::format("Warning: Failed to materialize EXT_lights_image_based '{}': {}\n", pending_env->Source.Name, pre.error());
                ReleaseCubeSamplerSlot(slots, pending_env->DiffuseCubeSlot);
                ReleaseCubeSamplerSlot(slots, pending_env->SpecularCubeSlot);
            }
        }
        r.remove<PendingEnvironmentImport>(viewport);
    }
    // Drop encoded Bytes for external-URI images now that materialization has consumed them.
    // SourceAbsPath is the persistence — SaveGltf re-reads from there.
    if (!r.any_of<PendingTextureUploads, PendingEnvironmentImport>(viewport)) {
        if (auto *src_assets = r.try_get<gltf::SourceAssets>(viewport)) {
            for (auto &img : src_assets->Images) {
                if (!img.Uri.empty()) img.Bytes = {};
            }
        }
    }

    // Prefilter and activate the studio HDRI named by the StudioEnvironment selection whenever it changes.
    if (!reactive<changes::StudioEnvironment>(r).empty()) {
        SetStudioEnvironment(r, r.get<const StudioEnvironment>(viewport).Name);
    }

    // Pending* handlers run before the reactive checks so their patches land in trackers in time.
    if (const auto *pending = r.try_get<const PendingSetEditMode>(viewport)) {
        const auto mode = pending->Mode;
        r.remove<PendingSetEditMode>(viewport);
        SetEditMode(r, viewport, mode);
    }
    if (auto *pending = r.try_get<PendingImportMesh>(viewport)) {
        auto path = std::move(pending->Path);
        auto info = std::move(pending->Info);
        r.remove<PendingImportMesh>(viewport);
        ImportMesh(r, path, std::move(info));
    }
    // Selection resolves on the GPU through SceneViewUBO.ViewProj, and replay doesn't log view-camera navigation,
    // so each pending selection carries the view-projection it was recorded with: stamp it for the resolve.
    // No restore needed - the UBO ends the frame live regardless: a view change re-fills it at the rebuild below
    // (which runs after selection), and if the view didn't change the captured view already equals the live one.
    // Stamping a recorded view invalidates the selection buffer, so mark it stale to force a re-render against it.
    const auto stamp_view_proj = [&r, &buffers](const mat4 &view_proj) {
        buffers.SceneViewUBO.Update(as_bytes(view_proj), offsetof(SceneViewUBO, ViewProj));
        r.ctx().get<DrawState>().SelectionStale = true;
    };
    if (const auto *pending = r.try_get<const PendingEditElementClick>(viewport)) {
        const auto mouse_px = pending->MousePx;
        const bool toggle = pending->Toggle;
        stamp_view_proj(pending->ViewProj);
        r.remove<PendingEditElementClick>(viewport);

        const auto edit_mode = r.get<const EditMode>(viewport).Value;
        const auto ranges = GetBitsetRangesForSelected(r);
        auto *bits = buffers.SelectionBitset.Data();
        if (!toggle) {
            for (const auto &range : ranges) {
                const uint32_t first_word = range.Offset / 32;
                const uint32_t last_word = (range.Offset + range.Count + 31) / 32;
                memset(&bits[first_word], 0, (last_word - first_word) * sizeof(uint32_t));
                r.remove<MeshActiveElement>(range.MeshEntity);
            }
        }
        const auto hit = RunElementPickFromRanges(r, viewport, ranges, edit_mode, mouse_px);
        if (hit) {
            const auto [mesh_entity, element_index] = *hit;
            const auto *current_active = r.try_get<MeshActiveElement>(mesh_entity);
            const bool is_active = current_active && current_active->Handle == element_index;
            if (const auto *br = r.try_get<const MeshSelectionBitsetRange>(mesh_entity)) {
                const uint32_t global_bit = br->Offset + element_index;
                const bool was_selected = (bits[global_bit >> 5] >> (global_bit & 31)) & 1;
                if (toggle && was_selected) bits[global_bit >> 5] &= ~(1u << (global_bit & 31));
                else bits[global_bit >> 5] |= 1u << (global_bit & 31);
            }
            if (toggle && is_active) r.remove<MeshActiveElement>(mesh_entity);
            else r.emplace_or_replace<MeshActiveElement>(mesh_entity, element_index);
        } else if (!toggle) {
            for (const auto &range : ranges) r.remove<MeshActiveElement>(range.MeshEntity);
        }
        if (!ranges.empty() && (!toggle || hit)) r.emplace_or_replace<SelectionBitsDirty>(viewport);
    }
    if (const auto *pending = r.try_get<const PendingBoxSelect>(viewport)) {
        const auto box_px = pending->BoxPx;
        const bool additive = pending->Additive;
        stamp_view_proj(pending->ViewProj);
        r.remove<PendingBoxSelect>(viewport);

        const auto &interaction = r.get<const Interaction>(viewport);
        if (interaction.Mode == InteractionMode::Edit && FindArmatureObject(r, FindActiveEntity(r)) == entt::null) {
            // Edit-mode element box-select resolves straight into the GPU selection buffers.
            RunBoxSelectElements(r, viewport, GetBitsetRangesForSelected(r), r.get<const EditMode>(viewport).Value, box_px, additive);
        } else {
            const bool bone_mode = interaction.Mode == InteractionMode::Pose || IsBoneEditMode(r, viewport);
            const auto hits = ResolveHits(r, RunBoxSelect(r, viewport, box_px), bone_mode, true);
            const auto *baseline = additive ? r.try_get<const AdditiveBoxSelectBaseline>(viewport) : nullptr;
            if (bone_mode) {
                r.clear<BoneSelection>();
                if (baseline) {
                    for (const auto &[e, sel] : baseline->BoneSelections) {
                        if (r.valid(e)) r.emplace_or_replace<BoneSelection>(e, sel);
                    }
                }
                for (const auto &hit : hits) {
                    const auto sel = hit.Part ? BoneSelection::From(*hit.Part) : BoneSelection{};
                    const auto *cur = r.try_get<BoneSelection>(hit.Entity);
                    r.emplace_or_replace<BoneSelection>(hit.Entity, additive && cur ? *cur | sel : sel);
                }
            } else {
                r.clear<Selected>();
                if (baseline) {
                    for (const auto e : baseline->SelectedEntities) {
                        if (r.valid(e)) r.emplace_or_replace<Selected>(e);
                    }
                }
                for (const auto &hit : hits) r.emplace_or_replace<Selected>(hit.Entity);
            }
        }
    }
    if (const auto *pending = r.try_get<const PendingPick>(viewport)) {
        const auto mouse_px = pending->MousePx;
        const bool shift = pending->Shift, cycle = pending->Cycle;
        stamp_view_proj(pending->ViewProj);
        r.remove<PendingPick>(viewport);

        const bool bone_mode = r.get<const Interaction>(viewport).Mode == InteractionMode::Pose || IsBoneEditMode(r, viewport);
        const auto active = bone_mode ? FindActiveBone(r) : FindActiveEntity(r);
        const auto logical_extent = r.ctx().get<ViewportExtent>().Value;
        const auto render_extent = RenderExtentPx(r);
        const float render_scale = std::max(
            logical_extent.x > 0u ? float(render_extent.x) / float(logical_extent.x) : 1.f,
            logical_extent.y > 0u ? float(render_extent.y) / float(logical_extent.y) : 1.f
        );
        const auto radius = std::max(1u, uint32_t(std::lround(float(ObjectSelectRadiusPx) * render_scale)));
        auto &frame = r.ctx().get<FrameState>();
        const auto hits = ResolveHits(r, RunObjectPick(r, viewport, frame.ObjectPickEpochTag, mouse_px, radius), bone_mode);
        const auto pick = hits.empty() ? std::optional<SelectionHit>{} : [&]() -> std::optional<SelectionHit> {
            if (!cycle) return hits.front();
            // Re-click at the same spot: advance from the currently-active hit to the next overlapping one.
            const auto *bs = r.try_get<const BoneSelection>(active);
            auto it = std::ranges::find_if(hits, [&](const SelectionHit &h) { return h.Entity == active && (!h.Part || (bs && bs->Has(*h.Part))); });
            return it != hits.end() && ++it != hits.end() ? *it : hits.front();
        }();
        using namespace action::selection;
        if (pick && shift) {
            if (active == pick->Entity && !bone_mode) Apply(r, viewport, ToggleSelected{pick->Entity});
            else if (bone_mode) Apply(r, viewport, ExtendBoneActive{pick->Entity, pick->Part, true});
            else Apply(r, viewport, ExtendActive{pick->Entity});
        } else if (pick || !shift) {
            if (pick && bone_mode) Apply(r, viewport, SelectBone{pick->Entity, pick->Part, false});
            else if (pick) Apply(r, viewport, Select{pick->Entity});
            else Apply(r, viewport, DeselectAll{});
        }
    }

    // Create/destroy wireframe overlay instances and their buffer entities before SyncModelsBuffers
    // consumes the RenderInstance/NewBufferEntity reactive events they fire.
    EnsureWireframes(r, viewport);

    // Keep camera/light/empty gizmo wireframes in sync with their objects: rebuild on object creation and on a
    // shape-affecting param change (a light's color/intensity edit changes no geometry, so it isn't triggered).
    {
        const auto rebuild = [&](entt::entity object, ObjectType type) {
            if (const auto *inst = r.try_get<const Instance>(object); inst && r.valid(inst->Entity)) {
                RebuildGizmoGeometry(r, meshes, buffers, object, inst->Entity, type);
                request(RenderRequest::Submit);
            }
        };
        for (auto object : reactive<changes::ObjectCreated>(r)) {
            // Cameras rebuild via their lens reactive below; here, lights and empties.
            if (const auto type = r.get<const ObjectKind>(object).Value; type == ObjectType::Light || type == ObjectType::Empty) rebuild(object, type);
        }
        for (auto object : reactive<changes::CameraLens>(r)) {
            if (r.all_of<Camera>(object)) rebuild(object, ObjectType::Camera);
        }
        for (auto object : r.view<LightWireframeDirty>()) rebuild(object, ObjectType::Light);
    }

    auto sync = SyncModelsBuffers(r); // Runs first so BufferIndex is valid for all downstream code.
    if (!sync.NewlyInserted.empty() || sync.Compacted) request(RenderRequest::Submit);
    const std::unordered_set<entt::entity> newly_inserted_set(sync.NewlyInserted.begin(), sync.NewlyInserted.end());
    const auto is_newly_inserted = [&](entt::entity e) { return newly_inserted_set.contains(e); };

    // Build the derived ArmaturePoseState for any armature missing one.
    // ArmaturePose (canonical) is restored or starts at rest; bone Transforms are recomputed from it.
    bool pose_state_created = false;
    for (const auto [arm_obj_entity, arm_obj_comp] : r.view<const ArmatureObject>().each()) {
        const auto data_entity = arm_obj_comp.Entity;
        const auto *armature = r.try_get<const Armature>(data_entity);
        if (!armature || armature->Bones.empty() || r.all_of<ArmaturePoseState>(data_entity)) continue;
        const auto n = armature->Bones.size();
        if (!r.all_of<ArmaturePose>(data_entity)) r.emplace<ArmaturePose>(data_entity, std::vector<Transform>(n));
        r.emplace<ArmaturePoseState>(
            data_entity,
            ArmaturePoseState{.BoneUserOffset = std::vector<Transform>(n), .BonePoseWorld = std::vector<mat4>(n, I4), .GpuDeformRange = {}}
        );
        // Bone Transform is derived (unserialized), so reconstruct it from rest + delta. Scale stays at rest.
        const auto &deltas = r.get<const ArmaturePose>(data_entity).BoneDeltas;
        for (uint32_t i = 0; i < n && i < arm_obj_comp.BoneEntities.size(); ++i) {
            const auto &rest = armature->Bones[i].RestLocal;
            const auto posed = ComposeWithDelta(rest, deltas[i]);
            r.emplace_or_replace<Transform>(arm_obj_comp.BoneEntities[i], Transform{posed.P, posed.R, rest.S});
        }
        pose_state_created = true;
    }

    {
        // Allocate ArmatureDeform for newly created ArmaturePoseState (GpuDeformRange.Count == 0).
        // Count total joints, single Reserve, then per-armature Allocate.
        uint32_t total_joints = 0;
        std::vector<entt::entity> pending_armatures; // armature data entities
        for (const auto [arm_obj_entity, arm_obj_comp] : r.view<const ArmatureObject>().each()) {
            auto *pose_state = r.try_get<ArmaturePoseState>(arm_obj_comp.Entity);
            if (!pose_state || pose_state->GpuDeformRange.Count > 0) continue;
            const auto *armature = r.try_get<const Armature>(arm_obj_comp.Entity);
            if (armature && armature->ImportedSkin) {
                total_joints += armature->ImportedSkin->OrderedJointNodeIndices.size();
                pending_armatures.emplace_back(arm_obj_comp.Entity);
            }
        }
        buffers.ArmatureDeformBuffer.ReserveAdditional(total_joints);
        for (const auto arm_data_entity : pending_armatures) {
            auto &pose_state = r.get<ArmaturePoseState>(arm_data_entity);
            const auto &armature = r.get<const Armature>(arm_data_entity);
            pose_state.GpuDeformRange = buffers.ArmatureDeformBuffer.Allocate(armature.ImportedSkin->OrderedJointNodeIndices.size());
        }
        if (!pending_armatures.empty()) request(RenderRequest::ReRecord);
    }

    {
        // Allocate MorphWeights for newly created MorphWeightState (GpuWeightRange.Count == 0).
        // Count total weights, single Reserve, then per-instance Allocate + copy.
        uint32_t total_weights = 0;
        std::vector<entt::entity> pending_morphs;
        for (auto [entity, morph_state] : r.view<MorphWeightState>().each()) {
            if (morph_state.Weights.empty()) continue;

            const auto *gpu = r.try_get<const MorphWeightGpuRange>(entity);
            if (!gpu || gpu->Weights.Count == 0) {
                total_weights += morph_state.Weights.size();
                pending_morphs.emplace_back(entity);
            }
        }
        buffers.MorphWeightBuffer.ReserveAdditional(total_weights);
        for (const auto entity : pending_morphs) {
            const auto &morph_state = r.get<const MorphWeightState>(entity);
            const auto range = buffers.MorphWeightBuffer.Allocate(morph_state.Weights.size());
            r.emplace_or_replace<MorphWeightGpuRange>(entity, MorphWeightGpuRange{range});
            auto gpu_weights = buffers.MorphWeightBuffer.GetMutable(range);
            std::copy(morph_state.Weights.begin(), morph_state.Weights.end(), gpu_weights.begin());
        }
        if (!pending_morphs.empty()) request(RenderRequest::ReRecord);
    }

    // Deferred index buffer creation for new mesh entities.
    // Count total indices, then write directly into GPU-mapped memory.
    if (!sync.NewMeshEntities.empty()) {
        uint32_t total_face = 0, total_edge = 0, total_vertex = 0;
        for (auto entity : sync.NewMeshEntities) {
            const auto &mesh = GetMesh(r, entity);
            total_face += mesh.TriangleIndexCount();
            total_edge += mesh.EdgeCount() * 2;
            total_vertex += mesh.VertexCount();
        }
        buffers.ReserveAdditionalIndices(total_face, total_edge, total_vertex);
        for (auto entity : sync.NewMeshEntities) {
            const auto &mesh = GetMesh(r, entity);
            r.patch<MeshBuffers>(entity, [&](auto &mb) {
                if (const auto tri_idx_count = mesh.TriangleIndexCount(); tri_idx_count > 0) {
                    auto [sr, dest] = buffers.AllocateIndices(tri_idx_count, IndexKind::Face);
                    mesh.WriteTriangleIndices(dest);
                    mb.FaceIndices = sr;
                }
                if (mesh.EdgeCount() > 0) {
                    auto [sr, dest] = buffers.AllocateIndices(mesh.EdgeCount() * 2, IndexKind::Edge);
                    mesh.WriteEdgeIndices(dest);
                    mb.EdgeIndices = sr;
                }
                if (mesh.VertexCount() > 0) {
                    auto [sr, dest] = buffers.AllocateIndices(mesh.VertexCount(), IndexKind::Vertex);
                    std::iota(dest.begin(), dest.end(), 0u);
                    mb.VertexIndices = sr;
                }
            });
        }
        request(RenderRequest::ReRecord);
    }

    // Deferred index buffer creation for new bone/joint buffer entities.
    if (!sync.NewExtrasEntities.empty()) {
        auto flatten_tri_indices = [](const std::vector<std::vector<uint32_t>> &faces) {
            std::vector<uint32_t> indices;
            indices.reserve(faces.size() * 3);
            for (const auto &face : faces)
                for (const auto idx : face) indices.emplace_back(idx);
            return indices;
        };

        // Shared primitive data — all bones/joints use identical geometry. Generated once.
        static const auto bone = primitive::BoneOctahedron(1.f);
        static const auto bone_faces = flatten_tri_indices(bone.Mesh.Faces);
        static const auto bone_verts = iota(0u, uint32_t(bone.Mesh.Positions.size())) | to<std::vector>();
        static const auto sphere = primitive::BoneSphereDisc();
        static const auto sphere_faces = flatten_tri_indices(sphere.Mesh.Faces);
        static const auto sphere_verts = iota(0u, uint32_t(sphere.Mesh.Positions.size())) | to<std::vector>();

        uint32_t total_face = 0, total_edge = 0, total_vertex = 0;
        for (auto entity : sync.NewExtrasEntities) {
            if (r.all_of<ArmatureObject>(entity)) {
                total_face += bone_faces.size();
                total_edge += bone.AdjacencyIndices.size();
                total_vertex += bone_verts.size();
            } else if (r.all_of<BoneJoint>(entity)) {
                total_face += sphere_faces.size();
                total_edge += sphere.OutlineIndices.size();
                total_vertex += sphere_verts.size();
            } else if (const auto *pending = r.try_get<const PendingEdgeIndices>(entity)) {
                total_edge += pending->Indices.size();
            }
        }
        buffers.ReserveAdditionalIndices(total_face, total_edge, total_vertex);

        for (auto entity : sync.NewExtrasEntities) {
            if (r.all_of<ArmatureObject>(entity)) {
                r.patch<MeshBuffers>(entity, [&](auto &mb) {
                    mb.FaceIndices = buffers.CreateIndices(bone_faces, IndexKind::Face);
                    mb.VertexIndices = buffers.CreateIndices(bone_verts, IndexKind::Vertex);
                });
                r.emplace_or_replace<BoneAdjacencyIndices>(entity, buffers.CreateIndices(bone.AdjacencyIndices, IndexKind::Edge));
            } else if (r.all_of<BoneJoint>(entity)) {
                r.patch<MeshBuffers>(entity, [&](auto &mb) {
                    mb.FaceIndices = buffers.CreateIndices(sphere_faces, IndexKind::Face);
                    mb.EdgeIndices = buffers.CreateIndices(sphere.OutlineIndices, IndexKind::Edge);
                    mb.VertexIndices = buffers.CreateIndices(sphere_verts, IndexKind::Vertex);
                });
            } else if (auto *pending = r.try_get<PendingEdgeIndices>(entity)) {
                r.patch<MeshBuffers>(entity, [&](auto &mb) {
                    mb.EdgeIndices = buffers.CreateIndices(pending->Indices, IndexKind::Edge);
                });
                r.remove<PendingEdgeIndices>(entity);
            }
        }
        request(RenderRequest::ReRecord);
    }

    { // Register changed lights into the GPU Lights buffer, the single path for both new and restored lights.
        bool synced = false;
        for (const auto entity : reactive<changes::PunctualLight>(r)) {
            if (!r.valid(entity) || !r.all_of<PunctualLight, Instance>(entity)) continue;
            const auto *ri = r.try_get<const RenderInstance>(entity);
            if (!ri || ri->BufferIndex == UINT32_MAX) continue;
            const auto index = r.all_of<LightIndex>(entity) ? r.get<const LightIndex>(entity).Value : buffers.Lights.Count();
            if (!r.all_of<LightIndex>(entity)) r.emplace<LightIndex>(entity, index);
            // Write a copy with the transform slot offset, leaving the authored component untouched.
            auto gpu_light = r.get<const PunctualLight>(entity);
            gpu_light.TransformSlotOffset = {buffers.Instances.TransformBuffer.Slot, ri->BufferIndex};
            buffers.Lights.Set(index, gpu_light);
            synced = true;
        }
        if (synced) request(RenderRequest::Submit);
    }

    // Batch-compact light buffer for destroyed lights (indices collected in Destroy()).
    if (auto *pending = r.try_get<PendingLightRemovals>(viewport); pending && !pending->Indices.empty()) {
        auto &indices = pending->Indices;
        std::sort(indices.begin(), indices.end(), std::greater<>());
        auto buffer_count = buffers.Lights.Count();
        for (const auto remove_index : indices) {
            if (remove_index >= buffer_count) continue; // Light was never synced to GPU (created and destroyed same frame).
            --buffer_count;
            if (remove_index != buffer_count) {
                buffers.Lights.Set(remove_index, buffers.Lights.Get(buffer_count));
                for (auto [other_entity, other_light_index] : r.view<LightIndex>().each()) {
                    if (other_light_index.Value == buffer_count) {
                        r.replace<LightIndex>(other_entity, remove_index);
                        break;
                    }
                }
            }
        }
        buffers.Lights.SetCount(buffer_count);
        r.remove<PendingLightRemovals>(viewport);
        request(RenderRequest::ReRecord);
    }

    if (const auto *handlers = r.ctx().find<std::vector<ComponentEventHandler>>()) {
        for (const auto &h : *handlers) h(r);
    }

    { // Note: Can mutate InteractionMode, so do this first before `changes::InteractionMode` handling below.
        const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
        auto &enabled_modes = r.get<EnabledInteractionModes>(viewport).Value;
        if (r.storage<SoundVertices>().empty()) {
            if (interaction_mode == InteractionMode::Excite) SetInteractionMode(r, viewport, *enabled_modes.begin());
            enabled_modes.erase(InteractionMode::Excite);
        } else if (!reactive<changes::SoundVertices>(r).empty()) {
            enabled_modes.insert(InteractionMode::Excite);
            if (interaction_mode == InteractionMode::Excite) request(RenderRequest::ReRecord);
            else SetInteractionMode(r, viewport, InteractionMode::Excite);
        }
    }
    std::unordered_set<entt::entity> dirty_overlay_meshes, dirty_element_state_meshes;

    { // Selected/Active instance changes - batch instance state buffer writes per buffer entity.
        auto &selected_tracker = reactive<changes::Selected>(r);
        auto &active_tracker = reactive<changes::ActiveInstance>(r);
        if (!selected_tracker.empty()) {
            // In Edit mode, selection changes primary_edit_instances which affects fill/edge/point batches.
            // In Object/Pose mode, only the silhouette batch is affected.
            const auto mode = r.get<const Interaction>(viewport).Mode;
            request(mode == InteractionMode::Edit ? RenderRequest::ReRecord : RenderRequest::ReRecordSilhouette);
        }

        // Collect instance state writes, then batch via GetMutableRange.
        // SyncModelsBuffers writes the full initial state for newly inserted instances,
        // so we skip those here to avoid redundant writes.
        std::vector<std::pair<uint32_t, uint8_t>> state_writes;
        const auto collect_instance_state = [&](entt::entity instance_entity) {
            if (is_newly_inserted(instance_entity)) return;
            if (const auto *ri = r.try_get<RenderInstance>(instance_entity); ri && ri->BufferIndex != UINT32_MAX) {
                state_writes.emplace_back(ri->BufferIndex, InstanceStateBits(r, instance_entity));
            }
        };
        // Update SelectedInstanceCount on mesh entities before per-entity processing.
        for (auto instance_entity : selected_tracker) {
            if (const auto *instance = r.try_get<Instance>(instance_entity); instance && HasMesh(r, instance->Entity)) {
                auto &sc = r.get_or_emplace<SelectedInstanceCount>(instance->Entity);
                sc.Value += r.all_of<Selected>(instance_entity) ? 1 : -1;
            }
        }
        for (auto instance_entity : selected_tracker) {
            collect_instance_state(instance_entity);
            if (const auto arm = FindArmatureObject(r, instance_entity); arm != entt::null) r.emplace_or_replace<BoneInstanceStateDirty>(arm);
            if (const auto *instance = r.try_get<Instance>(instance_entity); instance && HasMesh(r, instance->Entity)) {
                const auto mesh_entity = instance->Entity;
                if (r.all_of<Selected>(instance_entity)) {
                    dirty_overlay_meshes.insert(mesh_entity);
                } else if (!r.get_or_emplace<SelectedInstanceCount>(mesh_entity).Value) {
                    // Clean up overlays for this mesh
                    if (auto *mesh_buffers = r.try_get<MeshBuffers>(mesh_entity)) {
                        for (auto &[_, rb] : mesh_buffers->NormalIndicators) buffers.Release(rb);
                        mesh_buffers->NormalIndicators.clear();
                    }
                }
            }
        }
        for (auto instance_entity : active_tracker) {
            collect_instance_state(instance_entity);
            if (const auto arm = FindArmatureObject(r, instance_entity); arm != entt::null) r.emplace_or_replace<BoneInstanceStateDirty>(arm);
        }

        // Batch-write all collected instance state changes.
        if (FlushIndexedWrites(state_writes, [&] { return buffers.Instances.GetMutableStates(); })) request(RenderRequest::Submit);
    }
    { // Bone selection changes — tag armature objects for GPU state sync.
        auto &bone_sel_tracker = reactive<changes::BoneSelection>(r);
        if (!bone_sel_tracker.empty()) {
            request(RenderRequest::ReRecordSilhouette);
            for (auto bone_entity : bone_sel_tracker) {
                if (const auto arm = FindArmatureObject(r, bone_entity); arm != entt::null) r.emplace_or_replace<BoneInstanceStateDirty>(arm);
            }
        }
    }
    auto &destroy_tracker = r.ctx().get<EntityDestroyTracker>();
    if (!reactive<changes::Rerecord>(r).empty() || !destroy_tracker.Storage.empty()) request(RenderRequest::ReRecord);

    const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
    const bool is_edit_mode = interaction_mode == InteractionMode::Edit;

    const auto edit_transform_context = is_edit_mode ? EditTransformContext{selection::ComputePrimaryEditInstances(r, false)} : EditTransformContext{};
    const auto orbit_to_active = [&](entt::entity instance_entity, Element element, uint32_t handle) {
        if (!r.get<const OrbitToActive>(viewport).Value) return;
        const auto world_pos = ComputeElementWorldPosition(r, instance_entity, element, handle);
        r.patch<ViewCamera>(viewport, [&](auto &camera) {
            if (const auto dir = world_pos - camera.Target; glm::dot(dir, dir) >= 1e-6f) {
                camera.SetTargetDirection(glm::normalize(dir));
            }
        });
    };

    if (r.all_of<SelectionBitsDirty>(viewport)) {
        r.remove<SelectionBitsDirty>(viewport);
        if (is_edit_mode) ApplySelectionStateUpdate(r, viewport, GetBitsetRangesForSelected(r), r.get<const EditMode>(viewport).Value);
    }
    if (const auto &tracker = reactive<changes::MeshActiveElement>(r); !tracker.empty()) {
        const auto edit_mode = r.get<const EditMode>(viewport).Value;
        const auto active_entity = FindActiveEntity(r);
        const auto *active_instance = r.try_get<Instance>(active_entity);
        for (auto mesh_entity : tracker) {
            if (const auto *active_element = r.try_get<MeshActiveElement>(mesh_entity);
                active_element && edit_mode != Element::None && active_instance && active_instance->Entity == mesh_entity) {
                orbit_to_active(active_entity, edit_mode, active_element->Handle);
            }
            dirty_element_state_meshes.insert(mesh_entity); // for Excite mode
        }
    }
    for (auto instance_entity : reactive<changes::VertexForce>(r)) {
        if (const auto *inst = r.try_get<Instance>(instance_entity)) dirty_element_state_meshes.insert(inst->Entity);
        if (const auto *ev = r.try_get<VertexForce>(instance_entity)) orbit_to_active(instance_entity, Element::Vertex, ev->Vertex);
    }
    for (auto instance_entity : reactive<changes::SoundVerticesUpdated>(r)) {
        if (const auto *inst = r.try_get<const Instance>(instance_entity)) dirty_element_state_meshes.insert(inst->Entity);
    }
    for (auto camera_entity : reactive<changes::CameraLens>(r)) {
        // When looking through this camera, refresh the ViewCamera so the view picks up its FOV.
        if (r.all_of<Camera>(camera_entity) && r.all_of<LookingThrough>(camera_entity)) r.patch<ViewCamera>(viewport, [](auto &) {});
    }
    bool light_count_changed = false;
    if (const uint32_t required_count = r.storage<LightIndex>().size();
        buffers.Lights.Count() != required_count) {
        buffers.Lights.SetCount(required_count);
        light_count_changed = true;
    }
    if (light_count_changed) request(RenderRequest::Submit);
    if (!reactive<changes::WorkspaceLights>(r).empty()) {
        buffers.WorkspaceLightsUBO.Update(as_bytes(r.get<const WorkspaceLights>(viewport)));
        request(RenderRequest::Submit);
    }
    if (auto &tracker = reactive<changes::MeshGeometry>(r); !tracker.empty()) {
        const auto edit_mode = r.get<const EditMode>(viewport).Value;
        std::vector<ElementRange> geometry_ranges;
        for (auto mesh_entity : tracker) {
            if (r.get_or_emplace<SelectedInstanceCount>(mesh_entity).Value > 0) dirty_overlay_meshes.insert(mesh_entity);
            if (auto *br = r.try_get<MeshSelectionBitsetRange>(mesh_entity); br && edit_mode != Element::None) {
                // Topology changed: zero stale selection bits and update count.
                const auto &mesh = GetMesh(r, mesh_entity);
                const uint32_t new_count = selection::GetElementCount(mesh, edit_mode);
                const uint32_t max_words = (std::max(br->Count, new_count) + 31) / 32;
                memset(&buffers.SelectionBitset.Data()[br->Offset / 32], 0, max_words * sizeof(uint32_t));
                br->Count = new_count;
                if (new_count > 0) geometry_ranges.emplace_back(mesh_entity, br->Offset, br->Count);
            }
        }
        if (!geometry_ranges.empty()) ApplySelectionStateUpdate(r, viewport, geometry_ranges, edit_mode);
        request(RenderRequest::Submit);
    }
    if (auto &tracker = reactive<changes::MeshMaterial>(r); !tracker.empty()) {
        for (auto mesh_entity : tracker) {
            const auto *assignment = r.try_get<const MeshMaterialAssignment>(mesh_entity);
            const auto mesh = TryGetMesh(r, mesh_entity);
            if (!assignment || !mesh) continue;
            const auto material_count = buffers.Materials.Count();
            if (material_count == 0u) continue;
            auto primitive_materials = meshes.GetPrimitiveMaterialIndices(mesh->GetStoreId());
            if (assignment->PrimitiveIndex < primitive_materials.size()) {
                primitive_materials[assignment->PrimitiveIndex] = std::min(assignment->MaterialIndex, material_count - 1u);
            }
        }
        request(RenderRequest::Submit);
    }
    if (!reactive<changes::ViewportTheme>(r).empty()) {
        UpdateDerivedColors(r.get<ViewportTheme>(viewport));
        auto theme = r.get<const ViewportTheme>(viewport);
        theme.EdgeWidth *= r.ctx().get<FrameState>().DisplayFramebufferScale.x;
        buffers.ViewportThemeUBO.Update(as_bytes(theme));
        request(RenderRequest::Submit);
    }
    if (!reactive<changes::Materials>(r).empty()) {
        if (const auto *dirty = r.try_get<const MaterialDirty>(viewport);
            dirty && dirty->Index < buffers.Materials.Count()) {
            buffers.Materials.Set(dirty->Index, buffers.Materials.Get(dirty->Index));
        }
        request(RenderRequest::Submit);
    }
    if (!reactive<changes::ActiveMaterialVariant>(r).empty()) {
        const auto *mv = r.try_get<const MaterialVariants>(viewport);
        const auto active = mv ? mv->Active : std::nullopt;
        for (const auto [e, layout, _] : r.view<const MeshSourceLayout, const MeshHandle>().each()) {
            const auto mesh = GetMesh(r, e);
            auto primitive_materials = meshes.GetPrimitiveMaterialIndices(mesh.GetStoreId());
            for (size_t i = 0; i < layout.DefaultMaterials.size(); ++i) {
                const auto &mapping = layout.VariantMappings[i];
                primitive_materials[i] = active && *active < mapping.size() && mapping[*active] ?
                    *mapping[*active] :
                    layout.DefaultMaterials[i];
            }
        }
        request(RenderRequest::Submit);
    }
    if (!reactive<changes::ViewportDisplay>(r).empty()) {
        request(RenderRequest::ReRecord);
        dirty_overlay_meshes.merge(selection::GetSelectedMeshEntities(r));
        auto &vk_mut = r.ctx().get<VulkanResources>();
        if (const float requested = ClampMaxAnisotropy(vk_mut, ToMaxAnisotropy(r.get<const ViewportDisplay>(viewport).AnisotropicFilter));
            requested != vk_mut.MaxSamplerAnisotropy) {
            vk_mut.MaxSamplerAnisotropy = requested;
            RebuildTextureSamplers(vk_mut, r.ctx().get<DescriptorSlots>(), r.ctx().get<TextureStore>(), requested);
        }
    }
    if (!reactive<changes::InteractionMode>(r).empty()) {
        request(RenderRequest::ReRecord);
        // Dispatch UpdateSelectionState for all meshes entering Edit mode (MeshSelectionBitsetRange assigned in SetInteractionMode).
        if (r.get<const Interaction>(viewport).Mode == InteractionMode::Edit) {
            if (const auto edit_mode = r.get<const EditMode>(viewport).Value; edit_mode != Element::None) ApplySelectionStateUpdate(r, viewport, GetBitsetRangesForSelected(r), edit_mode);
        }
        for (const auto [_, instance, __] : r.view<const Instance, const SoundVertices>().each()) {
            dirty_element_state_meshes.insert(instance.Entity);
        }
        // Mark all armatures dirty for bone state + pose sync on mode change.
        for (const auto arm : r.view<ArmatureObject>()) r.emplace_or_replace<BoneInstanceStateDirty>(arm);
    }
    // Handle mesh Edit mode transform commit when StartTransform is cleared.
    // Bone Edit mode commits are handled in the bone pose transform section below.
    if (!reactive<changes::TransformEnd>(r).empty()) {
        if (is_edit_mode && FindArmatureObject(r, FindActiveEntity(r)) == entt::null) {
            if (const auto &pending = r.get<const PendingTransform>(viewport); pending.Delta != Transform{}) {
                // Apply edit transform once per selected mesh via a representative selected instance.
                // This keeps linked instances from receiving duplicate per-instance edits.
                for (const auto &[mesh_entity, instance_entity] : edit_transform_context.TransformInstances) {
                    if (selection::HasScaleLockedInstance(r, mesh_entity)) continue;
                    const auto &mesh = GetMesh(r, mesh_entity);
                    const auto vertex_states = meshes.GetVertexStates(mesh.GetStoreId());
                    const auto vertices = mesh.GetVerticesSpan();
                    const auto &wt = r.get<const WorldTransform>(instance_entity);
                    const auto inv_rot = glm::conjugate(wt.R);
                    const auto inv_scale = 1.f / wt.S;
                    const auto pivot_rel = pending.Pivot - wt.P;
                    bool any_moved{false};
                    for (uint32_t vi = 0; vi < vertex_states.size(); ++vi) {
                        if ((vertex_states[vi] & ElementStateSelected) == 0u) continue;
                        const auto local_pos = vertices[vi].Position;
                        const auto world_rel = glm::rotate(wt.R, wt.S * local_pos);
                        const auto offset = glm::rotate(pending.Delta.R, pending.Delta.S * (world_rel - pivot_rel));
                        const auto new_rel = pivot_rel + offset + pending.Delta.P;
                        const auto new_local = inv_scale * glm::rotate(inv_rot, new_rel);
                        if (glm::length2(new_local - local_pos) > 1e-12f) {
                            meshes.SetPosition(mesh, vi, new_local);
                            any_moved = true;
                        }
                    }
                    if (any_moved) {
                        meshes.UpdateNormals(mesh);
                        dirty_overlay_meshes.insert(mesh_entity);
                        r.remove<PrimitiveShape>(mesh_entity);
                        r.emplace_or_replace<MeshGeometryDirty>(mesh_entity);
                    }
                }
            }
            r.remove<PendingTransform>(viewport);
        }
    }

    { // Rederive colliders
        std::unordered_set<entt::entity> to_rederive;
        for (auto e : reactive<changes::ColliderPolicy>(r)) to_rederive.insert(e);
        if (const auto &mesh_dirty = reactive<changes::MeshGeometry>(r); !mesh_dirty.empty()) {
            for (auto [ce, cs] : r.view<const ColliderShape>().each()) {
                const auto me = cs.MeshEntity != null_entity ? cs.MeshEntity : FindMeshEntity(r, ce);
                if (mesh_dirty.contains(me)) to_rederive.insert(ce);
            }
        }
        for (auto e : to_rederive) RederiveCollider(r, e);
    }

    const bool mode_changed = !reactive<changes::InteractionMode>(r).empty();
    bool anim_advanced;
    // A moving pose or weight changes the recorded draw list only through the transparent sort (rendered shading with a blend-mode material).
    // Otherwise its values reach the GPU via already-bound buffers.
    const auto anim_render_request = [&] {
        const auto shading = r.get<const ViewportDisplay>(viewport).ViewportShading;
        if (shading != ViewportShadingMode::MaterialPreview && shading != ViewportShadingMode::Rendered) return RenderRequest::Submit;
        for (uint32_t i = 0, n = buffers.Materials.Count(); i < n; ++i) {
            if (buffers.Materials.Get(i).AlphaMode == MaterialAlphaMode::Blend) return RenderRequest::ReRecord;
        }
        return RenderRequest::Submit;
    }();
    { // Animation timeline tick
        const auto &range = r.get<const TimelineRange>(viewport);
        auto &playback = r.get<TimelinePlayback>(viewport);
        auto &pf = r.get<PlaybackFrame>(viewport).Value;
        auto &frame_state = r.ctx().get<FrameState>();
        anim_advanced = [&] {
            // A motion blur sub-frame pins `pf`; hold the displayed frame and re-evaluate poses there.
            if (frame_state.MotionBlurSubFrame) return true;
            if (playback.Playing) {
                pf += frame_state.FixedFrameStep ? 1.f : frame_state.DeltaTime * range.Fps;
                if (pf > float(range.EndFrame)) pf = float(range.StartFrame);
                const int new_frame = int(std::floor(pf));
                if (new_frame != playback.CurrentFrame) r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.CurrentFrame = new_frame; });
            } else {
                pf = float(playback.CurrentFrame);
            }
            return playback.CurrentFrame != r.get<LastEvaluatedFrame>(viewport).Value || !reactive<changes::ActiveAnimationClip>(r).empty();
        }();

        const bool cache_invalid = r.all_of<PhysicsCacheInvalid>(viewport);
        if (cache_invalid) r.remove<PhysicsCacheInvalid>(viewport);
        const bool range_changed = !reactive<changes::TimelineRange>(r).empty();
        const int from = r.get<LastEvaluatedFrame>(viewport).Value;
        // A motion-blur sub-frame pins the frame (from == to), so this no-ops; bodies come from interpolation instead.
        if (physics::AdvancePlayback(r, viewport, from, playback.CurrentFrame, range.StartFrame, range.EndFrame, range.Fps, range_changed, cache_invalid)) {
            request(RenderRequest::Submit);
        }

        if (anim_advanced) r.get<LastEvaluatedFrame>(viewport).Value = playback.CurrentFrame;
        // Timeline frames are displayed 1-based, but animation time starts at t=0 on frame 1.
        // A motion-blur sub-frame samples the continuous `pf` to evaluate clips between frames; the normal
        // path stays on the integer frame.
        const auto eval_seconds = (frame_state.MotionBlurSubFrame ? std::max(0.f, pf - 1.f) : float(std::max(0, playback.CurrentFrame - 1))) / range.Fps;
        const auto clip_time = [eval_seconds](const auto &clip) {
            return clip.DurationSeconds > 0 ? std::fmod(eval_seconds, clip.DurationSeconds) : 0.f;
        };

        // Evaluate animation deltas (data only — no bone entity iteration).
        bool request_rerecord = false;
        if (anim_advanced) {
            for (const auto [arm_obj_entity, arm_obj_comp] : r.view<const ArmatureObject>().each()) {
                auto *pose = r.try_get<ArmaturePose>(arm_obj_comp.Entity);
                if (!pose) continue;
                const auto &armature = r.get<const Armature>(arm_obj_comp.Entity);
                if (!armature.ImportedSkin) continue;
                if (const auto *anim = r.try_get<const ArmatureAnimation>(arm_obj_comp.Entity);
                    anim && !anim->Clips.empty() && anim->ActiveClipIndex < anim->Clips.size()) {
                    const auto &clip = anim->Clips[anim->ActiveClipIndex];
                    EvaluateAnimationDeltas(clip, clip_time(clip), armature.Bones, pose->BoneDeltas);
                }
            }
        }

        if (anim_advanced) {
            // Evaluate morph weight animations
            for (auto [entity, morph_anim, morph_state, gpu_range, instance] :
                 r.view<const MorphWeightAnimation, MorphWeightState, const MorphWeightGpuRange, const Instance>().each()) {
                if (morph_anim.Clips.empty() || morph_anim.ActiveClipIndex >= morph_anim.Clips.size()) continue;
                const auto &clip = morph_anim.Clips[morph_anim.ActiveClipIndex];
                const auto &mesh = GetMesh(r, instance.Entity);
                const auto default_weights = meshes.GetDefaultMorphWeights(mesh.GetStoreId());
                std::copy(default_weights.begin(), default_weights.end(), morph_state.Weights.begin());
                EvaluateMorphWeights(clip, clip_time(clip), morph_state.Weights);
                auto gpu_weights = buffers.MorphWeightBuffer.GetMutable(gpu_range.Weights);
                std::copy(morph_state.Weights.begin(), morph_state.Weights.end(), gpu_weights.begin());
                request_rerecord = true;
            }
        }

        // Evaluate node TRS animations into PosedLocal, leaving Transform at the authored local.
        // Runs when the frame advanced or PosedLocal is missing (first frame / restore).
        for (auto [entity, node_anim] : r.view<const NodeTransformAnimation>().each()) {
            if (node_anim.Clips.empty() || node_anim.ActiveClipIndex >= node_anim.Clips.size()) continue;
            if (!anim_advanced && r.all_of<PosedLocal>(entity)) continue;
            const auto &clip = node_anim.Clips[node_anim.ActiveClipIndex];
            std::array local_pose{r.get<const Transform>(entity)};
            EvaluateAnimation(clip, clip_time(clip), local_pose);
            r.emplace_or_replace<PosedLocal>(entity, local_pose.front());
            request_rerecord = true;
        }
        if (request_rerecord) request(anim_render_request);
    }
    { // Bones
        // GPU instance state
        const bool is_object_mode = interaction_mode == InteractionMode::Object;
        for (const auto arm_obj_entity : r.view<BoneInstanceStateDirty>()) {
            if (!r.all_of<MeshBuffers>(arm_obj_entity)) continue;
            const auto &arm_obj = r.get<const ArmatureObject>(arm_obj_entity);
            const auto &bone_entities = arm_obj.BoneEntities;
            // Object mode: all bone/joint slots get the armature's object-level state (no per-bone color).
            // Edit/Pose: each bone/joint gets its own state based on BoneActive/BoneSelection.
            uint8_t max_state = 0;
            if (is_object_mode) {
                // Only show active/selected bone colors when the armature is selected.
                // In solid mode, unselected armatures don't draw wires at all.
                // In wireframe mode, they draw with neutral Wire color.
                if (r.all_of<Selected>(arm_obj_entity)) {
                    max_state |= ElementStateSelected;
                    if (r.all_of<Active>(arm_obj_entity)) max_state |= ElementStateActive;
                }
            }
            const bool is_edit = interaction_mode == InteractionMode::Edit;
            auto compute_state = [&](entt::entity b, BoneSel part) {
                if (is_object_mode) return max_state;
                const auto *parts = r.try_get<const BoneSelection>(b);
                const bool selected = is_edit ?
                    parts && (part == BoneSel::Body ? parts->Body : part == BoneSel::Root ? parts->Root :
                                                                                            parts->Tip) :
                    r.all_of<BoneSelection>(b);
                uint8_t s = r.all_of<BoneActive>(b) ? ElementStateActive : 0;
                if (selected) s |= ElementStateSelected;
                return s;
            };

            for (const auto b : bone_entities) {
                if (const auto *ri = r.try_get<RenderInstance>(b)) {
                    const auto state = compute_state(b, BoneSel::Body);
                    buffers.Instances.UpdateState(ri->BufferIndex, state);
                }
            }
            if (arm_obj.JointEntity != entt::null && r.valid(arm_obj.JointEntity)) {
                for (const auto b : bone_entities) {
                    const auto *joints = r.try_get<const BoneJointEntities>(b);
                    if (!joints) continue;
                    for (const auto &[je, part] : {std::pair{joints->Head, BoneSel::Root}, {joints->Tail, BoneSel::Tip}}) {
                        if (je != entt::null) {
                            if (const auto *ri = r.try_get<const RenderInstance>(je)) {
                                const auto state = compute_state(b, part);
                                buffers.Instances.UpdateState(ri->BufferIndex, state);
                            }
                        }
                    }
                }
            }
            request(RenderRequest::Submit);
        }
        r.clear<BoneInstanceStateDirty>();

        // Bone pose sync: classify bone changes, update pose deltas, compute deform matrices.
        // Runs before the reactive WorldTransform pass so bone sync's Transform patches are included.
        // A new pose state has no GPU deform yet — refresh fills it from the reconstructed pose.
        const bool bones_need_refresh = anim_advanced || mode_changed || pose_state_created;
        if (bones_need_refresh || !reactive<changes::TransformDirty>(r).empty() || !reactive<changes::TransformEnd>(r).empty()) {
            const auto &local_changes = reactive<changes::TransformDirty>(r);
            const auto &transform_end = reactive<changes::TransformEnd>(r);
            for (const auto [arm_obj_entity, arm_obj_comp] : r.view<const ArmatureObject>().each()) {
                auto *pose_state = r.try_get<ArmaturePoseState>(arm_obj_comp.Entity);
                if (!pose_state) continue;
                auto &deltas = r.get<ArmaturePose>(arm_obj_comp.Entity).BoneDeltas;
                auto &armature = r.get<Armature>(arm_obj_comp.Entity);
                if (!armature.ImportedSkin) continue;

                // Constraints can depend on external targets (e.g. physics bodies), so we can't early-out on bone-dirty alone.
                const bool has_any_constraint = std::any_of(
                    arm_obj_comp.BoneEntities.begin(), arm_obj_comp.BoneEntities.end(),
                    [&](auto e) { return r.all_of<BoneConstraints>(e); }
                );

                if (!bones_need_refresh && !has_any_constraint) {
                    bool has_dirty = false;
                    for (const auto b : arm_obj_comp.BoneEntities) {
                        if (local_changes.contains(b) || transform_end.contains(b)) {
                            has_dirty = true;
                            break;
                        }
                    }
                    if (!has_dirty) continue;
                }

                const mat4 armature_world_inv = has_any_constraint ? glm::inverse(ToMatrix(r.get<const WorldTransform>(arm_obj_entity))) : I4;

                bool need_sync = has_any_constraint || pose_state_created;
                bool rest_pose_edited = false;
                for (uint32_t i = 0; i < arm_obj_comp.BoneEntities.size(); ++i) {
                    const auto b = arm_obj_comp.BoneEntities[i];
                    if (i >= deltas.size()) continue;
                    const auto &rest = armature.Bones[i].RestLocal;
                    const auto &bt = r.get<const Transform>(b);
                    Transform local{bt.P, bt.R, rest.S}; // Default; branches that recompute overwrite it.
                    bool should_patch = false;
                    if (is_edit_mode) {
                        if (mode_changed) {
                            // Entering Edit mode: snap to rest pose.
                            local = {rest.P, rest.R, rest.S};
                            should_patch = need_sync = true;
                        } else if (transform_end.contains(b) || (local_changes.contains(b) && !r.all_of<StartTransform>(b))) {
                            // Commit Edit mode transform (gizmo drag end or UI slider edit).
                            armature.Bones[i].RestLocal.P = bt.P;
                            armature.Bones[i].RestLocal.R = bt.R;
                            rest_pose_edited = need_sync = true;
                        }
                    } else if (const auto *st = r.try_get<const StartTransform>(b)) {
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
                        const Transform gizmo_local{bt.P, bt.R, rest.S};
                        pose_state->BoneUserOffset[i] = AbsoluteToDelta(grab_delta, AbsoluteToDelta(rest, gizmo_local));
                        local = ComposeWithDelta(rest, ComposeWithDelta(deltas[i], pose_state->BoneUserOffset[i]));
                        should_patch = need_sync = true;
                    } else if (transform_end.contains(b)) {
                        // Commit drag: bake P/R into the delta, then re-derive Transform so it stays Compose(rest, delta).
                        deltas[i] = AbsoluteToDelta(rest, {bt.P, bt.R, rest.S});
                        pose_state->BoneUserOffset[i] = {};
                        local = ComposeWithDelta(rest, deltas[i]);
                        should_patch = need_sync = true;
                    } else if (anim_advanced || mode_changed || pose_state_created) {
                        // Recompute entity P/R from the delta (also derives Transform on first load).
                        local = ComposeWithDelta(rest, ComposeWithDelta(deltas[i], pose_state->BoneUserOffset[i]));
                        should_patch = need_sync = true;
                    } else if (local_changes.contains(b)) {
                        // Manual transform: bake if P/R changed, then re-derive.
                        if (const auto expected = ComposeWithDelta(rest, ComposeWithDelta(deltas[i], pose_state->BoneUserOffset[i]));
                            bt.P != expected.P || bt.R != expected.R) {
                            deltas[i] = AbsoluteToDelta(rest, {bt.P, bt.R, rest.S});
                            pose_state->BoneUserOffset[i] = {};
                            local = ComposeWithDelta(rest, deltas[i]);
                            should_patch = need_sync = true;
                        }
                    }

                    const uint32_t parent_idx = armature.Bones[i].ParentIndex;
                    const mat4 parent_pose_world = (parent_idx == InvalidBoneIndex) ? I4 : pose_state->BonePoseWorld[parent_idx];

                    // Apply bone constraints. Skipped in edit mode (rest-pose edits bypass pose constraints).
                    if (!is_edit_mode) {
                        if (const auto *cs = r.try_get<const BoneConstraints>(b); cs && !cs->Stack.empty()) {
                            const auto before = local;
                            for (const auto &c : cs->Stack) {
                                if (c.TargetEntity == null_entity || !r.valid(c.TargetEntity)) continue;
                                const auto *twt = r.try_get<const WorldTransform>(c.TargetEntity);
                                if (twt) local = ApplyBoneConstraint(c, local, parent_pose_world, armature_world_inv, ToMatrix(*twt));
                            }
                            if (local.P != before.P || local.R != before.R) should_patch = true;
                        }
                    }

                    if (should_patch) r.patch<Transform>(b, [&](auto &t) { t.P = local.P; t.R = local.R; });
                    pose_state->BonePoseWorld[i] = parent_pose_world * ToMatrix(local);
                }
                if (rest_pose_edited) {
                    // Recompute RestWorld in topological order, preserving world positions of untransformed bones.
                    // Topological order guarantees: when we reach bone i, its RestWorld hasn't been overwritten yet
                    // (only ancestors have been processed), so we can read the pre-edit value directly.
                    for (uint32_t i = 0; i < armature.Bones.size(); ++i) {
                        const auto parent = armature.Bones[i].ParentIndex;
                        const mat4 parent_world = (parent == InvalidBoneIndex) ? I4 : armature.Bones[parent].RestWorld;
                        const auto b = arm_obj_comp.BoneEntities[i];
                        if (transform_end.contains(b) || (local_changes.contains(b) && !r.all_of<StartTransform>(b))) {
                            armature.Bones[i].RestWorld = parent_world * ToMatrix(armature.Bones[i].RestLocal);
                        } else {
                            // Preserve old world position, adjust RestLocal to compensate for parent change.
                            const mat4 new_local_mat = glm::inverse(parent_world) * armature.Bones[i].RestWorld;
                            armature.Bones[i].RestLocal.P = vec3(new_local_mat[3]);
                            armature.Bones[i].RestLocal.R = glm::normalize(glm::quat_cast(mat3(new_local_mat)));
                            r.patch<Transform>(b, [&](auto &t) { t.P = armature.Bones[i].RestLocal.P; t.R = armature.Bones[i].RestLocal.R; });
                        }
                        armature.Bones[i].InvRestWorld = glm::inverse(armature.Bones[i].RestWorld);
                    }
                    armature.RecomputeInverseBindMatrices();
                }
                if (need_sync) {
                    if (!is_edit_mode) {
                        ComputeDeformMatrices(
                            armature, pose_state->BonePoseWorld, armature.ImportedSkin->InverseBindMatrices,
                            buffers.ArmatureDeformBuffer.GetMutable(pose_state->GpuDeformRange)
                        );
                    }
                    request(anim_render_request);
                }
            }
        }
        // Recompute WorldTransform for entities whose local pose or parenting changed, and the descendants they drive.
        if (const auto &dirty = reactive<changes::TransformDirty>(r); !dirty.empty()) {
            const bool bone_edit = is_edit_mode && FindArmatureObject(r, FindActiveEntity(r)) != entt::null;
            std::unordered_set<entt::entity> recompute;
            const auto collect = [&](this const auto &self, entt::entity e, bool propagate) -> void {
                if (!recompute.insert(e).second) return;
                if (propagate)
                    for (const auto child : Children{&r, e}) self(child, true);
            };
            for (const auto e : dirty) {
                if (r.valid(e)) collect(e, !(bone_edit && r.all_of<StartTransform>(e)));
            }

            std::unordered_set<entt::entity> done;
            const auto compute = [&](this const auto &self, entt::entity e) -> void {
                if (!done.insert(e).second) return;
                const auto *node = r.try_get<const SceneNode>(e);
                if (node && node->Parent != entt::null && (recompute.contains(node->Parent) || !r.all_of<WorldTransform>(node->Parent))) {
                    self(node->Parent); // parent must be current before GetParentDelta reads it
                }
                const auto *posed = r.try_get<const PosedLocal>(e); // animated node: compose world from the pose
                const Transform &t = posed ? static_cast<const Transform &>(*posed) : r.get<const Transform>(e);
                if (node && node->Parent != entt::null) r.emplace_or_replace<WorldTransform>(e, ToTransform(GetParentDelta(r, e) * ToMatrix(t)));
                else r.emplace_or_replace<WorldTransform>(e, t);
            };
            for (const auto e : recompute) compute(e);
        }
        // Write deformed bounds into the object's instance slot: once per inserted instance, then
        // per-frame only while an AABB wireframe shows it.
        std::unordered_set<entt::entity> deformed_bounds_changed;
        {
            const auto &geom_changed = reactive<changes::MeshGeometry>(r);
            // Refresh the shared local AABB in every instance slot of each changed mesh.
            // Runs after every MeshGeometryDirty producer in this call.
            for (const auto mesh_entity : geom_changed) {
                const auto *models = r.try_get<const ModelsBuffer>(mesh_entity);
                if (!models || models->InstanceCount == 0) continue;
                if (const auto mesh = TryGetMesh(r, mesh_entity)) {
                    std::ranges::fill(buffers.Instances.GetMutableBounds({models->InstanceRange.Offset, models->InstanceCount}), mesh->CalcAABB());
                }
            }
            for (const auto object : r.view<const Instance, const RenderInstance>()) {
                if (!r.any_of<ArmatureModifier, MorphWeightState>(object)) continue;
                const auto &instance = r.get<const Instance>(object);
                const bool deformed_stored = r.all_of<DeformedBounds>(object);
                if (!r.all_of<AABBWireframe>(object) && deformed_stored && !geom_changed.contains(instance.Entity) && !is_newly_inserted(object)) continue;
                const auto mesh = TryGetMesh(r, instance.Entity);
                if (!mesh) continue;
                const auto buffer_index = r.get<const RenderInstance>(object).BufferIndex;
                if (buffer_index == UINT32_MAX) continue;
                if (const auto aabb = ComputeDeformedLocalAABB(r, meshes, buffers, object, *mesh)) {
                    buffers.Instances.UpdateBounds(buffer_index, *aabb);
                    if (!deformed_stored) r.emplace<DeformedBounds>(object);
                    deformed_bounds_changed.insert(object);
                } else if (deformed_stored) {
                    buffers.Instances.UpdateBounds(buffer_index, mesh->CalcAABB());
                    r.remove<DeformedBounds>(object);
                    deformed_bounds_changed.insert(object);
                }
            }
        }
        UpdateWireframeTransforms(r, deformed_bounds_changed); // Needs updated WorldTransforms
        {
            // Batch WorldTransform writes: collect all (BufferIndex, WorldTransform) pairs,
            // sort by BufferIndex for cache-friendly access, then write via single GetMutableRange.
            const auto &wt_reactive = reactive<changes::WorldTransform>(r);
            std::vector<std::pair<uint32_t, WorldTransform>> wt_writes;
            wt_writes.reserve(wt_reactive.size() + sync.NewlyInserted.size());

            const auto collect_wt = [&](entt::entity e) {
                if (!r.valid(e)) return;

                const auto *ri = r.try_get<const RenderInstance>(e);
                if (!ri || ri->BufferIndex == UINT32_MAX) return;

                const auto *wt = r.try_get<const WorldTransform>(e);
                if (!wt) return;

                auto display_wt = *wt;
                if (const auto *ds = r.try_get<BoneDisplayScale>(e)) display_wt.S = vec3{ds->Value};
                wt_writes.emplace_back(ri->BufferIndex, display_wt);
                // Bone joint sphere transforms (head and tail).
                if (const auto *joints = r.try_get<const BoneJointEntities>(e); joints && r.all_of<BoneDisplayScale>(e)) {
                    const float bone_length = r.get<BoneDisplayScale>(e).Value;
                    const float sphere_scale = bone_length * 0.06f;
                    if (joints->Head != entt::null) {
                        if (const auto *jri = r.try_get<const RenderInstance>(joints->Head)) {
                            wt_writes.emplace_back(jri->BufferIndex, Transform{wt->P, {1, 0, 0, 0}, vec3{sphere_scale}});
                        }
                    }
                    if (joints->Tail != entt::null) {
                        if (const auto *jri = r.try_get<const RenderInstance>(joints->Tail)) {
                            const vec3 tail_pos = wt->P + wt->R * vec3{0, bone_length, 0};
                            wt_writes.emplace_back(jri->BufferIndex, Transform{tail_pos, {1, 0, 0, 0}, vec3{sphere_scale}});
                        }
                    }
                }
            };
            for (auto e : wt_reactive) collect_wt(e);
            // Newly inserted entities whose WorldTransform wasn't already in the reactive set
            // (e.g., visibility toggled on without a Transform change).
            for (auto e : sync.NewlyInserted) {
                if (!wt_reactive.contains(e)) collect_wt(e);
            }
            if (FlushIndexedWrites(wt_writes, [&] { return buffers.Instances.GetMutableTransforms(); })) request(RenderRequest::Submit);
        }
    }
    { // Sync RotationUiVariant from Transform. Must run after bone block.
        for (auto e : reactive<changes::Rotation>(r)) {
            if (!r.all_of<Transform>(e)) continue;
            if (r.all_of<RotationUiDriving>(e)) {
                r.remove<RotationUiDriving>(e);
                continue;
            }
            const auto v = r.get<const Transform>(e).R;
            if (auto *ui = r.try_get<RotationUiVariant>(e)) *ui = ToUiVariant(v, ui->index());
            else r.emplace<RotationUiVariant>(e, RotationQuat{v});
        }
    }
    // If looking through a camera and it moved (animation or manual edit), snap the ViewCamera.
    // Must run before the SceneView handler so the ViewCamera replacement is picked up.
    if (const auto camera = LookThroughCameraEntity(r); camera != entt::null &&
        reactive<changes::WorldTransform>(r).contains(camera)) {
        const auto &wt = r.get<WorldTransform>(camera);
        r.replace<ViewCamera>(viewport, ViewCamera{wt.P, wt.R, r.get<Camera>(camera)});
    }
    { // Keep targeted PBR specialization mask in sync when one of its inputs changes.
        // Run before the UBO update below so Transmission pipeline is settled when the UBO reads it.
        const auto shading = r.get<const ViewportDisplay>(viewport).ViewportShading;
        if (!reactive<changes::ViewportDisplay>(r).empty() || !reactive<changes::PbrSpecialization>(r).empty()) {
            // SubmitViewport's full descriptor block runs only on resize, so write the transmission sampler
            // descriptor inline whenever EnsureTransmissionResources flips state.
            const auto refresh_transmission_descriptor = [&] {
                const auto info = pipelines.Main.TransmissionSamplerInfo();
                vk.Device.updateDescriptorSets({slots.MakeSamplerWrite(r.ctx().get<const SelectionSlots>().TransmissionSampler, info)}, {});
                request(RenderRequest::ReRecord);
            };
            if (shading == ViewportShadingMode::MaterialPreview || shading == ViewportShadingMode::Rendered) {
                PbrFeatureMask pbr_mask{0};
                const auto &active_lighting = GetActivePbrLighting(r, viewport, shading);
                if (active_lighting.UseSceneLights) pbr_mask |= PbrFeature::Punctual;
                for (const auto [_, feat] : r.view<const PbrMeshFeatures>().each()) pbr_mask |= feat.Mask;
                if (pipelines.Main.Compiler.CompilePipelines(pbr_mask)) request(RenderRequest::ReRecord);
                const bool want_transmission = active_lighting.RealTransmission && HasFeature(pbr_mask, PbrFeature::Transmission);
                const auto te_px = RenderExtentPx(r);
                if (pipelines.Main.EnsureTransmissionResources({te_px.x, te_px.y}, vk.Device, vk.PhysicalDevice, want_transmission)) refresh_transmission_descriptor();
            } else if (pipelines.Main.EnsureTransmissionResources({}, vk.Device, vk.PhysicalDevice, false)) {
                refresh_transmission_descriptor();
            }
        }
    }

    // Rebuild the view UBO (aspect, projection, look-through widening, ViewportSize).
    const auto render_extent = RenderExtentPx(r);
    if (!reactive<changes::SceneView>(r).empty() ||
        !reactive<changes::TransformPending>(r).empty() ||
        !reactive<changes::ViewportDisplay>(r).empty() ||
        !reactive<changes::InteractionMode>(r).empty() ||
        !reactive<changes::TransformEnd>(r).empty() ||
        light_count_changed ||
        resized) {
        const float aspect = render_extent.x == 0 || render_extent.y == 0 ? 1.f : float(render_extent.x) / float(render_extent.y);
        // When looking through a scene camera, keep the ViewCamera's widened FOV in sync
        // with the current viewport aspect ratio (handles viewport resize).
        if (const auto camera = LookThroughCameraEntity(r); camera != entt::null) {
            r.get<ViewCamera>(viewport).Data = WidenForLookThrough(r.get<Camera>(camera), aspect);
        }
        const auto &camera = r.get<const ViewCamera>(viewport);
        const auto &settings = r.get<const ViewportDisplay>(viewport);
        const bool is_pbr_mode = settings.ViewportShading == ViewportShadingMode::MaterialPreview || settings.ViewportShading == ViewportShadingMode::Rendered;
        const auto &active_lighting = GetActivePbrLighting(r, viewport, settings.ViewportShading);
        const bool use_scene_lights = is_pbr_mode && active_lighting.UseSceneLights;
        const bool use_scene_world = is_pbr_mode && active_lighting.UseSceneWorld;
        const auto &active_environment = use_scene_world ? environments.SceneWorld : environments.StudioWorld;
        const auto *source_assets = r.try_get<const gltf::SourceAssets>(viewport);
        const auto *source_ibl = source_assets && source_assets->ImageBasedLight.has_value() ? &*source_assets->ImageBasedLight : nullptr;
        const float env_intensity = use_scene_world && source_ibl ? source_ibl->Intensity : active_lighting.EnvIntensity;
        const mat3 env_rotation = [&]() -> mat3 {
            if (use_scene_world) return environments.SceneWorldRotation;
            const float radians = active_lighting.EnvRotationDegrees * (Pi / 180.f);
            const float s = std::sin(radians), c = std::cos(radians);
            return {c, 0, -s, 0, 1, 0, s, 0, c};
        }();
        const float background_blur = active_lighting.BackgroundBlur;
        const float world_opacity = is_pbr_mode ? active_lighting.WorldOpacity : 0.f;
        const auto *pending = r.try_get<const PendingTransform>(viewport);
        // ScreenPixelScale: world-space size per pixel at unit distance (perspective) or absolute (ortho).
        // Sign encodes camera type: positive = perspective (shader multiplies by distance), negative = orthographic.
        const float screen_pixel_scale = ScreenPixelScale(camera.Data, std::max(float(render_extent.y), 1.f));
        const auto proj = camera.Projection(aspect);
        buffers.SceneViewUBO.Update(as_bytes(SceneViewUBO{
            .ViewProj = proj * camera.View(),
            .ViewRotation = mat3(camera.View()),
            .CameraPosition = camera.Position(),
            .CameraNear = camera.NearClip(),
            .CameraFar = camera.FarClip(),
            .LightCount = buffers.Lights.Count(),
            .LightSlot = buffers.Lights.Slot(),
            .UseSceneLightsRender = use_scene_lights ? 1u : 0u,
            .EnvIntensity = env_intensity,
            .Exposure = std::exp2(active_lighting.ExposureEV),
            .EnvRotation = env_rotation,
            .BackgroundBlur = background_blur,
            .WorldOpacity = world_opacity,
            .Ibl = active_environment.Ibl,
            .InteractionMode = interaction_mode,
            .EditElement = r.get<const EditMode>(viewport).Value,
            .IsTransforming = pending ? 1u : 0u,
            .PendingPivot = pending ? pending->Pivot : vec3{},
            .PendingTranslation = pending ? pending->Delta.P : vec3{},
            .PendingRotation = pending ? pending->Delta.R : quat{1, 0, 0, 0},
            .PendingScale = pending ? pending->Delta.S : vec3{1},
            .ScreenPixelScale = screen_pixel_scale,
            .ViewportSize = render_extent,
            .FaceFirstTriSlot = meshes.GetFaceFirstTriangleSlot(),
            .BoneDeformSlot = meshes.GetBoneDeformSlot(),
            .ArmatureDeformSlot = buffers.ArmatureDeformBuffer.Buffer.Slot,
            .MorphDeformSlot = meshes.GetMorphTargetSlot(),
            .MorphWeightsSlot = buffers.MorphWeightBuffer.Buffer.Slot,
            .VertexClassSlot = buffers.VertexClassBuffer.Buffer.Slot,
            .MaterialSlot = buffers.Materials.Slot(),
            .PrimitiveMaterialSlot = meshes.GetPrimitiveMaterialSlot(),
            .FacePrimitiveSlot = meshes.GetFacePrimitiveSlot(),
            .DrawDataSlot = buffers.RenderDraw.DrawData.Slot,
            .VisibleIndexSlot = buffers.RenderDraw.VisibleIndices.Slot,
            .BoneXRay = settings.ViewportShading == ViewportShadingMode::Wireframe ? 1u : 0u,
            // Polygon offset factor matching Blender's GPU_polygon_offset_calc (viewdist = max ortho extent)
            .NdcOffsetFactor = std::holds_alternative<Perspective>(camera.Data) ? proj[3][2] * -0.00125f : 0.000005f * std::max(std::abs(1.f / proj[0][0]), std::abs(1.f / proj[1][1])),
            .TransmissionFramebufferSamplerSlot = r.ctx().get<const SelectionSlots>().TransmissionSampler,
            .TransmissionFramebufferMipCount = pipelines.Main.Transmission ? pipelines.Main.Transmission->MipCount : 1u,
            .UseRealTransmission = (is_pbr_mode && active_lighting.RealTransmission && pipelines.Main.Transmission) ? 1u : 0u,
            .DebugChannel = is_pbr_mode ? settings.DebugChannel : DebugChannel::None,
        }));
        r.ctx().get<DrawState>().SelectionStale = true;
        request(RenderRequest::Submit);
    }

    const auto &settings = r.get<const ViewportDisplay>(viewport);
    for (const auto mesh_entity : dirty_overlay_meshes) {
        const auto &mesh = GetMesh(r, mesh_entity);
        r.patch<MeshBuffers>(mesh_entity, [&](auto &mesh_buffers) {
            for (const auto element : NormalElements) {
                if (ElementMaskContains(settings.NormalOverlays, element)) {
                    if (!mesh_buffers.NormalIndicators.contains(element)) {
                        const auto index_kind = element == Element::Face ? IndexKind::Face : IndexKind::Vertex;
                        mesh_buffers.NormalIndicators.emplace(
                            element,
                            buffers.CreateRenderBuffers(CreateNormalVertices(mesh, element), CreateNormalIndices(mesh, element), index_kind)
                        );
                    } else {
                        buffers.VertexBuffer.Update(mesh_buffers.NormalIndicators.at(element).Vertices, CreateNormalVertices(mesh, element));
                    }
                } else if (mesh_buffers.NormalIndicators.contains(element)) {
                    buffers.Release(mesh_buffers.NormalIndicators.at(element));
                    mesh_buffers.NormalIndicators.erase(element);
                }
            }
        });
    }
    // Update mesh element state buffers (Excite mode only; Edit mode handled by GPU compute)
    for (const auto mesh_entity : dirty_element_state_meshes) {
        if (interaction_mode != InteractionMode::Excite) continue;
        const auto &mesh = GetMesh(r, mesh_entity);
        std::unordered_set<VH> selected_vertices;
        std::optional<uint32_t> active_handle, excited_handle;
        for (auto [entity, instance, excitable] : r.view<const Instance, const SoundVertices>().each()) {
            if (instance.Entity != mesh_entity) continue;
            selected_vertices.insert(excitable.Vertices.begin(), excitable.Vertices.end());
            if (const auto *force = r.try_get<const VertexForce>(entity)) excited_handle = force->Vertex;
            break;
        }
        if (const auto *active = r.try_get<const MeshActiveElement>(mesh_entity)) active_handle = active->Handle;
        meshes.UpdateElementStates(mesh, Element::Vertex, selected_vertices, {}, {}, {}, active_handle, excited_handle);
        r.ctx().get<DrawState>().SelectionStale = true;
    }
    if (!dirty_element_state_meshes.empty()) request(RenderRequest::Submit);
    if (r.all_of<ElementStatesDirty>(viewport)) {
        r.remove<ElementStatesDirty>(viewport);
        request(RenderRequest::Submit);
    }
    for (auto &&[id, storage] : r.storage()) {
        if (storage.info() == entt::type_id<entt::reactive>()) storage.clear();
    }
    destroy_tracker.Storage.clear();
    r.clear<MeshGeometryDirty, MeshMaterialAssignment, MaterialDirty, LightWireframeDirty>();

    buffers.Ctx.FlushDeferredDescriptorUpdates(vk.Device);
}

void RegisterSceneComponentHandlers(entt::registry &r) {
    // Reactive storage subscriptions for deferred once-per-frame processing
    track<changes::TimelineRange>(r).on<TimelineRange>(On::Update);
    track<changes::Selected>(r).on<Selected>(On::Create | On::Destroy);
    track<changes::ActiveInstance>(r).on<Active>(On::Create | On::Destroy);
    track<changes::BoneSelection>(r).on<BoneSelection>(On::Create | On::Update | On::Destroy).on<BoneActive>(On::Create | On::Destroy);
    track<changes::Rerecord>(r)
        .on<RenderInstance>(On::Create | On::Destroy)
        .on<Active>(On::Create | On::Destroy)
        .on<StartTransform>(On::Create | On::Destroy)
        .on<EditMode>(On::Create | On::Update)
        .on<SmoothShading>(On::Create | On::Destroy);
    track<changes::MeshActiveElement>(r).on<MeshActiveElement>(On::Create | On::Update);
    track<changes::MeshGeometry>(r).on<MeshGeometryDirty>(On::Create);
    track<changes::MeshMaterial>(r).on<MeshMaterialAssignment>(On::Create | On::Update);
    track<changes::SoundVertices>(r).on<SoundVertices>(On::Create | On::Destroy);
    track<changes::SoundVerticesUpdated>(r).on<SoundVertices>(On::Update);
    track<changes::VertexForce>(r).on<VertexForce>(On::Create | On::Destroy);
    track<changes::NewBufferEntity>(r).on<MeshBuffers>(On::Create);
    track<changes::ObjectCreated>(r).on<ObjectKind>(On::Create);
    track<changes::RenderInstanceCreated>(r).on<RenderInstance>(On::Create);
    track<changes::ViewportDisplay>(r).on<ViewportDisplay>(On::Create | On::Update);
    track<changes::InteractionMode>(r).on<Interaction>(On::Create | On::Update);
    track<changes::WorkspaceLights>(r).on<WorkspaceLights>(On::Create | On::Update);
    track<changes::ViewportTheme>(r).on<ViewportTheme>(On::Create | On::Update);
    track<changes::Materials>(r).on<MaterialDirty>(On::Create | On::Update);
    track<changes::MaterializedTextures>(r).on<MaterializedTextures>(On::Create | On::Update);
    track<changes::StudioEnvironment>(r).on<StudioEnvironment>(On::Create | On::Update);
    track<changes::SceneWorld>(r).on<gltf::SourceAssets>(On::Create | On::Update);
    track<changes::PunctualLight>(r).on<PunctualLight>(On::Create | On::Update);
    track<changes::ActiveMaterialVariant>(r).on<MaterialVariants>(On::Create | On::Update);
    track<changes::PbrSpecialization>(r)
        .on<PbrMeshFeatures>(On::Create | On::Update | On::Destroy)
        .on<MaterialPreviewLighting>(On::Create | On::Update)
        .on<RenderedLighting>(On::Create | On::Update);
    track<changes::SceneView>(r)
        .on<ViewCamera>(On::Create | On::Update)
        .on<MaterialPreviewLighting>(On::Create | On::Update)
        .on<RenderedLighting>(On::Create | On::Update)
        .on<LightIndex>(On::Create | On::Destroy)
        .on<EditMode>(On::Create | On::Update);
    track<changes::CameraLens>(r).on<Camera>(On::Create | On::Update).on<LookingThrough>(On::Create | On::Destroy);
    track<changes::Rotation>(r).on<Transform>(On::Create | On::Update);
    track<changes::WorldTransform>(r).on<WorldTransform>(On::Create | On::Update);
    track<changes::TransformPending>(r).on<PendingTransform>(On::Create | On::Update);
    track<changes::TransformEnd>(r).on<StartTransform>(On::Destroy);
    track<changes::TransformDirty>(r)
        .on<Transform>(On::Create | On::Update)
        .on<PosedLocal>(On::Create | On::Update)
        .on<SceneNode>(On::Create | On::Update)
        .on<BoneDisplayScale>(On::Update);
    track<changes::ActiveAnimationClip>(r)
        .on<ArmatureAnimation>(On::Update)
        .on<MorphWeightAnimation>(On::Update)
        .on<NodeTransformAnimation>(On::Update);
    r.ctx().emplace<EntityDestroyTracker>().Bind(r);

    // BoneConstraints edits change the resolved local Transform; poke it to drive the WorldTransform recompute.
    r.on_update<BoneConstraints>().connect<[](entt::registry &r, entt::entity e) {
        r.patch<Transform>(e, [](auto &) {});
    }>();

    RegisterSceneSetupHandler(r, [](entt::registry &r, entt::entity viewport) {
        r.emplace_or_replace<AudioOutputConfig>(viewport);
        r.emplace_or_replace<AudioOutputMix>(viewport);
        r.emplace_or_replace<Striker>(viewport);
        r.emplace_or_replace<ModalSoundControls>(viewport);
        r.emplace_or_replace<PlaybackFrame>(viewport);
        r.emplace_or_replace<LastEvaluatedFrame>(viewport);
        r.emplace_or_replace<AnimationTimelineView>(viewport);
        r.emplace_or_replace<TimelineRange>(viewport);
        r.emplace_or_replace<TimelinePlayback>(viewport);
        r.emplace_or_replace<SelectionXRay>(viewport);
        r.emplace_or_replace<BoxSelectState>(viewport);
        r.emplace_or_replace<GizmoInteraction>(viewport);
    });
    RegisterSceneClearHandler(r, [](entt::registry &r) {
        // Drop ColliderShapeBuffers' cached handles to the destroyed wireframe buffer entities, so
        // EnsureWireframes rebuilds them instead of mistaking a reused entity id for a live buffer.
        r.ctx().get<ColliderShapeBuffers>().Entities.fill(entt::entity{entt::null});
    });
}
