#include "ProcessEvents.h"

#include "AnimationTimeline.h"
#include "Apply.h"
#include "Armature.h"
#include "BBox.h"
#include "Bindless.h"
#include "Changes.h"
#include "Defaults.h"
#include "EntityDestroyTracker.h"
#include "ExtrasMesh.h"
#include "GpuBuffers.h"
#include "Instance.h"
#include "InteractionComponents.h"
#include "MeshComponents.h"
#include "NodeTransformAnimation.h"
#include "ObjectOps.h"
#include "Pipelines.h"
#include "Reactive.h"
#include "SceneGraph.h"
#include "Selection.h"
#include "SelectionComponents.h"
#include "SelectionGpu.h"
#include "SoundVertices.h"
#include "Textures.h"
#include "Timer.h"
#include "TransformMath.h"
#include "ViewportComponents.h"
#include "VulkanResources.h"
#include "gltf/GltfScene.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "physics/PhysicsDebugDraw.h"
#include "physics/PhysicsWorld.h"

#include "imgui.h"

#include "AxisColors.h" // Must be after imgui.h

#include <iostream>
#include <numbers>

using std::ranges::to;
using std::views::iota;

namespace {

using namespace he;

static inline const std::vector<Element> NormalElements{Element::Vertex, Element::Face};

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
    const auto &mesh = r.get<Mesh>(r.get<Instance>(instance_entity).Entity);
    const auto &wt = r.get<WorldTransform>(instance_entity);
    return {wt.P + glm::rotate(wt.R, wt.S * ComputeElementLocalPosition(mesh, element, handle))};
}

void RederiveCollider(entt::registry &r, entt::entity e) {
    const auto *cs = r.try_get<const ColliderShape>(e);
    const auto *policy = r.try_get<const ColliderPolicy>(e);
    if (!cs || !policy) return;
    const auto mesh_entity = cs->MeshEntity != null_entity ? cs->MeshEntity : FindMeshEntity(r, e);
    const auto *mesh = r.try_get<const Mesh>(mesh_entity);
    if (!mesh) return;

    const auto verts = mesh->GetVerticesSpan();
    const bool has_verts = !verts.empty();
    const auto bbox = [&] {
        BBox b;
        for (const auto &v : verts) {
            b.Min = glm::min(b.Min, v.Position);
            b.Max = glm::max(b.Max, v.Position);
        }
        return b;
    }();
    const vec3 bbox_center = has_verts ? (bbox.Min + bbox.Max) * 0.5f : vec3{0};
    const vec3 bbox_extents = has_verts ? (bbox.Max - bbox.Min) : vec3{0};

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
        // Tightest radius for an axis-aligned (Y-axis) cylinder/capsule centered at bbox_center.
        const auto xz_radius = [&] {
            const vec2 c{bbox_center.x, bbox_center.z};
            float r = 0;
            for (const auto &v : verts) r = glm::max(r, glm::length(vec2{v.Position.x, v.Position.z} - c));
            return r;
        };

        std::visit(
            overloaded{
                [&](physics::Box &s) {
                    s.Size = bbox_extents;
                    local_offset = bbox_center;
                },
                [&](physics::Sphere &s) {
                    const auto [c, radius] = ritter();
                    s.Radius = radius;
                    local_offset = c;
                },
                [&](physics::Cylinder &s) {
                    const float radius = xz_radius();
                    s.RadiusTop = s.RadiusBottom = radius;
                    s.Height = glm::max(physics::MinShapeHeight, bbox_extents.y);
                    local_offset = bbox_center;
                },
                [&](physics::Capsule &s) {
                    const float radius = xz_radius();
                    s.RadiusTop = s.RadiusBottom = radius;
                    // 2r ≥ bbox.y degenerates toward a sphere; clamp height to keep it spec-valid.
                    s.Height = glm::max(physics::MinShapeHeight, bbox_extents.y - 2.f * radius);
                    local_offset = bbox_center;
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

void SetEditMode(entt::registry &R, entt::entity viewport, Element mode) {
    const auto current_mode = R.get<const EditMode>(viewport).Value;
    if (current_mode == mode) return;

    auto &meshes = R.ctx().get<MeshStore>();
    auto &buffers = R.ctx().get<GpuBuffers>();
    auto *bits = buffers.SelectionBitset.Data();

    struct PendingConvert {
        entt::entity MeshEntity;
        uint32_t NewCount;
        std::vector<uint32_t> FromHandles;
    };
    std::vector<PendingConvert> pending;
    std::vector<ElementRange> old_ranges;
    uint32_t old_max_end = 0;
    for (auto [mesh_entity, br, mesh] : R.view<MeshSelectionBitsetRange, const Mesh>().each()) {
        const uint32_t old_count = br.Count, new_count = selection::GetElementCount(mesh, mode);
        auto from_handles = selection::ScanBitsetRange(bits, br.Offset, old_count);
        if (old_count > 0) old_ranges.emplace_back(mesh_entity, br.Offset, old_count);
        old_max_end = std::max(old_max_end, br.Offset + old_count);
        R.remove<MeshActiveElement>(mesh_entity);
        pending.emplace_back(PendingConvert{mesh_entity, new_count, std::move(from_handles)});
    }

    // Clear the superset of old and new packed bit ranges once, to avoid stale/overlap bits.
    uint32_t new_max_end = 0;
    for (const auto &p : pending) new_max_end += (p.NewCount + 31) / 32 * 32;
    const uint32_t clear_words = (std::max(old_max_end, new_max_end) + 31) / 32;
    if (clear_words > 0) memset(bits, 0, clear_words * sizeof(uint32_t));

    if (!old_ranges.empty()) {
        DispatchUpdateSelectionStates(R, viewport, old_ranges, current_mode);
        // Face mode also derives edge states via CPU; clear them when exiting face-select.
        if (current_mode == Element::Face) {
            for (const auto &p : pending) meshes.UpdateEdgeStatesFromFaces(R.get<const Mesh>(p.MeshEntity), {}, {});
        }
    }

    std::vector<ElementRange> new_ranges;
    uint32_t next_offset = 0;
    for (const auto &p : pending) {
        auto &br = R.get<MeshSelectionBitsetRange>(p.MeshEntity);
        br.Offset = next_offset;
        br.Count = p.NewCount;
        const auto &mesh = R.get<const Mesh>(p.MeshEntity);
        for (const uint32_t h : selection::ConvertSelectionElement(p.FromHandles, mesh, current_mode, mode)) {
            if (h >= p.NewCount) continue;
            const uint32_t gbit = next_offset + h;
            bits[gbit >> 5] |= 1u << (gbit & 31u);
        }
        if (p.NewCount > 0) new_ranges.emplace_back(p.MeshEntity, next_offset, p.NewCount);
        next_offset = (next_offset + p.NewCount + 31) / 32 * 32;
    }

    R.patch<EditMode>(viewport, [mode](auto &edit_mode) { edit_mode.Value = mode; });
    if (!new_ranges.empty()) ApplySelectionStateUpdate(R, viewport, new_ranges, mode);
    else if (!old_ranges.empty()) R.emplace_or_replace<ElementStatesDirty>(viewport);
}

struct SyncResult {
    std::vector<entt::entity> NewlyInserted; // Entities inserted into GPU buffers — callers must write their WorldTransform before submit.
    std::vector<entt::entity> NewMeshEntities; // Mesh entities needing deferred index buffer creation.
    std::vector<entt::entity> NewExtrasEntities; // Non-mesh buffer entities (extras/bone/joint) needing deferred index creation.
};
} // namespace

SyncResult SyncModelsBuffers(entt::registry &R) {
    auto &Buffers = R.ctx().get<GpuBuffers>();
    // Consume the new-buffer-entity tracker. Categorize by type for deferred index buffer creation.
    std::vector<entt::entity> new_mesh_entities, new_extras_entities;
    for (auto e : reactive<changes::NewBufferEntity>(R)) {
        if (!R.valid(e) || !R.all_of<MeshBuffers>(e)) continue;
        if (R.all_of<Mesh>(e)) new_mesh_entities.push_back(e);
        else if (R.all_of<ObjectExtrasTag>(e) || R.all_of<ArmatureObject>(e) || R.all_of<BoneJoint>(e))
            new_extras_entities.push_back(e);
    }

    // Hides — compact-erase removed instances within their entity's range in the shared InstanceArena.
    for (auto [buffer_entity, pending] : R.view<PendingHide>().each()) {
        // Sort descending so erasing from back-to-front avoids index shifting within the batch.
        auto &indices = pending.BufferIndices;
        std::sort(indices.begin(), indices.end(), std::greater<>());
        R.patch<ModelsBuffer>(buffer_entity, [&](auto &mb) {
            for (const auto global_idx : indices) {
                Buffers.Instances.CompactErase(global_idx, mb.InstanceRange.Offset + mb.InstanceCount);
                --mb.InstanceCount;
            }
        });
        // Fixup BufferIndex on remaining RenderInstances for this buffer entity.
        for (auto [other_entity, ri] : R.view<RenderInstance>().each()) {
            if (ri.Entity != buffer_entity || ri.BufferIndex == UINT32_MAX) continue;
            // Count how many erased indices were below this instance's index.
            uint32_t shift = 0;
            for (const auto erased_idx : indices) {
                if (erased_idx < ri.BufferIndex) ++shift;
            }
            if (shift > 0) R.patch<RenderInstance>(other_entity, [shift](auto &r) { r.BufferIndex -= shift; });
        }
        R.remove<PendingHide>(buffer_entity);
    }

    // Shows — batch-insert new instances into GPU buffers. Creates ModelsBuffer on first use.
    // WorldTransform is NOT written here — only ObjectIds and InstanceStates.
    // Callers must ensure WorldTransform is written before submit (see newly_inserted return value).
    std::vector<entt::entity> newly_inserted;
    std::unordered_map<entt::entity, std::vector<entt::entity>> shows_by_buffer;
    for (auto entity : reactive<changes::RenderInstanceCreated>(R)) {
        if (!R.valid(entity) || !R.all_of<RenderInstance>(entity)) continue;
        const auto &ri = R.get<const RenderInstance>(entity);
        if (ri.BufferIndex != UINT32_MAX) continue; // already synced
        shows_by_buffer[ri.Entity].push_back(entity);
    }
    // Pre-reserve InstanceArena for all new instances to avoid per-Allocate growth checks.
    {
        uint32_t total_new_instances = 0;
        for (const auto &[_, entities] : shows_by_buffer) total_new_instances += entities.size();
        if (total_new_instances > 0) Buffers.Instances.ReserveAdditional(total_new_instances);
    }
    // Shared write buffers — reused across groups to avoid per-group heap allocations.
    std::vector<uint32_t> object_ids;
    std::vector<uint8_t> states;
    for (auto &[buffer_entity, entities] : shows_by_buffer) {
        const uint32_t n = entities.size();
        // Create ModelsBuffer on first show (deferred from MeshBuffers creation so we know the initial capacity).
        if (!R.all_of<ModelsBuffer>(buffer_entity)) {
            R.emplace<ModelsBuffer>(buffer_entity, ModelsBuffer{Buffers.Instances.Allocate(n), 0});
        }
        auto &mb = R.get<ModelsBuffer>(buffer_entity);
        // Grow the range if needed (e.g., adding instances to existing entity).
        const auto new_total = mb.InstanceCount + n;
        if (new_total > mb.InstanceRange.Count) {
            auto old_range = mb.InstanceRange;
            const auto new_capacity = std::max(mb.InstanceRange.Count * 2, new_total);
            mb.InstanceRange = Buffers.Instances.Allocate(new_capacity);
            Buffers.Instances.CopyInstances(old_range.Offset, mb.InstanceRange.Offset, mb.InstanceCount);
            // Fixup existing BufferIndex values for this entity.
            for (auto [other_entity, ri] : R.view<RenderInstance>().each()) {
                if (ri.Entity == buffer_entity && ri.BufferIndex != UINT32_MAX) {
                    ri.BufferIndex = mb.InstanceRange.Offset + (ri.BufferIndex - old_range.Offset);
                }
            }
            Buffers.Instances.Free(old_range);
        }
        // Build contiguous arrays for ObjectIds and InstanceStates (reusing shared vectors).
        object_ids.resize(n);
        states.resize(n);
        const auto base_index = mb.InstanceRange.Offset + mb.InstanceCount;
        for (uint32_t j = 0; j < n; ++j) {
            const auto instance_entity = entities[j];
            R.get<RenderInstance>(instance_entity).BufferIndex = base_index + j;
            object_ids[j] = R.get<const RenderInstance>(instance_entity).ObjectId;
            states[j] = InstanceStateBits(R, instance_entity);
        }
        // Write ObjectIds and InstanceStates at the new slots. WorldTransform slots are left unwritten —
        // the WorldTransform reactive pass writes them before submit.
        Buffers.Instances.ObjectIdBuffer.Update(as_bytes(object_ids), vk::DeviceSize(base_index) * sizeof(uint32_t));
        Buffers.Instances.StateBuffer.Update(as_bytes(states), vk::DeviceSize(base_index) * sizeof(uint8_t));
        mb.InstanceCount = new_total;
        // No R.patch needed — ProcessComponentEvents handles Submit explicitly.
        newly_inserted.insert(newly_inserted.end(), entities.begin(), entities.end());
    }
    return {std::move(newly_inserted), std::move(new_mesh_entities), std::move(new_extras_entities)};
}

void EnsureWireframes(entt::registry &R, entt::entity viewport) {
    auto &Buffers = R.ctx().get<GpuBuffers>();
    auto &Meshes = R.ctx().get<MeshStore>();
    auto &shape_buffers = R.ctx().get<ColliderShapeBuffers>();
    const auto &settings = R.get<const ViewportDisplay>(viewport);
    const bool show_bbox = settings.ShowBoundingBoxes;
    const bool show_tets = settings.ShowTetWireframe;
    if (R.view<const ColliderShape>().empty() && R.view<ColliderWireframe>().empty() &&
        R.view<BBoxWireframe>().empty() && R.view<TetWireframe>().empty() &&
        !show_bbox && !show_tets) return;

    using enum ColliderShapeBuffer;
    auto buf = [&](ColliderShapeBuffer kind) -> entt::entity & { return shape_buffers.Entities[uint8_t(kind)]; };

    // Lazily create canonical buffer entities (recreated if destroyed by last-instance cleanup).
    auto ensure_buffer = [&](ColliderShapeBuffer kind, auto generator) {
        if (R.valid(buf(kind))) return;
        auto mesh = generator();
        if (!mesh.Positions.empty()) buf(kind) = ::CreateExtrasBufferEntity(R, Meshes, Buffers, mesh.Positions, {}, mesh.EdgeIndices);
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
            const auto &cw = R.get<const ColliderWireframe>(e);
            for (uint8_t i = 0; i < cw.Count; ++i) {
                if (R.valid(cw.Instances[i])) Destroy(R, viewport, cw.Instances[i]);
            }
            R.remove<ColliderWireframe>(e);
        }
    };

    auto make_instance = [&](entt::entity buffer_entity, entt::entity parent) -> entt::entity {
        if (buffer_entity == entt::null) return entt::null;
        const auto inst = R.create();
        R.emplace<Instance>(inst, buffer_entity);
        R.emplace<Transform>(inst);
        R.emplace<WorldTransform>(inst);
        R.emplace<SubElementOf>(inst, parent);
        Show(R, inst);
        return inst;
    };

    // Drop wireframes whose backing buffer no longer matches the shape kind (e.g. Box → ConvexHull).
    // The creation loop below recreates them (or leaves none for non-wireframe kinds).
    std::vector<entt::entity> stale;
    for (auto [entity, cs, cw] : R.view<const ColliderShape, const ColliderWireframe>().each()) {
        if (cw.Count > 0 && R.valid(cw.Instances[0]) && R.get<const Instance>(cw.Instances[0]).Entity != primary_buffer(cs.Shape)) {
            stale.emplace_back(entity);
        }
    }
    drop_wireframes(stale);

    for (auto [entity, cs] : R.view<const ColliderShape>().each()) {
        if (R.all_of<ColliderWireframe>(entity)) continue;

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
        if (cw.Count > 0) R.emplace<ColliderWireframe>(entity, cw);
    }

    // Remove wireframe instances for colliders that no longer exist.
    std::vector<entt::entity> orphans;
    for (auto [entity, cw] : R.view<ColliderWireframe>().each()) {
        if (!R.all_of<ColliderShape>(entity)) orphans.push_back(entity);
    }
    drop_wireframes(orphans);

    std::vector<entt::entity> bbox_stale;
    for (auto [entity, bw] : R.view<BBoxWireframe>().each()) {
        if (!show_bbox || !R.all_of<Selected>(entity)) bbox_stale.push_back(entity);
    }
    for (auto e : bbox_stale) {
        if (auto &bw = R.get<BBoxWireframe>(e); R.valid(bw.Instance)) Destroy(R, viewport, bw.Instance);
        R.remove<BBoxWireframe>(e);
    }

    if (show_bbox) {
        for (auto entity : R.view<Selected>()) {
            if (R.all_of<BBoxWireframe>(entity)) continue;

            const auto *instance = R.try_get<const Instance>(entity);
            if (instance && R.all_of<Mesh>(instance->Entity)) R.emplace<BBoxWireframe>(entity, make_instance(buf(Box), entity));
        }
    }

    // Drop tet wireframes whose backing geometry no longer matches (toggle off, deselected,
    // TetMeshData removed, or point count differs after regeneration).
    std::vector<entt::entity> tet_stale;
    for (auto [entity, tw] : R.view<TetWireframe>().each()) {
        const auto *inst = R.try_get<const Instance>(entity);
        const auto *tm = inst ? R.try_get<const TetMeshData>(inst->Entity) : nullptr;
        if (!show_tets || !R.all_of<Selected>(entity) || !tm || tm->Positions.empty()) {
            tet_stale.push_back(entity);
            continue;
        }
        const auto *wi = R.try_get<const Instance>(tw.Instance);
        const auto *mb = wi ? R.try_get<const MeshBuffers>(wi->Entity) : nullptr;
        if (!mb || mb->Vertices.Count != tm->Positions.size()) tet_stale.push_back(entity);
    }
    for (auto e : tet_stale) {
        if (auto &tw = R.get<TetWireframe>(e); R.valid(tw.Instance)) Destroy(R, viewport, tw.Instance);
        R.remove<TetWireframe>(e);
    }

    if (show_tets) {
        for (auto entity : R.view<Selected>()) {
            if (R.all_of<TetWireframe>(entity)) continue;
            const auto *instance = R.try_get<const Instance>(entity);
            if (!instance) continue;

            const auto *tm = R.try_get<const TetMeshData>(instance->Entity);
            if (!tm || tm->Positions.empty()) continue;

            const auto tet_buf = ::CreateExtrasBufferEntity(R, Meshes, Buffers, tm->Positions, {}, tm->EdgeIndices);
            R.emplace<TetWireframe>(entity, make_instance(tet_buf, entity));
        }
    }
}

void UpdateWireframeTransforms(entt::registry &R) {
    if (R.view<ColliderWireframe>().empty() && R.view<BBoxWireframe>().empty() && R.view<TetWireframe>().empty()) return;

    const auto &wt_changed = reactive<changes::WorldTransform>(R);
    const auto &shape_changed = reactive<changes::PhysicsShape>(R);
    const auto &mesh_geom_changed = reactive<changes::MeshGeometry>(R);
    for (auto [entity, cs, cw] : R.view<const ColliderShape, const ColliderWireframe>().each()) {
        const auto *wt = R.try_get<const WorldTransform>(entity);
        if (!wt) continue;

        const bool parent_moved = wt_changed.contains(entity);
        const bool shape_resized = shape_changed.contains(entity);
        const bool newly_created = [&] {
            for (uint8_t i = 0; i < cw.Count; ++i) {
                if (R.valid(cw.Instances[i]) && wt_changed.contains(cw.Instances[i])) return true;
            }
            return false;
        }();
        if (!parent_moved && !shape_resized && !newly_created) continue;

        auto set_wt = [&](entt::entity inst, mat4 m) {
            if (!R.valid(inst)) return;
            R.replace<WorldTransform>(inst, ToTransform(m));
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

    // Recompute BBox when the mesh geometry changed, the parent moved, or the wireframe instance was just created this frame.
    for (auto [entity, bw] : R.view<const BBoxWireframe>().each()) {
        if (!R.valid(bw.Instance)) continue;

        const auto *wt = R.try_get<const WorldTransform>(entity);
        const auto *instance = R.try_get<const Instance>(entity);
        if (!wt || !instance) continue;

        const auto *mesh = R.try_get<const Mesh>(instance->Entity);
        if (!mesh) continue;

        const bool parent_moved = wt_changed.contains(entity);
        const bool mesh_changed = mesh_geom_changed.contains(instance->Entity);
        const bool newly_created = wt_changed.contains(bw.Instance);
        if (!parent_moved && !mesh_changed && !newly_created) continue;

        BBox aabb;
        for (const auto &v : mesh->GetVerticesSpan()) {
            aabb.Min = glm::min(aabb.Min, v.Position);
            aabb.Max = glm::max(aabb.Max, v.Position);
        }
        const auto size = aabb.Max - aabb.Min, center = (aabb.Min + aabb.Max) * 0.5f;
        R.replace<WorldTransform>(bw.Instance, ToTransform(ToMatrix(*wt) * glm::translate(mat4{1}, center) * glm::scale(mat4{1}, size)));
    }

    for (auto [entity, tw] : R.view<const TetWireframe>().each()) {
        if (!R.valid(tw.Instance)) continue;

        const auto *wt = R.try_get<const WorldTransform>(entity);
        if (wt && (wt_changed.contains(entity) || wt_changed.contains(tw.Instance))) R.replace<WorldTransform>(tw.Instance, *wt);
    }
}

RenderRequest ProcessComponentEvents(entt::registry &R, entt::entity viewport) {
    const auto &Vk = R.ctx().get<const VulkanResources>();
    const auto &one_shot = R.ctx().get<const OneShotGpu>();
    auto &Slots = R.ctx().get<DescriptorSlots>();
    auto &Buffers = R.ctx().get<GpuBuffers>();
    auto &Meshes = R.ctx().get<MeshStore>();
    auto &Textures = R.ctx().get<TextureStore>();
    auto &Environments = R.ctx().get<EnvironmentStore>();
    auto &Physics = R.ctx().get<PhysicsWorld>();
    auto &pipelines = R.ctx().get<Pipelines>();
    const bool profile = R.all_of<ProfileNextProcessComponentEvents>(viewport);
    if (profile) R.remove<ProfileNextProcessComponentEvents>(viewport);
    std::optional<Timer> timer;
    if (profile) timer.emplace("ProcessComponentEvents");

    auto render_request = RenderRequest::None;
    auto request = [&render_request](RenderRequest req) { render_request = std::max(render_request, req); };

    if (R.all_of<PendingShaderRecompile>(viewport)) {
        R.remove<PendingShaderRecompile>(viewport);
        pipelines.CompileShaders();
        request(RenderRequest::ReRecord);
    }

    if (auto *pending_tex = R.try_get<PendingTextureUploads>(viewport); pending_tex && !pending_tex->Items.empty()) {
        const auto *src = R.try_get<const gltf::SourceAssets>(viewport);
        static const std::vector<gltf::Image> empty_images;
        const auto &gltf_images = src ? src->Images : empty_images;
        auto batch = BeginTextureUploadBatch(Vk.Device, *one_shot.Pool, Buffers.Ctx);
        for (const auto &item : pending_tex->Items) {
            auto entry = MaterializeTextureEntry(Vk, batch, Slots, item, gltf_images);
            if (!entry) {
                std::cerr << std::format("Warning: Failed to materialize texture '{}': {}\n", item.Name, entry.error());
                ReleaseSamplerSlots(Slots, std::span{&item.SamplerSlot, 1});
                continue;
            }
            Textures.Textures.emplace_back(std::move(*entry));
        }
        SubmitTextureUploadBatch(batch, Vk.Queue, *one_shot.Fence, Vk.Device);
        R.remove<PendingTextureUploads>(viewport);
    }
    // PendingSceneWorldClear takes precedence: a non-EXT load arrived after a previous EXT-IBL load
    // and the import is now stale. Cancel any pending import (release its allocated slots) before reset.
    if (R.all_of<PendingSceneWorldClear>(viewport)) {
        auto &env = Environments;
        if (auto *imp = R.try_get<PendingEnvironmentImport>(viewport)) {
            ReleaseCubeSamplerSlot(Slots, imp->DiffuseCubeSlot);
            ReleaseCubeSamplerSlot(Slots, imp->SpecularCubeSlot);
            R.remove<PendingEnvironmentImport>(viewport);
        }
        if (env.ImportedSceneWorld) {
            ReleaseCubeSamplerSlot(Slots, env.ImportedSceneWorld->DiffuseEnv.SamplerSlot);
            ReleaseCubeSamplerSlot(Slots, env.ImportedSceneWorld->SpecularEnv.SamplerSlot);
            env.ImportedSceneWorld.reset();
        }
        env.SceneWorldRotation = mat3{1.f};
        env.SceneWorld = {.Ibl = MakeIblSamplers(env.EmptySceneWorld, env), .Name = env.EmptySceneWorld.Name};
        R.patch<RenderedLighting>(viewport, [](auto &l) { l.WorldOpacity = 0.f; });
        R.remove<PendingSceneWorldClear>(viewport);
    }
    if (auto *pending_env = R.try_get<PendingEnvironmentImport>(viewport)) {
        if (const auto *src = R.try_get<const gltf::SourceAssets>(viewport)) {
            auto batch = BeginTextureUploadBatch(Vk.Device, *one_shot.Pool, Buffers.Ctx);
            auto pre = MaterializeEnvironmentImport(Vk, batch, Slots, *pending_env, src->Images);
            SubmitTextureUploadBatch(batch, Vk.Queue, *one_shot.Fence, Vk.Device);
            if (pre) {
                auto &env = Environments;
                if (env.ImportedSceneWorld) {
                    ReleaseCubeSamplerSlot(Slots, env.ImportedSceneWorld->DiffuseEnv.SamplerSlot);
                    ReleaseCubeSamplerSlot(Slots, env.ImportedSceneWorld->SpecularEnv.SamplerSlot);
                }
                env.ImportedSceneWorld = std::move(*pre);
                env.SceneWorldRotation = glm::mat3_cast(pending_env->Source.Rotation);
                env.SceneWorld = {.Ibl = MakeIblSamplers(*env.ImportedSceneWorld, env), .Name = env.ImportedSceneWorld->Name};
                R.patch<RenderedLighting>(viewport, [](auto &l) { l.WorldOpacity = 1.f; });
            } else {
                std::cerr << std::format("Warning: Failed to materialize EXT_lights_image_based '{}': {}\n", pending_env->Source.Name, pre.error());
                ReleaseCubeSamplerSlot(Slots, pending_env->DiffuseCubeSlot);
                ReleaseCubeSamplerSlot(Slots, pending_env->SpecularCubeSlot);
            }
        }
        R.remove<PendingEnvironmentImport>(viewport);
    }
    // Drop encoded Bytes for external-URI images now that materialization has consumed them.
    // SourceAbsPath is the persistence — SaveGltf re-reads from there.
    if (!R.any_of<PendingTextureUploads, PendingEnvironmentImport>(viewport)) {
        if (auto *src_assets = R.try_get<gltf::SourceAssets>(viewport)) {
            for (auto &img : src_assets->Images) {
                if (!img.Uri.empty()) img.Bytes = {};
            }
        }
    }

    // Pending* handlers run before the reactive checks so their patches land in trackers in time.
    if (const auto *pending = R.try_get<const PendingSetStudioEnvironment>(viewport)) {
        const auto index = pending->Index;
        R.remove<PendingSetStudioEnvironment>(viewport);
        SetStudioEnvironment(R, index);
    }
    if (const auto *pending = R.try_get<const PendingSetEditMode>(viewport)) {
        const auto mode = pending->Mode;
        R.remove<PendingSetEditMode>(viewport);
        SetEditMode(R, viewport, mode);
    }
    if (auto *pending = R.try_get<PendingImportMesh>(viewport)) {
        auto path = std::move(pending->Path);
        auto info = std::move(pending->Info);
        R.remove<PendingImportMesh>(viewport);
        ImportMesh(R, path, std::move(info));
    }
    if (const auto *pending = R.try_get<const PendingEditElementClick>(viewport)) {
        const auto mouse_px = pending->MousePx;
        const bool toggle = pending->Toggle;
        R.remove<PendingEditElementClick>(viewport);

        const auto edit_mode = R.get<const EditMode>(viewport).Value;
        const auto ranges = GetBitsetRangesForSelected(R);
        auto *bits = Buffers.SelectionBitset.Data();
        if (!toggle) {
            for (const auto &range : ranges) {
                const uint32_t first_word = range.Offset / 32;
                const uint32_t last_word = (range.Offset + range.Count + 31) / 32;
                memset(&bits[first_word], 0, (last_word - first_word) * sizeof(uint32_t));
                R.remove<MeshActiveElement>(range.MeshEntity);
            }
        }
        const auto hit = RunElementPickFromRanges(R, viewport, ranges, edit_mode, mouse_px);
        if (hit) {
            const auto [mesh_entity, element_index] = *hit;
            const auto *current_active = R.try_get<MeshActiveElement>(mesh_entity);
            const bool is_active = current_active && current_active->Handle == element_index;
            if (const auto *br = R.try_get<const MeshSelectionBitsetRange>(mesh_entity)) {
                const uint32_t global_bit = br->Offset + element_index;
                const bool was_selected = (bits[global_bit >> 5] >> (global_bit & 31)) & 1;
                if (toggle && was_selected) bits[global_bit >> 5] &= ~(1u << (global_bit & 31));
                else bits[global_bit >> 5] |= 1u << (global_bit & 31);
            }
            if (toggle && is_active) R.remove<MeshActiveElement>(mesh_entity);
            else R.emplace_or_replace<MeshActiveElement>(mesh_entity, element_index);
        } else if (!toggle) {
            for (const auto &range : ranges) R.remove<MeshActiveElement>(range.MeshEntity);
        }
        if (!ranges.empty() && (!toggle || hit)) R.emplace_or_replace<SelectionBitsDirty>(viewport);
    }

    // Create/destroy wireframe overlay instances and their buffer entities before SyncModelsBuffers
    // consumes the RenderInstance/NewBufferEntity reactive events they fire.
    EnsureWireframes(R, viewport);

    auto sync = SyncModelsBuffers(R); // Runs first so BufferIndex is valid for all downstream code.
    if (!sync.NewlyInserted.empty()) request(RenderRequest::Submit);
    const std::unordered_set<entt::entity> newly_inserted_set(sync.NewlyInserted.begin(), sync.NewlyInserted.end());
    const auto is_newly_inserted = [&](entt::entity e) { return newly_inserted_set.contains(e); };

    // Deferred ArmatureDeform allocation for newly created ArmaturePoseState (GpuDeformRange.Count == 0).
    // Two-pass: count total joints, single Reserve, then per-armature Allocate + ComputeDeformMatrices.
    {
        uint32_t total_joints = 0;
        std::vector<entt::entity> pending_armatures; // armature data entities
        for (const auto [arm_obj_entity, arm_obj_comp] : R.view<const ArmatureObject>().each()) {
            auto *pose_state = R.try_get<ArmaturePoseState>(arm_obj_comp.Entity);
            if (!pose_state || pose_state->GpuDeformRange.Count > 0) continue;
            const auto *armature = R.try_get<const Armature>(arm_obj_comp.Entity);
            if (!armature || !armature->ImportedSkin) continue;
            total_joints += armature->ImportedSkin->OrderedJointNodeIndices.size();
            pending_armatures.push_back(arm_obj_comp.Entity);
        }
        Buffers.ArmatureDeformBuffer.ReserveAdditional(total_joints);
        for (const auto arm_data_entity : pending_armatures) {
            auto &pose_state = R.get<ArmaturePoseState>(arm_data_entity);
            const auto &armature = R.get<const Armature>(arm_data_entity);
            pose_state.GpuDeformRange = Buffers.ArmatureDeformBuffer.Allocate(armature.ImportedSkin->OrderedJointNodeIndices.size());
            for (uint32_t i = 0; i < armature.Bones.size() && i < pose_state.BonePoseWorld.size(); ++i) {
                pose_state.BonePoseWorld[i] = armature.Bones[i].RestWorld;
            }
            ComputeDeformMatrices(
                armature, pose_state.BonePoseWorld,
                armature.ImportedSkin->InverseBindMatrices,
                Buffers.ArmatureDeformBuffer.GetMutable(pose_state.GpuDeformRange)
            );
        }
        if (!pending_armatures.empty()) request(RenderRequest::ReRecord);
    }

    // Deferred MorphWeights allocation for newly created MorphWeightState (GpuWeightRange.Count == 0).
    // Two-pass: count total weights, single Reserve, then per-instance Allocate + copy.
    {
        uint32_t total_weights = 0;
        std::vector<entt::entity> pending_morphs;
        for (auto [entity, morph_state] : R.view<MorphWeightState>().each()) {
            if (morph_state.GpuWeightRange.Count > 0 || morph_state.Weights.empty()) continue;
            total_weights += morph_state.Weights.size();
            pending_morphs.push_back(entity);
        }
        Buffers.MorphWeightBuffer.ReserveAdditional(total_weights);
        for (const auto entity : pending_morphs) {
            auto &morph_state = R.get<MorphWeightState>(entity);
            morph_state.GpuWeightRange = Buffers.MorphWeightBuffer.Allocate(morph_state.Weights.size());
            auto gpu_weights = Buffers.MorphWeightBuffer.GetMutable(morph_state.GpuWeightRange);
            std::copy(morph_state.Weights.begin(), morph_state.Weights.end(), gpu_weights.begin());
        }
        if (!pending_morphs.empty()) request(RenderRequest::ReRecord);
    }

    // Deferred index buffer creation for new mesh entities.
    // Pass 1: count total indices using cached triangle counts (no topology iteration).
    // Pass 2: write directly into GPU-mapped memory (zero temporary allocations).
    if (!sync.NewMeshEntities.empty()) {
        uint32_t total_face = 0, total_edge = 0, total_vertex = 0;
        for (auto entity : sync.NewMeshEntities) {
            const auto &mesh = R.get<const Mesh>(entity);
            total_face += mesh.TriangleIndexCount();
            total_edge += mesh.EdgeCount() * 2;
            total_vertex += mesh.VertexCount();
        }
        Buffers.ReserveAdditionalIndices(total_face, total_edge, total_vertex);
        for (auto entity : sync.NewMeshEntities) {
            const auto &mesh = R.get<const Mesh>(entity);
            R.patch<MeshBuffers>(entity, [&](auto &mb) {
                if (const auto tri_idx_count = mesh.TriangleIndexCount(); tri_idx_count > 0) {
                    auto [sr, dest] = Buffers.AllocateIndices(tri_idx_count, IndexKind::Face);
                    mesh.WriteTriangleIndices(dest);
                    mb.FaceIndices = sr;
                }
                if (mesh.EdgeCount() > 0) {
                    auto [sr, dest] = Buffers.AllocateIndices(mesh.EdgeCount() * 2, IndexKind::Edge);
                    mesh.WriteEdgeIndices(dest);
                    mb.EdgeIndices = sr;
                }
                if (mesh.VertexCount() > 0) {
                    auto [sr, dest] = Buffers.AllocateIndices(mesh.VertexCount(), IndexKind::Vertex);
                    std::iota(dest.begin(), dest.end(), 0u);
                    mb.VertexIndices = sr;
                }
            });
        }
        request(RenderRequest::ReRecord);
    }

    // Deferred index buffer creation for new extras/bone/joint buffer entities.
    // Light buffer entities are skipped — LightWireframeDirty handles their full rebuild.
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
            if (R.all_of<ArmatureObject>(entity)) {
                total_face += bone_faces.size();
                total_edge += bone.AdjacencyIndices.size();
                total_vertex += bone_verts.size();
            } else if (R.all_of<BoneJoint>(entity)) {
                total_face += sphere_faces.size();
                total_edge += sphere.OutlineIndices.size();
                total_vertex += sphere_verts.size();
            } else if (const auto *pending = R.try_get<const PendingEdgeIndices>(entity)) {
                total_edge += pending->Indices.size();
            }
        }
        Buffers.ReserveAdditionalIndices(total_face, total_edge, total_vertex);
        for (auto entity : sync.NewExtrasEntities) {
            if (R.all_of<ArmatureObject>(entity)) {
                R.patch<MeshBuffers>(entity, [&](auto &mb) {
                    mb.FaceIndices = Buffers.CreateIndices(bone_faces, IndexKind::Face);
                    mb.VertexIndices = Buffers.CreateIndices(bone_verts, IndexKind::Vertex);
                });
                R.emplace_or_replace<BoneAdjacencyIndices>(entity, Buffers.CreateIndices(bone.AdjacencyIndices, IndexKind::Edge));
            } else if (R.all_of<BoneJoint>(entity)) {
                R.patch<MeshBuffers>(entity, [&](auto &mb) {
                    mb.FaceIndices = Buffers.CreateIndices(sphere_faces, IndexKind::Face);
                    mb.EdgeIndices = Buffers.CreateIndices(sphere.OutlineIndices, IndexKind::Edge);
                    mb.VertexIndices = Buffers.CreateIndices(sphere_verts, IndexKind::Vertex);
                });
            } else if (auto *pending = R.try_get<PendingEdgeIndices>(entity)) {
                R.patch<MeshBuffers>(entity, [&](auto &mb) {
                    mb.EdgeIndices = Buffers.CreateIndices(pending->Indices, IndexKind::Edge);
                });
                R.remove<PendingEdgeIndices>(entity);
            }
        }
        request(RenderRequest::ReRecord);
    }

    { // Sync deferred light slot offsets (needs valid InstanceArena slot and RenderInstance.BufferIndex from SyncModelsBuffers).
        std::vector<entt::entity> synced_lights;
        for (auto [entity, light, light_index, instance] : R.view<PunctualLight, LightIndex, Instance>().each()) {
            if (const auto *ri = R.try_get<const RenderInstance>(entity)) {
                light.TransformSlotOffset = {Buffers.Instances.TransformBuffer.Slot, ri->BufferIndex};
                Buffers.Lights.Set(light_index.Value, light);
                synced_lights.push_back(entity);
            }
        }
        if (!synced_lights.empty()) request(RenderRequest::Submit);
        for (auto e : synced_lights) R.remove<PunctualLight>(e);
    }

    // Batch-compact light buffer for destroyed lights (indices collected in Destroy()).
    if (auto *pending = R.try_get<PendingLightRemovals>(viewport); pending && !pending->Indices.empty()) {
        auto &indices = pending->Indices;
        std::sort(indices.begin(), indices.end(), std::greater<>());
        auto buffer_count = Buffers.Lights.Count();
        for (const auto remove_index : indices) {
            if (remove_index >= buffer_count) continue; // Light was never synced to GPU (created and destroyed same frame).
            --buffer_count;
            if (remove_index != buffer_count) {
                Buffers.Lights.Set(remove_index, Buffers.Lights.Get(buffer_count));
                for (auto [other_entity, other_light_index] : R.view<LightIndex>().each()) {
                    if (other_light_index.Value == buffer_count) {
                        R.replace<LightIndex>(other_entity, remove_index);
                        break;
                    }
                }
            }
        }
        Buffers.Lights.SetCount(buffer_count);
        R.remove<PendingLightRemovals>(viewport);
        request(RenderRequest::ReRecord);
    }

    if (const auto *handlers = R.ctx().find<std::vector<ComponentEventHandler>>()) {
        for (const auto &h : *handlers) h(R);
    }

    { // Note: Can mutate InteractionMode, so do this first before `changes::InteractionMode` handling below.
        const auto interaction_mode = R.get<const Interaction>(viewport).Mode;
        auto &enabled_modes = R.get<EnabledInteractionModes>(viewport).Value;
        if (R.storage<SoundVertices>().empty()) {
            if (interaction_mode == InteractionMode::Excite) SetInteractionMode(R, viewport, *enabled_modes.begin());
            enabled_modes.erase(InteractionMode::Excite);
        } else if (!reactive<changes::SoundVertices>(R).empty()) {
            enabled_modes.insert(InteractionMode::Excite);
            if (interaction_mode == InteractionMode::Excite) request(RenderRequest::ReRecord);
            else SetInteractionMode(R, viewport, InteractionMode::Excite);
        }
    }
    std::unordered_set<entt::entity> dirty_overlay_meshes, dirty_element_state_meshes;

    { // Selected/Active instance changes - batch instance state buffer writes per buffer entity.
        auto &selected_tracker = reactive<changes::Selected>(R);
        auto &active_tracker = reactive<changes::ActiveInstance>(R);
        if (!selected_tracker.empty()) {
            // In Edit mode, selection changes primary_edit_instances which affects fill/edge/point batches.
            // In Object/Pose mode, only the silhouette batch is affected.
            const auto mode = R.get<const Interaction>(viewport).Mode;
            request(mode == InteractionMode::Edit ? RenderRequest::ReRecord : RenderRequest::ReRecordSilhouette);
        }

        // Collect instance state writes, then batch via GetMutableRange.
        // SyncModelsBuffers writes the full initial state for newly inserted instances,
        // so we skip those here to avoid redundant writes.
        struct StateWrite {
            uint32_t index;
            uint8_t state;
            entt::entity buffer_entity;
        };
        std::vector<StateWrite> state_writes;
        const auto collect_instance_state = [&](entt::entity instance_entity) {
            if (is_newly_inserted(instance_entity)) return;
            if (const auto *ri = R.try_get<RenderInstance>(instance_entity); ri && ri->BufferIndex != UINT32_MAX) {
                state_writes.push_back({ri->BufferIndex, InstanceStateBits(R, instance_entity), ri->Entity});
            }
        };
        // Update SelectedInstanceCount on mesh entities before per-entity processing.
        for (auto instance_entity : selected_tracker) {
            if (const auto *instance = R.try_get<Instance>(instance_entity); instance && R.all_of<Mesh>(instance->Entity)) {
                auto &sc = R.get_or_emplace<SelectedInstanceCount>(instance->Entity);
                sc.Value += R.all_of<Selected>(instance_entity) ? 1 : -1;
            }
        }
        for (auto instance_entity : selected_tracker) {
            collect_instance_state(instance_entity);
            if (const auto arm = FindArmatureObject(R, instance_entity); arm != entt::null) R.emplace_or_replace<BoneInstanceStateDirty>(arm);
            if (const auto *instance = R.try_get<Instance>(instance_entity); instance && R.all_of<Mesh>(instance->Entity)) {
                const auto mesh_entity = instance->Entity;
                if (R.all_of<Selected>(instance_entity)) {
                    dirty_overlay_meshes.insert(mesh_entity);
                } else if (!R.get_or_emplace<SelectedInstanceCount>(mesh_entity).Value) {
                    // Clean up overlays for this mesh
                    if (auto *buffers = R.try_get<MeshBuffers>(mesh_entity)) {
                        for (auto &[_, rb] : buffers->NormalIndicators) Buffers.Release(rb);
                        buffers->NormalIndicators.clear();
                    }
                }
            }
        }
        for (auto instance_entity : active_tracker) {
            collect_instance_state(instance_entity);
            if (const auto arm = FindArmatureObject(R, instance_entity); arm != entt::null) R.emplace_or_replace<BoneInstanceStateDirty>(arm);
        }

        // Batch-write all collected instance state changes.
        if (!state_writes.empty()) {
            std::sort(state_writes.begin(), state_writes.end(), [](const auto &a, const auto &b) { return a.index < b.index; });
            auto states = Buffers.Instances.GetMutableStates();
            for (const auto &w : state_writes) {
                states[w.index] = w.state;
                R.patch<ModelsBuffer>(w.buffer_entity);
            }
        }
    }
    { // Bone selection changes — tag armature objects for GPU state sync.
        auto &bone_sel_tracker = reactive<changes::BoneSelection>(R);
        if (!bone_sel_tracker.empty()) {
            request(RenderRequest::ReRecordSilhouette);
            for (auto bone_entity : bone_sel_tracker) {
                if (const auto arm = FindArmatureObject(R, bone_entity); arm != entt::null) R.emplace_or_replace<BoneInstanceStateDirty>(arm);
            }
        }
    }
    auto &destroy_tracker = R.ctx().get<EntityDestroyTracker>();
    if (!reactive<changes::Rerecord>(R).empty() || !destroy_tracker.Storage.empty()) request(RenderRequest::ReRecord);

    const auto interaction_mode = R.get<const Interaction>(viewport).Mode;
    const bool is_edit_mode = interaction_mode == InteractionMode::Edit;

    const auto edit_transform_context = is_edit_mode ? EditTransformContext{selection::ComputePrimaryEditInstances(R, false)} : EditTransformContext{};
    const auto orbit_to_active = [&](entt::entity instance_entity, Element element, uint32_t handle) {
        if (!R.get<const OrbitToActive>(viewport).Value) return;
        const auto world_pos = ComputeElementWorldPosition(R, instance_entity, element, handle);
        R.patch<ViewCamera>(viewport, [&](auto &camera) {
            if (const auto dir = world_pos - camera.Target; glm::dot(dir, dir) >= 1e-6f) {
                camera.SetTargetDirection(glm::normalize(dir));
            }
        });
    };

    if (R.all_of<SelectionBitsDirty>(viewport)) {
        R.remove<SelectionBitsDirty>(viewport);
        if (is_edit_mode) ApplySelectionStateUpdate(R, viewport, GetBitsetRangesForSelected(R), R.get<const EditMode>(viewport).Value);
    }
    if (const auto &tracker = reactive<changes::MeshActiveElement>(R); !tracker.empty()) {
        const auto edit_mode = R.get<const EditMode>(viewport).Value;
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
    for (auto instance_entity : reactive<changes::VertexForce>(R)) {
        if (const auto *inst = R.try_get<Instance>(instance_entity)) dirty_element_state_meshes.insert(inst->Entity);
        if (const auto *ev = R.try_get<VertexForce>(instance_entity)) orbit_to_active(instance_entity, Element::Vertex, ev->Vertex);
    }
    for (auto instance_entity : reactive<changes::SoundVerticesUpdated>(R)) {
        if (const auto *inst = R.try_get<const Instance>(instance_entity)) dirty_element_state_meshes.insert(inst->Entity);
    }
    for (auto camera_entity : reactive<changes::CameraLens>(R)) {
        if (const auto *cd = R.try_get<Camera>(camera_entity)) {
            const bool look_through_view = R.all_of<LookingThrough>(camera_entity);
            const auto buffer_entity = R.get<Instance>(camera_entity).Entity;
            Meshes.SetPositions(R.get<const VertexStoreId>(buffer_entity).StoreId, BuildCameraFrustumMesh(*cd, look_through_view).Positions);
            R.emplace_or_replace<SubmitDirty>(buffer_entity);
            // If looking through this camera, trigger a ViewCamera update so the SceneView
            // handler re-derives the widened FOV from the updated camera.
            if (look_through_view) R.patch<ViewCamera>(viewport, [](auto &) {});
        }
    }
    { // Sync RotationUiVariant from Rotation, but skip entities where the UI is driving the change.
        for (auto e : reactive<changes::Rotation>(R)) {
            if (!R.all_of<Transform>(e)) continue;
            if (R.all_of<RotationUiDriving>(e)) {
                R.remove<RotationUiDriving>(e);
                continue;
            }
            const auto v = R.get<const Transform>(e).R;
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
    if (const uint32_t required_count = R.storage<LightIndex>().size();
        Buffers.Lights.Count() != required_count) {
        Buffers.Lights.SetCount(required_count);
        light_count_changed = true;
    }
    if (!reactive<changes::Submit>(R).empty() || light_count_changed) request(RenderRequest::Submit);
    for (auto light_entity : R.view<LightWireframeDirty, LightIndex, Instance>()) {
        const auto light = Buffers.Lights.Get(R.get<const LightIndex>(light_entity).Value);
        auto wireframe = BuildLightMesh(light);
        const auto buffer_entity = R.get<Instance>(light_entity).Entity;

        // Release old vertex classes (need old vertex count from MeshBuffers before releasing)
        if (const auto *old_vcr = R.try_get<VertexClass>(buffer_entity)) {
            Buffers.VertexClassBuffer.Release({old_vcr->Offset, R.get<const MeshBuffers>(buffer_entity).Vertices.Count});
            R.remove<VertexClass>(buffer_entity);
        }

        auto &vs = R.get<VertexStoreId>(buffer_entity);
        Meshes.Release(vs.StoreId);
        if (auto *mb = R.try_get<MeshBuffers>(buffer_entity)) Buffers.Release(*mb);
        R.erase<MeshBuffers>(buffer_entity);

        const auto [store_id, vertices] = Meshes.AllocateVertexBuffer(wireframe.Data.Positions, {});
        vs.StoreId = store_id;

        R.emplace<MeshBuffers>(
            buffer_entity, Meshes.GetVerticesRange(store_id),
            SlottedRange{},
            Buffers.CreateIndices(wireframe.Data.CreateEdgeIndices(), IndexKind::Edge),
            SlottedRange{}
        );
        if (!wireframe.VertexClasses.empty()) {
            const auto range = Buffers.VertexClassBuffer.Allocate(std::span<const uint8_t>(wireframe.VertexClasses));
            R.emplace<VertexClass>(buffer_entity, range.Offset);
        }
        request(RenderRequest::ReRecord);
    }
    if (auto &tracker = reactive<changes::MeshGeometry>(R); !tracker.empty()) {
        const auto edit_mode = R.get<const EditMode>(viewport).Value;
        std::vector<ElementRange> geometry_ranges;
        for (auto mesh_entity : tracker) {
            if (R.get_or_emplace<SelectedInstanceCount>(mesh_entity).Value > 0) dirty_overlay_meshes.insert(mesh_entity);
            if (auto *br = R.try_get<MeshSelectionBitsetRange>(mesh_entity); br && edit_mode != Element::None) {
                // Topology changed: zero stale selection bits and update count.
                const auto &mesh = R.get<const Mesh>(mesh_entity);
                const uint32_t new_count = selection::GetElementCount(mesh, edit_mode);
                const uint32_t max_words = (std::max(br->Count, new_count) + 31) / 32;
                memset(&Buffers.SelectionBitset.Data()[br->Offset / 32], 0, max_words * sizeof(uint32_t));
                br->Count = new_count;
                if (new_count > 0) geometry_ranges.emplace_back(mesh_entity, br->Offset, br->Count);
            }
        }
        if (!geometry_ranges.empty()) ApplySelectionStateUpdate(R, viewport, geometry_ranges, edit_mode);
        request(RenderRequest::Submit);
    }
    if (auto &tracker = reactive<changes::MeshMaterial>(R); !tracker.empty()) {
        for (auto mesh_entity : tracker) {
            const auto *assignment = R.try_get<const MeshMaterialAssignment>(mesh_entity);
            const auto *mesh = R.try_get<const Mesh>(mesh_entity);
            if (!assignment || !mesh) continue;
            const auto material_count = Buffers.Materials.Count();
            if (material_count == 0u) continue;
            auto primitive_materials = Meshes.GetPrimitiveMaterialIndices(mesh->GetStoreId());
            if (assignment->PrimitiveIndex >= primitive_materials.size()) continue;
            primitive_materials[assignment->PrimitiveIndex] = std::min(assignment->MaterialIndex, material_count - 1u);
        }
        request(RenderRequest::Submit);
    }
    if (!reactive<changes::ModelsBuffer>(R).empty()) request(RenderRequest::Submit);
    if (!reactive<changes::ViewportTheme>(R).empty()) {
        UpdateDerivedColors(R.get<ViewportTheme>(viewport));
        R.get<colors::AxesArray>(viewport) = colors::MakeAxes(R.get<const ViewportTheme>(viewport).AxisColors);
        auto theme = R.get<const ViewportTheme>(viewport);
        theme.EdgeWidth *= ImGui::GetIO().DisplayFramebufferScale.x;
        Buffers.ViewportThemeUBO.Update(as_bytes(theme));
        request(RenderRequest::Submit);
    }
    if (!reactive<changes::Materials>(R).empty()) {
        if (const auto *dirty = R.try_get<const MaterialDirty>(viewport);
            dirty && dirty->Index < Buffers.Materials.Count()) {
            Buffers.Materials.Set(dirty->Index, Buffers.Materials.Get(dirty->Index));
        }
        request(RenderRequest::Submit);
    }
    if (!reactive<changes::ActiveMaterialVariant>(R).empty()) {
        const auto *mv = R.try_get<const MaterialVariants>(viewport);
        const auto active = mv ? mv->Active : std::nullopt;
        for (const auto [_, layout, mesh] : R.view<const MeshSourceLayout, const Mesh>().each()) {
            auto primitive_materials = Meshes.GetPrimitiveMaterialIndices(mesh.GetStoreId());
            for (std::size_t i = 0; i < layout.DefaultMaterials.size(); ++i) {
                const auto &mapping = layout.VariantMappings[i];
                primitive_materials[i] = active && *active < mapping.size() && mapping[*active] ?
                    *mapping[*active] :
                    layout.DefaultMaterials[i];
            }
        }
        request(RenderRequest::Submit);
    }
    if (!reactive<changes::ViewportDisplay>(R).empty()) {
        request(RenderRequest::ReRecord);
        dirty_overlay_meshes.merge(selection::GetSelectedMeshEntities(R));
    }
    if (!reactive<changes::InteractionMode>(R).empty()) {
        request(RenderRequest::ReRecord);
        // Dispatch UpdateSelectionState for all meshes entering Edit mode (MeshSelectionBitsetRange assigned in SetInteractionMode).
        if (R.get<const Interaction>(viewport).Mode == InteractionMode::Edit) {
            if (const auto edit_mode = R.get<const EditMode>(viewport).Value; edit_mode != Element::None) ApplySelectionStateUpdate(R, viewport, GetBitsetRangesForSelected(R), edit_mode);
        }
        for (const auto [_, instance, __] : R.view<const Instance, const SoundVertices>().each()) {
            dirty_element_state_meshes.insert(instance.Entity);
        }
        // Mark all armatures dirty for bone state + pose sync on mode change.
        for (const auto arm : R.view<ArmatureObject>()) R.emplace_or_replace<BoneInstanceStateDirty>(arm);
    }
    // Handle mesh Edit mode transform commit when StartTransform is cleared.
    // Bone Edit mode commits are handled in the bone pose transform section below.
    if (!reactive<changes::TransformEnd>(R).empty()) {
        if (is_edit_mode && FindArmatureObject(R, FindActiveEntity(R)) == entt::null) {
            if (const auto &pending = R.get<const PendingTransform>(viewport); pending.Delta != Transform{}) {
                // Apply edit transform once per selected mesh via a representative selected instance.
                // This keeps linked instances from receiving duplicate per-instance edits.
                for (const auto &[mesh_entity, instance_entity] : edit_transform_context.TransformInstances) {
                    if (selection::HasScaleLockedInstance(R, mesh_entity)) continue;
                    const auto &mesh = R.get<const Mesh>(mesh_entity);
                    const auto vertex_states = Meshes.GetVertexStates(mesh.GetStoreId());
                    const auto vertices = mesh.GetVerticesSpan();
                    const auto &wt = R.get<const WorldTransform>(instance_entity);
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
                        if (glm::length2(new_local - local_pos) <= 1e-12f) continue;
                        Meshes.SetPosition(mesh, vi, new_local);
                        any_moved = true;
                    }
                    if (any_moved) {
                        Meshes.UpdateNormals(mesh);
                        dirty_overlay_meshes.insert(mesh_entity);
                        R.remove<PrimitiveShape>(mesh_entity);
                        R.emplace_or_replace<MeshGeometryDirty>(mesh_entity);
                    }
                }
            }
            R.remove<PendingTransform>(viewport);
        }
    }

    { // Rederive colliders
        std::unordered_set<entt::entity> to_rederive;
        for (auto e : reactive<changes::ColliderPolicy>(R)) to_rederive.insert(e);
        if (const auto &mesh_dirty = reactive<changes::MeshGeometry>(R); !mesh_dirty.empty()) {
            for (auto [ce, cs] : R.view<const ColliderShape>().each()) {
                const auto me = cs.MeshEntity != null_entity ? cs.MeshEntity : FindMeshEntity(R, ce);
                if (mesh_dirty.contains(me)) to_rederive.insert(ce);
            }
        }
        for (auto e : to_rederive) RederiveCollider(R, e);
    }

    const bool mode_changed = !reactive<changes::InteractionMode>(R).empty();
    bool anim_advanced;
    { // Animation timeline tick
        const auto &range = R.get<const TimelineRange>(viewport);
        auto &playback = R.get<TimelinePlayback>(viewport);
        auto &pf = R.get<PlaybackFrame>(viewport).Value;
        if (playback.Playing) {
            pf += ImGui::GetIO().DeltaTime * range.Fps;
            if (pf > float(range.EndFrame)) pf = float(range.StartFrame);
            const int new_frame = int(std::floor(pf));
            if (new_frame != playback.CurrentFrame) R.patch<TimelinePlayback>(viewport, [&](auto &p) { p.CurrentFrame = new_frame; });
        } else {
            pf = float(playback.CurrentFrame);
        }
        anim_advanced = playback.CurrentFrame != R.get<LastEvaluatedFrame>(viewport).Value;

        if (!reactive<changes::PhysicsSimulationSettings>(R).empty()) {
            Physics.ApplySimulationSettings(R.get<const PhysicsSimulationSettings>(viewport));
        }
        // Range edits invalidate the cache: bake frontier becomes meaningless when bounds shift.
        const bool range_changed = !reactive<changes::TimelineRange>(R).empty();
        if (R.all_of<PhysicsCacheInvalid>(viewport)) {
            if (Physics.HasBodies()) Physics.ClearCache();
            R.remove<PhysicsCacheInvalid>(viewport);
        } else if (range_changed && Physics.HasBodies()) {
            Physics.ClearCache();
        }
        // The cache is the timeline: bake past the frontier, restore within it.
        if (Physics.HasBodies()) {
            Physics.EnsureCacheRange(range.StartFrame, range.EndFrame);
            // Any physics-mutating change invalidates cached future frames so the next play-forward re-bakes.
            if (!reactive<changes::PhysicsShape>(R).empty() ||
                !reactive<changes::PhysicsMotion>(R).empty() ||
                !reactive<changes::PhysicsMaterial>(R).empty() ||
                !reactive<changes::PhysicsTrigger>(R).empty() ||
                !reactive<changes::PhysicsJoint>(R).empty() ||
                !reactive<changes::PhysicsMaterialDef>(R).empty() ||
                !reactive<changes::CollisionSystemDef>(R).empty() ||
                !reactive<changes::CollisionFilterDef>(R).empty() ||
                !reactive<changes::PhysicsJointDef>(R).empty() ||
                !reactive<changes::PhysicsSimulationSettings>(R).empty()) {
                Physics.InvalidateFromFrame(playback.CurrentFrame);
            }
            if (anim_advanced) {
                const int from = R.get<LastEvaluatedFrame>(viewport).Value, to = playback.CurrentFrame;
                const float dt = range.Fps > 0 ? 1.f / range.Fps : 1.f / 60.f;
                const auto baked = Physics.BakedThrough();
                if (to == from + 1 && (!baked || uint32_t(to) > *baked)) {
                    Physics.BakeFrame(R, R.get<const PhysicsSimulationSettings>(viewport), to, dt);
                    request(RenderRequest::Submit);
                } else if (Physics.HasCachedFrame(to)) {
                    Physics.RestoreFrame(R, to);
                    request(RenderRequest::Submit);
                } else if (to == range.StartFrame) {
                    Physics.Rebuild(R); // Reset sim: covers wrap-after-play and user-initiated jump-to-start.
                    request(RenderRequest::Submit);
                }
                // Else: scrub past the bake frontier — hold current pose until play resumes baking.
            }
        }

        if (anim_advanced) R.get<LastEvaluatedFrame>(viewport).Value = playback.CurrentFrame;
        // Timeline frames are displayed 1-based, but animation time starts at t=0 on frame 1.
        const auto eval_seconds = float(std::max(0, playback.CurrentFrame - 1)) / range.Fps;
        const auto clip_time = [eval_seconds](const auto &clip) {
            return clip.DurationSeconds > 0 ? std::fmod(eval_seconds, clip.DurationSeconds) : 0.f;
        };

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
                    EvaluateAnimationDeltas(clip, clip_time(clip), armature.Bones, pose_state->BonePoseDelta);
                }
            }
        }

        if (anim_advanced) {
            // Evaluate morph weight animations
            for (auto [entity, morph_anim, morph_state, instance] :
                 R.view<const MorphWeightAnimation, MorphWeightState, const Instance>().each()) {
                if (morph_anim.Clips.empty() || morph_anim.ActiveClipIndex >= morph_anim.Clips.size()) continue;
                const auto &clip = morph_anim.Clips[morph_anim.ActiveClipIndex];
                const auto &mesh = R.get<const Mesh>(instance.Entity);
                const auto default_weights = Meshes.GetDefaultMorphWeights(mesh.GetStoreId());
                std::copy(default_weights.begin(), default_weights.end(), morph_state.Weights.begin());
                EvaluateMorphWeights(clip, clip_time(clip), morph_state.Weights);
                auto gpu_weights = Buffers.MorphWeightBuffer.GetMutable(morph_state.GpuWeightRange);
                std::copy(morph_state.Weights.begin(), morph_state.Weights.end(), gpu_weights.begin());
                request_rerecord = true;
            }
            // Evaluate glTF node transform animations (empties/meshes/cameras/lights).
            for (auto [entity, node_anim] : R.view<const NodeTransformAnimation>().each()) {
                if (node_anim.Clips.empty() || node_anim.ActiveClipIndex >= node_anim.Clips.size()) continue;
                const auto &clip = node_anim.Clips[node_anim.ActiveClipIndex];
                std::array<Transform, 1> local_pose{node_anim.RestLocal};
                EvaluateAnimation(clip, clip_time(clip), local_pose);
                R.replace<Transform>(entity, local_pose.front());
                request_rerecord = true;
            }
        }
        if (request_rerecord) request(RenderRequest::ReRecord);
    }
    { // Bones
        // GPU instance state
        const bool is_object_mode = interaction_mode == InteractionMode::Object;
        for (const auto arm_obj_entity : R.view<BoneInstanceStateDirty>()) {
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
            auto compute_state = [&](entt::entity b, BoneSel part) {
                if (is_object_mode) return max_state;
                const auto *parts = R.try_get<const BoneSelection>(b);
                const bool selected = is_edit ?
                    parts && (part == BoneSel::Body ? parts->Body : part == BoneSel::Root ? parts->Root :
                                                                                            parts->Tip) :
                    R.all_of<BoneSelection>(b);
                uint8_t s = R.all_of<BoneActive>(b) ? ElementStateActive : 0;
                if (selected) s |= ElementStateSelected;
                return s;
            };

            for (const auto b : bone_entities) {
                if (const auto *ri = R.try_get<RenderInstance>(b)) {
                    const auto state = compute_state(b, BoneSel::Body);
                    Buffers.Instances.UpdateState(ri->BufferIndex, state);
                }
            }
            R.patch<ModelsBuffer>(arm_obj_entity);
            if (arm_obj.JointEntity != entt::null && R.valid(arm_obj.JointEntity)) {
                for (const auto b : bone_entities) {
                    const auto *joints = R.try_get<const BoneJointEntities>(b);
                    if (!joints) continue;
                    for (const auto &[je, part] : {std::pair{joints->Head, BoneSel::Root}, {joints->Tail, BoneSel::Tip}}) {
                        if (je != entt::null) {
                            if (const auto *ri = R.try_get<const RenderInstance>(je)) {
                                const auto state = compute_state(b, part);
                                Buffers.Instances.UpdateState(ri->BufferIndex, state);
                            }
                        }
                    }
                }
                R.patch<ModelsBuffer>(arm_obj.JointEntity);
            }
        }
        R.clear<BoneInstanceStateDirty>();

        // Bone pose sync: classify bone changes, update pose deltas, compute deform matrices.
        // Runs before the reactive WorldTransform pass so bone sync's Transform patches are included.
        const bool bones_need_refresh = anim_advanced || mode_changed;
        if (bones_need_refresh || !reactive<changes::TransformDirty>(R).empty() || !reactive<changes::TransformEnd>(R).empty()) {
            const auto &local_changes = reactive<changes::TransformDirty>(R);
            const auto &transform_end = reactive<changes::TransformEnd>(R);
            for (const auto [arm_obj_entity, arm_obj_comp] : R.view<const ArmatureObject>().each()) {
                auto *pose_state = R.try_get<ArmaturePoseState>(arm_obj_comp.Entity);
                if (!pose_state) continue;
                auto &armature = R.get<Armature>(arm_obj_comp.Entity);
                if (!armature.ImportedSkin) continue;

                // Constraints can depend on external targets (e.g. physics bodies), so we can't early-out on bone-dirty alone.
                const bool has_any_constraint = std::any_of(
                    arm_obj_comp.BoneEntities.begin(), arm_obj_comp.BoneEntities.end(),
                    [&](auto e) { return R.all_of<BoneConstraints>(e); }
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

                const mat4 armature_world_inv = has_any_constraint ? glm::inverse(ToMatrix(R.get<const WorldTransform>(arm_obj_entity))) : I4;

                bool need_sync = has_any_constraint;
                bool rest_pose_edited = false;
                for (uint32_t i = 0; i < arm_obj_comp.BoneEntities.size(); ++i) {
                    const auto b = arm_obj_comp.BoneEntities[i];
                    if (i >= pose_state->BonePoseDelta.size()) continue;
                    const auto &rest = armature.Bones[i].RestLocal;
                    const auto &bt = R.get<const Transform>(b);
                    Transform local{bt.P, bt.R, rest.S}; // Default; branches that recompute overwrite it.
                    bool should_patch = false;
                    if (is_edit_mode) {
                        if (mode_changed) {
                            // Entering Edit mode: snap to rest pose.
                            local = {rest.P, rest.R, rest.S};
                            should_patch = true;
                            need_sync = true;
                        } else if (transform_end.contains(b) || (local_changes.contains(b) && !R.all_of<StartTransform>(b))) {
                            // Commit Edit mode transform (gizmo drag end or UI slider edit).
                            armature.Bones[i].RestLocal.P = bt.P;
                            armature.Bones[i].RestLocal.R = bt.R;
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
                        const Transform gizmo_local{bt.P, bt.R, rest.S};
                        pose_state->BoneUserOffset[i] = AbsoluteToDelta(grab_delta, AbsoluteToDelta(rest, gizmo_local));
                        local = ComposeWithDelta(rest, ComposeWithDelta(pose_state->BonePoseDelta[i], pose_state->BoneUserOffset[i]));
                        should_patch = true;
                        need_sync = true;
                    } else if (transform_end.contains(b)) {
                        // Commit drag: bake current P/R into delta, clear offset.
                        pose_state->BonePoseDelta[i] = AbsoluteToDelta(rest, {bt.P, bt.R, rest.S});
                        pose_state->BoneUserOffset[i] = {};
                        need_sync = true;
                    } else if (bones_need_refresh) {
                        // Animation advanced or leaving Edit mode: recompute entity P/R from deltas.
                        local = ComposeWithDelta(rest, ComposeWithDelta(pose_state->BonePoseDelta[i], pose_state->BoneUserOffset[i]));
                        should_patch = true;
                        need_sync = true;
                    } else if (local_changes.contains(b)) {
                        // Manual transform: bake if position actually changed.
                        if (const auto expected = ComposeWithDelta(rest, ComposeWithDelta(pose_state->BonePoseDelta[i], pose_state->BoneUserOffset[i]));
                            bt.P != expected.P || bt.R != expected.R) {
                            pose_state->BonePoseDelta[i] = AbsoluteToDelta(rest, {bt.P, bt.R, rest.S});
                            pose_state->BoneUserOffset[i] = {};
                            need_sync = true;
                        }
                    }

                    const uint32_t parent_idx = armature.Bones[i].ParentIndex;
                    const mat4 parent_pose_world = (parent_idx == InvalidBoneIndex) ? I4 : pose_state->BonePoseWorld[parent_idx];

                    // Apply bone constraints. Skipped in edit mode (rest-pose edits bypass pose constraints).
                    if (!is_edit_mode) {
                        if (const auto *cs = R.try_get<const BoneConstraints>(b); cs && !cs->Stack.empty()) {
                            const auto before = local;
                            for (const auto &c : cs->Stack) {
                                if (c.TargetEntity == null_entity || !R.valid(c.TargetEntity)) continue;
                                const auto *twt = R.try_get<const WorldTransform>(c.TargetEntity);
                                if (!twt) continue;
                                local = ApplyBoneConstraint(c, local, parent_pose_world, armature_world_inv, ToMatrix(*twt));
                            }
                            if (local.P != before.P || local.R != before.R) should_patch = true;
                        }
                    }

                    if (should_patch) R.patch<Transform>(b, [&](auto &t) { t.P = local.P; t.R = local.R; });
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
                        if (transform_end.contains(b) || (local_changes.contains(b) && !R.all_of<StartTransform>(b))) {
                            armature.Bones[i].RestWorld = parent_world * ToMatrix(armature.Bones[i].RestLocal);
                        } else {
                            // Preserve old world position, adjust RestLocal to compensate for parent change.
                            const mat4 new_local_mat = glm::inverse(parent_world) * armature.Bones[i].RestWorld;
                            armature.Bones[i].RestLocal.P = vec3(new_local_mat[3]);
                            armature.Bones[i].RestLocal.R = glm::normalize(glm::quat_cast(mat3(new_local_mat)));
                            R.patch<Transform>(b, [&](auto &t) { t.P = armature.Bones[i].RestLocal.P; t.R = armature.Bones[i].RestLocal.R; });
                        }
                        armature.Bones[i].InvRestWorld = glm::inverse(armature.Bones[i].RestWorld);
                    }
                    armature.RecomputeInverseBindMatrices();
                }
                if (need_sync) {
                    if (!is_edit_mode) {
                        ComputeDeformMatrices(
                            armature, pose_state->BonePoseWorld, armature.ImportedSkin->InverseBindMatrices,
                            Buffers.ArmatureDeformBuffer.GetMutable(pose_state->GpuDeformRange)
                        );
                    }
                    request(RenderRequest::ReRecord);
                }
            }
        }
        // Recompute WorldTransform for all entities with changed local transforms or parenting.
        // Runs after bone sync so bone Transform patches are included in a single pass.
        if (const auto &dirty = reactive<changes::TransformDirty>(R); !dirty.empty()) {
            const auto update_wt = [&](this const auto &self, entt::entity e, bool propagate) -> void {
                const auto *node = R.try_get<SceneNode>(e);
                const auto &t = R.get<const Transform>(e);
                node && node->Parent != entt::null ? R.emplace_or_replace<WorldTransform>(e, ToTransform(GetParentDelta(R, e) * ToMatrix(t))) : R.emplace_or_replace<WorldTransform>(e, t);
                if (propagate) {
                    for (const auto child : Children{&R, e}) self(child, true);
                }
            };
            const bool bone_edit = is_edit_mode && FindArmatureObject(R, FindActiveEntity(R)) != entt::null;
            for (auto e : dirty) {
                // Skip newly inserted entities — their WorldTransform is already correct from creation.
                if (R.valid(e) && !is_newly_inserted(e)) update_wt(e, !(bone_edit && R.all_of<StartTransform>(e)));
            }
        }
        UpdateWireframeTransforms(R); // Needs updated WorldTransforms

        // Batch WorldTransform writes: collect all (BufferIndex, WorldTransform) pairs,
        // sort by BufferIndex for cache-friendly access, then write via single GetMutableRange.
        {
            const auto &wt_reactive = reactive<changes::WorldTransform>(R);
            struct WtWrite {
                uint32_t index;
                WorldTransform wt;
            };
            std::vector<WtWrite> wt_writes;
            wt_writes.reserve(wt_reactive.size() + sync.NewlyInserted.size());

            const auto collect_wt = [&](entt::entity e) {
                if (!R.valid(e)) return;

                const auto *ri = R.try_get<const RenderInstance>(e);
                if (!ri || ri->BufferIndex == UINT32_MAX) return;

                const auto *wt = R.try_get<const WorldTransform>(e);
                if (!wt) return;

                auto display_wt = *wt;
                if (const auto *ds = R.try_get<BoneDisplayScale>(e)) display_wt.S = vec3{ds->Value};
                wt_writes.push_back({ri->BufferIndex, display_wt});
                // Bone joint sphere transforms (head and tail).
                if (const auto *joints = R.try_get<const BoneJointEntities>(e); joints && R.all_of<BoneDisplayScale>(e)) {
                    const float bone_length = R.get<BoneDisplayScale>(e).Value;
                    const float sphere_scale = bone_length * 0.06f;
                    if (joints->Head != entt::null) {
                        if (const auto *jri = R.try_get<const RenderInstance>(joints->Head)) {
                            wt_writes.push_back({jri->BufferIndex, Transform{wt->P, {1, 0, 0, 0}, vec3{sphere_scale}}});
                        }
                    }
                    if (joints->Tail != entt::null) {
                        if (const auto *jri = R.try_get<const RenderInstance>(joints->Tail)) {
                            const vec3 tail_pos = wt->P + wt->R * vec3{0, bone_length, 0};
                            wt_writes.push_back({jri->BufferIndex, Transform{tail_pos, {1, 0, 0, 0}, vec3{sphere_scale}}});
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
            if (!wt_writes.empty()) {
                std::sort(wt_writes.begin(), wt_writes.end(), [](const auto &a, const auto &b) { return a.index < b.index; });
                auto transforms = Buffers.Instances.GetMutableTransforms();
                for (const auto &w : wt_writes) transforms[w.index] = w.wt;
                request(RenderRequest::Submit);
            }
        }
    }
    // If looking through a camera and it moved (animation or manual edit), snap the ViewCamera.
    // Must run before the SceneView handler so the ViewCamera replacement is picked up.
    if (const auto camera = LookThroughCameraEntity(R); camera != entt::null &&
        reactive<changes::WorldTransform>(R).contains(camera)) {
        const auto &wt = R.get<WorldTransform>(camera);
        R.replace<ViewCamera>(viewport, ViewCamera{wt.P, wt.P + CameraForward(wt), R.get<Camera>(camera)});
    }
    { // Keep targeted PBR specialization mask in sync when one of its inputs changes.
        // Run before the UBO update below so Transmission pipeline is settled when the UBO reads it.
        const auto shading = R.get<const ViewportDisplay>(viewport).ViewportShading;
        if (!reactive<changes::ViewportDisplay>(R).empty() || !reactive<changes::PbrSpecialization>(R).empty()) {
            // SubmitViewport's full descriptor block runs only on resize, so write the transmission sampler
            // descriptor inline whenever EnsureTransmissionResources flips state.
            const auto refresh_transmission_descriptor = [&] {
                const auto &main = pipelines.Main;
                const vk::DescriptorImageInfo info = main.Transmission ? vk::DescriptorImageInfo{*main.Transmission->Sampler, *main.Transmission->Image.View, vk::ImageLayout::eShaderReadOnlyOptimal} : vk::DescriptorImageInfo{*main.Resources->NearestSampler, *main.Resources->ColorImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
                Vk.Device.updateDescriptorSets({Slots.MakeSamplerWrite(R.get<const SelectionSlots>(viewport).TransmissionSampler, info)}, {});
                request(RenderRequest::ReRecord);
            };
            if (shading == ViewportShadingMode::MaterialPreview || shading == ViewportShadingMode::Rendered) {
                PbrFeatureMask pbr_mask{0};
                const auto &active_lighting = GetActivePbrLighting(R, viewport, shading);
                if (active_lighting.UseSceneLights) pbr_mask |= PbrFeature::Punctual;
                for (const auto [_, feat] : R.view<const PbrMeshFeatures>().each()) pbr_mask |= feat.Mask;
                if (pipelines.Main.Compiler.CompilePipelines(pbr_mask)) request(RenderRequest::ReRecord);
                const bool want_transmission = active_lighting.RealTransmission && HasFeature(pbr_mask, PbrFeature::Transmission);
                const auto render_extent_now = ComputeRenderExtentPx(R.get<const ViewportExtent>(viewport).Value, std::bit_cast<vec2>(ImGui::GetIO().DisplayFramebufferScale));
                if (pipelines.Main.EnsureTransmissionResources(render_extent_now, Vk.Device, Vk.PhysicalDevice, want_transmission)) refresh_transmission_descriptor();
            } else if (pipelines.Main.EnsureTransmissionResources({}, Vk.Device, Vk.PhysicalDevice, false)) {
                refresh_transmission_descriptor();
            }
        }
    }

    if (!reactive<changes::SceneView>(R).empty() ||
        !reactive<changes::TransformPending>(R).empty() ||
        !reactive<changes::ViewportDisplay>(R).empty() ||
        !reactive<changes::InteractionMode>(R).empty() ||
        !reactive<changes::TransformEnd>(R).empty() ||
        light_count_changed) {
        const auto logical_extent = R.get<const ViewportExtent>(viewport).Value;
        const auto render_extent = ComputeRenderExtentPx(logical_extent, std::bit_cast<vec2>(ImGui::GetIO().DisplayFramebufferScale));
        const float aspect = render_extent.width == 0 || render_extent.height == 0 ? 1.f : float(render_extent.width) / float(render_extent.height);
        // When looking through a scene camera, keep the ViewCamera's widened FOV in sync
        // with the current viewport aspect ratio (handles viewport resize).
        if (const auto camera = LookThroughCameraEntity(R); camera != entt::null) {
            R.get<ViewCamera>(viewport).Data = WidenForLookThrough(R.get<Camera>(camera), aspect);
        }
        const auto &camera = R.get<const ViewCamera>(viewport);
        const auto &settings = R.get<const ViewportDisplay>(viewport);
        const bool is_pbr_mode = settings.ViewportShading == ViewportShadingMode::MaterialPreview || settings.ViewportShading == ViewportShadingMode::Rendered;
        const auto &active_lighting = GetActivePbrLighting(R, viewport, settings.ViewportShading);
        const bool use_scene_lights = is_pbr_mode && active_lighting.UseSceneLights;
        const bool use_scene_world = is_pbr_mode && active_lighting.UseSceneWorld;
        const auto &active_environment = use_scene_world ? Environments.SceneWorld : Environments.StudioWorld;
        const auto *source_assets = R.try_get<const gltf::SourceAssets>(viewport);
        const auto *source_ibl = source_assets && source_assets->ImageBasedLight.has_value() ? &*source_assets->ImageBasedLight : nullptr;
        const float env_intensity = use_scene_world && source_ibl ? source_ibl->Intensity : active_lighting.EnvIntensity;
        const mat3 env_rotation = [&]() -> mat3 {
            if (use_scene_world) return Environments.SceneWorldRotation;
            const float radians = active_lighting.EnvRotationDegrees * (Pi / 180.f);
            const float s = std::sin(radians), c = std::cos(radians);
            return {c, 0, -s, 0, 1, 0, s, 0, c};
        }();
        const float background_blur = active_lighting.BackgroundBlur;
        const float world_opacity = is_pbr_mode ? active_lighting.WorldOpacity : 0.f;
        const auto *pending = R.try_get<const PendingTransform>(viewport);
        // ScreenPixelScale: world-space size per pixel at unit distance (perspective) or absolute (ortho).
        // Sign encodes camera type: positive = perspective (shader multiplies by distance), negative = orthographic.
        const float screen_pixel_scale = ScreenPixelScale(camera.Data, std::max(float(render_extent.height), 1.f));
        const auto proj = camera.Projection(aspect);
        Buffers.SceneViewUBO.Update(as_bytes(SceneViewUBO{
            .ViewProj = proj * camera.View(),
            .ViewRotation = mat3(camera.View()),
            .CameraPosition = camera.Position(),
            .CameraNear = camera.NearClip(),
            .CameraFar = camera.FarClip(),
            .LightCount = Buffers.Lights.Count(),
            .LightSlot = Buffers.Lights.Slot(),
            .UseSceneLightsRender = use_scene_lights ? 1u : 0u,
            .EnvIntensity = env_intensity,
            .EnvRotation = env_rotation,
            .BackgroundBlur = background_blur,
            .WorldOpacity = world_opacity,
            .Ibl = active_environment.Ibl,
            .InteractionMode = interaction_mode,
            .EditElement = R.get<const EditMode>(viewport).Value,
            .IsTransforming = pending ? 1u : 0u,
            .PendingPivot = pending ? pending->Pivot : vec3{},
            .PendingTranslation = pending ? pending->Delta.P : vec3{},
            .PendingRotation = pending ? pending->Delta.R : quat{1, 0, 0, 0},
            .PendingScale = pending ? pending->Delta.S : vec3{1},
            .ScreenPixelScale = screen_pixel_scale,
            .ViewportSize = {float(render_extent.width), float(render_extent.height)},
            .FaceFirstTriSlot = Meshes.FaceFirstTriangleBuffer.Buffer.Slot,
            .BoneDeformSlot = Meshes.BoneDeformBuffer.Buffer.Slot,
            .ArmatureDeformSlot = Buffers.ArmatureDeformBuffer.Buffer.Slot,
            .MorphDeformSlot = Meshes.MorphTargetBuffer.Buffer.Slot,
            .MorphWeightsSlot = Buffers.MorphWeightBuffer.Buffer.Slot,
            .VertexClassSlot = Buffers.VertexClassBuffer.Buffer.Slot,
            .MaterialSlot = Buffers.Materials.Slot(),
            .PrimitiveMaterialSlot = Meshes.PrimitiveMaterialBuffer.Buffer.Slot,
            .FacePrimitiveSlot = Meshes.FacePrimitiveBuffer.Buffer.Slot,
            .DrawDataSlot = Buffers.RenderDraw.DrawData.Slot,
            .BoneXRay = settings.ViewportShading == ViewportShadingMode::Wireframe ? 1u : 0u,
            // Polygon offset factor matching Blender's GPU_polygon_offset_calc (viewdist = max ortho extent)
            .NdcOffsetFactor = std::holds_alternative<Perspective>(camera.Data) ? proj[3][2] * -0.00125f : 0.000005f * std::max(std::abs(1.f / proj[0][0]), std::abs(1.f / proj[1][1])),
            .TransmissionFramebufferSamplerSlot = R.get<const SelectionSlots>(viewport).TransmissionSampler,
            .TransmissionFramebufferMipCount = pipelines.Main.Transmission ? pipelines.Main.Transmission->MipCount : 1u,
            .UseRealTransmission = (is_pbr_mode && active_lighting.RealTransmission && pipelines.Main.Transmission) ? 1u : 0u,
            .DebugChannel = is_pbr_mode ? settings.DebugChannel : DebugChannel::None,
        }));
        R.emplace_or_replace<SelectionStale>(viewport);
        request(RenderRequest::Submit);
    }

    const auto &settings = R.get<const ViewportDisplay>(viewport);
    for (const auto mesh_entity : dirty_overlay_meshes) {
        const auto &mesh = R.get<const Mesh>(mesh_entity);
        R.patch<MeshBuffers>(mesh_entity, [&](auto &mesh_buffers) {
            for (const auto element : NormalElements) {
                if (ElementMaskContains(settings.NormalOverlays, element)) {
                    if (!mesh_buffers.NormalIndicators.contains(element)) {
                        const auto index_kind = element == Element::Face ? IndexKind::Face : IndexKind::Vertex;
                        mesh_buffers.NormalIndicators.emplace(
                            element,
                            Buffers.CreateRenderBuffers(CreateNormalVertices(mesh, element), CreateNormalIndices(mesh, element), index_kind)
                        );
                    } else {
                        Buffers.VertexBuffer.Update(mesh_buffers.NormalIndicators.at(element).Vertices, CreateNormalVertices(mesh, element));
                    }
                } else if (mesh_buffers.NormalIndicators.contains(element)) {
                    Buffers.Release(mesh_buffers.NormalIndicators.at(element));
                    mesh_buffers.NormalIndicators.erase(element);
                }
            }
        });
    }
    // Update mesh element state buffers (Excite mode only; Edit mode handled by GPU compute)
    for (const auto mesh_entity : dirty_element_state_meshes) {
        if (interaction_mode != InteractionMode::Excite) continue;
        const auto &mesh = R.get<const Mesh>(mesh_entity);
        std::unordered_set<VH> selected_vertices;
        std::unordered_set<EH> selected_edges, active_edges;
        std::unordered_set<FH> selected_faces;
        std::optional<uint32_t> active_handle, excited_handle;
        for (auto [entity, instance, excitable] : R.view<const Instance, const SoundVertices>().each()) {
            if (instance.Entity != mesh_entity) continue;
            selected_vertices.insert(excitable.Vertices.begin(), excitable.Vertices.end());
            if (const auto *force = R.try_get<const VertexForce>(entity)) excited_handle = force->Vertex;
            break;
        }
        if (const auto *active = R.try_get<const MeshActiveElement>(mesh_entity)) active_handle = active->Handle;
        Meshes.UpdateElementStates(mesh, Element::Vertex, selected_vertices, selected_edges, active_edges, selected_faces, active_handle, excited_handle);
        R.emplace_or_replace<SelectionStale>(viewport);
    }
    if (!dirty_element_state_meshes.empty()) request(RenderRequest::Submit);
    if (R.all_of<ElementStatesDirty>(viewport)) {
        R.remove<ElementStatesDirty>(viewport);
        request(RenderRequest::Submit);
    }
    for (auto &&[id, storage] : R.storage()) {
        if (storage.info() == entt::type_id<entt::reactive>()) storage.clear();
    }
    destroy_tracker.Storage.clear();
    R.clear<MeshGeometryDirty, MeshMaterialAssignment, MaterialDirty, SubmitDirty, LightWireframeDirty>();

    return render_request;
}
