#include "numeric/VectorMath.h"
#include "numeric/vec2.h"

#include "Variant.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "metal/Dispatch.h"
#include "physics/ColliderUpdate.h"
#include "physics/PhysicsTypes.h"
#include "scene/Entity.h"
#include "scene/WorldTransform.h"
#include "state/Scene.h"
using numeric::Max;

namespace {
state::Entity ColliderMesh(const state::Scene &r, state::Entity collider, const ColliderShape &shape) {
    return shape.MeshEntity != state::Null ? shape.MeshEntity : FindMeshEntity(r, collider);
}
} // namespace

void UpdateMeshColliders(state::Scene &r) {
    const auto &changed = reactive(r, state::Change::Colliders);
    if (changed.empty()) return;
    auto &index = r.Context.get<MeshColliders>();
    // A collider keeping its mesh, as a dimension fit does, leaves the index in place.
    const bool placed = std::ranges::all_of(changed, [&](state::Entity e) {
        const auto *shape = r.valid(e) ? r.try_get<const ColliderShape>(e) : nullptr;
        return shape && std::ranges::contains(index.Of(ColliderMesh(r, e, *shape)), e);
    });
    if (placed) return;
    index.ByMesh.clear();
    for (const auto [e, shape] : r.view<const ColliderShape>().each()) index.ByMesh[ColliderMesh(r, e, shape)].push_back(e);
}

void RederiveColliders(state::Scene &r, std::span<const state::Entity> entities) {
    // A collider's shape before fitting, with the store whose vertex bounds fit its dimensions, or none.
    struct Derivation {
        state::Entity Entity, MeshEntity;
        PhysicsShape Shape;
        vec3 LocalOffset;
        uint32_t FitStoreId;
    };
    std::vector<Derivation> derivations;
    std::vector<uint32_t> fit_ids;
    for (const auto e : entities) {
        const auto *cs = r.try_get<const ColliderShape>(e);
        const auto *policy = r.try_get<const ColliderPolicy>(e);
        if (!cs || !policy) continue;
        const auto mesh_entity = ColliderMesh(r, e, *cs);
        const auto mesh = TryGetMesh(r, mesh_entity);
        if (!mesh) continue;

        const bool has_verts = mesh->VertexCount() != 0u;

        PhysicsShape shape = cs->Shape;
        // Preserve manually configured dimensions and offset.
        vec3 local_offset = policy->AutoFitDims ? vec3{0} : cs->LocalOffset;
        bool authored_primitive = false;

        if (policy->AutoFitDims && !policy->LockedKind) {
            if (const auto *prim = r.try_get<const PrimitiveShape>(mesh_entity)) {
                authored_primitive = true;
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
                // Use a convex hull for imported and de-primitivized meshes.
                shape = physics::ConvexHull{};
            }
        }

        const bool fit_dimensions = policy->AutoFitDims && !authored_primitive && has_verts &&
            (std::holds_alternative<physics::Box>(shape) || std::holds_alternative<physics::Sphere>(shape) ||
             std::holds_alternative<physics::Cylinder>(shape) || std::holds_alternative<physics::Capsule>(shape));
        const auto fit_id = fit_dimensions ? mesh->GetStoreId() : InvalidOffset;
        if (fit_id != InvalidOffset) fit_ids.push_back(fit_id);
        derivations.push_back({e, mesh_entity, std::move(shape), local_offset, fit_id});
    }
    auto &meshes = r.Context.get<MeshStore>();
    if (!fit_ids.empty()) {
        mtl::ComputeChain chain{meshes.BufferContext()};
        meshes.EnsureSelectionState(r, chain, fit_ids);
    }
    for (auto &[e, mesh_entity, shape, local_offset, fit_id] : derivations) {
        if (fit_id != InvalidOffset) {
            const auto aabb = meshes.GetSelectionRoot(fit_id, Element::Vertex).Bounds;
            const vec3 aabb_center = (aabb.Min + aabb.Max) * 0.5f;
            const vec3 aabb_extents = aabb.Max - aabb.Min;
            // The incrementally maintained bounds enclose every live vertex.
            const float sphere_radius = 0.5f * Length(aabb_extents);
            const float xz_radius = 0.5f * Length(vec2{aabb_extents.x, aabb_extents.z});

            std::visit(
                overloaded{
                    [&](physics::Box &s) {
                        s.Size = aabb_extents;
                        local_offset = aabb_center;
                    },
                    [&](physics::Sphere &s) {
                        s.Radius = sphere_radius;
                        local_offset = aabb_center;
                    },
                    [&](physics::Cylinder &s) {
                        s.RadiusTop = s.RadiusBottom = xz_radius;
                        s.Height = Max(physics::MinShapeHeight, aabb_extents.y);
                        local_offset = aabb_center;
                    },
                    [&](physics::Capsule &s) {
                        s.RadiusTop = s.RadiusBottom = xz_radius;
                        // 2r >= aabb.y degenerates toward a sphere, so clamp height to keep it spec-valid.
                        s.Height = Max(physics::MinShapeHeight, aabb_extents.y - 2.f * xz_radius);
                        local_offset = aabb_center;
                    },
                    [](auto &) {},
                },
                shape
            );
        }

        const auto derived_mesh = IsMeshBackedShape(shape) ? mesh_entity : state::Null;
        ColliderShape derived{.Shape = std::move(shape), .MeshEntity = derived_mesh, .LocalOffset = local_offset};
        if (r.get<const ColliderShape>(e) == derived) continue;
        r.patch<ColliderShape>(e, [&](ColliderShape &x) { x = std::move(derived); });
    }
}
