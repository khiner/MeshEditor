#include "physics/ColliderUpdate.h"
#include "Variant.h"
#include "mesh/MeshStore.h"
#include "physics/PhysicsChanges.h"
#include "physics/PhysicsTypes.h"
#include "procedure/Procedure.h"
#include "scene/Entity.h"
#include "state/Registry.h"
using numeric::Max;

namespace {
state::Entity ColliderMesh(const state::Registry &r, state::Entity collider, const ColliderShape &shape) {
    return shape.MeshEntity != state::Null ? shape.MeshEntity : FindMeshEntity(r, collider);
}
std::optional<PrimitiveParameters> PrimitiveOf(const state::Registry &r, state::Entity mesh_entity) {
    using namespace procedure;
    const auto *procedure = r.try_get<const GeometryProcedure>(mesh_entity);
    if (!procedure || procedure->DisplayNode == state::Null) return {};
    const auto display = procedure->DisplayNode;
    if (std::ranges::any_of(r.get<const NodeInputs>(display).Links, [](SocketLink input) { return input.FromNode != state::Null; })) return {};
    std::optional<PrimitiveParameters> parameters;
    VisitKind(r.get<const ProcedureNode>(display).Kind, [&]<typename K> {
        if constexpr (Generator<K>) parameters = r.get<const K>(display);
    });
    return parameters;
}
} // namespace

std::span<const state::Entity> MeshCollidersOf(const state::Registry &r, state::Entity mesh_entity) {
    const auto *list = r.try_get<const MeshColliderList>(mesh_entity);
    return list ? std::span{list->Colliders} : std::span<const state::Entity>{};
}

void UpdateMeshColliders(state::Registry &r) {
    for (const auto e : r.Resources.get<const PhysicsChanges>().Colliders) {
        if (!r.valid(e)) continue;
        const auto *shape = r.try_get<const ColliderShape>(e);
        const auto mesh_entity = shape ? ColliderMesh(r, e, *shape) : state::Null;
        if (const auto *listed = r.try_get<const ColliderListedUnder>(e); listed && listed->MeshEntity == mesh_entity) continue;
        r.remove<ColliderListedUnder>(e);
        if (!r.valid(mesh_entity)) continue;
        r.get_or_emplace<MeshColliderList>(mesh_entity).Colliders.push_back(e);
        r.emplace<ColliderListedUnder>(e, mesh_entity);
    }
}

void RegisterMeshColliderHandlers(state::Registry &r) {
    r.on_destroy<ColliderListedUnder, [](state::Registry &r, state::Entity e) {
        const auto mesh_entity = r.get<const ColliderListedUnder>(e).MeshEntity;
        auto *list = r.try_edit<MeshColliderList>(mesh_entity);
        if (!list) return;
        std::erase(list->Colliders, e);
        if (list->Colliders.empty()) r.remove<MeshColliderList>(mesh_entity);
    }>();
}

std::optional<ColliderShape> DeriveColliderShape(const state::Registry &r, state::Entity e, std::optional<AABB> vertex_bounds) {
    const auto *cs = r.try_get<const ColliderShape>(e);
    const auto *policy = r.try_get<const ColliderPolicy>(e);
    if (!cs || !policy) return {};
    const auto mesh_entity = ColliderMesh(r, e, *cs);
    if (!TryGetMesh(r, mesh_entity)) return {};

    PhysicsShape shape = cs->Shape;
    vec3 local_offset = policy->AutoFitDims ? vec3{0} : cs->LocalOffset;
    bool authored_primitive = false;

    if (policy->AutoFitDims && !policy->LockedKind) {
        if (const auto prim = PrimitiveOf(r, mesh_entity)) {
            authored_primitive = true;
            shape = std::visit(
                overloaded{
                    [](const primitive::Cube &s) -> PhysicsShape { return physics::Box{s.HalfExtents * 2.f}; },
                    [](const primitive::Plane &s) -> PhysicsShape { return physics::Plane{s.HalfExtents.x * 2.f, s.HalfExtents.y * 2.f}; },
                    [](const primitive::IcoSphere &s) -> PhysicsShape { return physics::Sphere{s.Radius}; },
                    [](const primitive::UVSphere &s) -> PhysicsShape { return physics::Sphere{s.Radius}; },
                    [](const primitive::QuadSphere &s) -> PhysicsShape { return physics::Sphere{s.Radius}; },
                    [](const primitive::Cylinder &s) -> PhysicsShape { return physics::Cylinder{s.Height, s.TopRadius.value_or(s.Radius), s.Radius}; },
                    [](const auto &) -> PhysicsShape { return physics::ConvexHull{}; },
                },
                *prim
            );
        } else {
            shape = physics::ConvexHull{};
        }
    }

    if (policy->AutoFitDims && !authored_primitive && vertex_bounds) {
        const auto &aabb = *vertex_bounds;
        const vec3 aabb_center = (aabb.Min + aabb.Max) * 0.5f;
        const vec3 aabb_extents = aabb.Max - aabb.Min;
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
                    // The height clamps at MinShapeHeight because 2r >= aabb.y degenerates toward a sphere.
                    s.Height = Max(physics::MinShapeHeight, aabb_extents.y - 2.f * xz_radius);
                    local_offset = aabb_center;
                },
                [](auto &) {},
            },
            shape
        );
    }

    const auto derived_mesh = IsMeshBackedShape(shape) ? mesh_entity : state::Null;
    return ColliderShape{.Shape = std::move(shape), .MeshEntity = derived_mesh, .LocalOffset = local_offset};
}

void RederiveColliders(state::Registry &r, std::span<const state::Entity> entities) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    for (const auto e : entities) {
        const auto *cs = r.try_get<const ColliderShape>(e);
        const auto mesh = cs ? TryGetMesh(r, ColliderMesh(r, e, *cs)) : std::nullopt;
        const auto bounds = mesh && mesh->VertexCount() != 0u ? std::optional{meshes.GetSelectionRoot(mesh->GetStoreId(), Element::Vertex).Bounds} : std::nullopt;
        auto derived = DeriveColliderShape(r, e, bounds);
        if (!derived || r.get<const ColliderShape>(e) == *derived) continue;
        r.patch<ColliderShape>(e, [&](ColliderShape &x) { x = std::move(*derived); });
    }
}
