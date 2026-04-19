#include "ColliderFit.h"

#include "BBox.h"
#include "Instance.h"
#include "Variant.h"
#include "mesh/Mesh.h"
#include "mesh/PrimitiveType.h"

#include <entt/entity/registry.hpp>

entt::entity FindMeshEntity(const entt::registry &r, entt::entity entity) {
    if (const auto *instance = r.try_get<const Instance>(entity)) return instance->Entity;
    return entity;
}

namespace {
std::pair<vec3, bool> MeshExtents(const Mesh &mesh) {
    if (mesh.VertexCount() == 0) return {vec3{0}, false};
    BBox bbox;
    for (const auto &v : mesh.GetVerticesSpan()) {
        bbox.Min = glm::min(bbox.Min, v.Position);
        bbox.Max = glm::max(bbox.Max, v.Position);
    }
    return {bbox.Max - bbox.Min, true};
}
} // namespace

void RederiveCollider(entt::registry &r, entt::entity e) {
    const auto *cs = r.try_get<const ColliderShape>(e);
    const auto *policy = r.try_get<const ColliderPolicy>(e);
    if (!cs || !policy) return;
    const auto mesh_entity = cs->MeshEntity != null_entity ? cs->MeshEntity : FindMeshEntity(r, e);
    const auto *mesh = r.try_get<const Mesh>(mesh_entity);
    if (!mesh) return;

    const auto [extents, has_extents] = MeshExtents(*mesh);
    PhysicsShape shape = cs->Shape;

    // AutoFitDims off → user owns kind and dims, leave both alone.
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

    if (policy->AutoFitDims && has_extents) {
        std::visit(
            overloaded{
                [&](physics::Box &s) { s.Size = extents; },
                [&](physics::Sphere &s) { s.Radius = glm::compMax(extents) * 0.5f; },
                [&](physics::Cylinder &s) {
                    const float r = glm::max(extents.x, extents.z) * 0.5f;
                    s.RadiusTop = s.RadiusBottom = r;
                    s.Height = extents.y;
                },
                [&](physics::Capsule &s) {
                    const float r = glm::max(extents.x, extents.z) * 0.5f;
                    s.RadiusTop = s.RadiusBottom = r;
                    s.Height = glm::max(0.f, extents.y - 2.f * r);
                },
                [](auto &) {}, // Plane, ConvexHull, TriangleMesh: no fittable dims
            },
            shape
        );
    }

    r.patch<ColliderShape>(e, [&](ColliderShape &x) {
        x.Shape = std::move(shape);
        x.MeshEntity = IsMeshBackedShape(x.Shape) ? mesh_entity : null_entity;
    });
}
