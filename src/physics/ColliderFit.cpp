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
BBox ComputeBBox(const Mesh &mesh) {
    BBox bbox;
    for (const auto &v : mesh.GetVerticesSpan()) {
        bbox.Min = glm::min(bbox.Min, v.Position);
        bbox.Max = glm::max(bbox.Max, v.Position);
    }
    return bbox;
}
} // namespace

ColliderShape CreateInitialCollider(const entt::registry &r, entt::entity entity) {
    const auto mesh_entity = FindMeshEntity(r, entity);
    const auto shape = [&]() -> PhysicsShape {
        if (const auto *prim = r.try_get<const PrimitiveShape>(mesh_entity)) {
            return std::visit(
                overloaded{
                    [](const primitive::Cuboid &s) -> PhysicsShape { return physics::Box{s.HalfExtents * 2.f}; },
                    [](const primitive::Plane &s) -> PhysicsShape { return physics::Plane{s.HalfExtents.x * 2.f, s.HalfExtents.y * 2.f}; },
                    [](const primitive::IcoSphere &s) -> PhysicsShape { return physics::Sphere{s.Radius}; },
                    [](const primitive::UVSphere &s) -> PhysicsShape { return physics::Sphere{s.Radius}; },
                    [](const primitive::Cylinder &s) -> PhysicsShape { return physics::Cylinder{s.Height, s.Radius, s.Radius}; },
                    [](const primitive::Cone &s) -> PhysicsShape { return physics::Cylinder{s.Height, 0.f, s.Radius}; },
                    // Circle, Torus → ConvexHull from mesh geometry
                    [](const auto &) -> PhysicsShape { return physics::ConvexHull{}; },
                },
                *prim
            );
        }

        // Fallback: derive shape kind from mesh BBox.
        const auto *mesh = r.try_get<const Mesh>(mesh_entity);
        if (!mesh || mesh->VertexCount() == 0) return physics::Box{};

        const auto bbox = ComputeBBox(*mesh);
        const vec3 extents = bbox.Max - bbox.Min;
        // If any BBox dimension is degenerate, use ConvexHull instead of a zero-thickness box.
        if (extents.x < 1e-6f || extents.y < 1e-6f || extents.z < 1e-6f) return physics::ConvexHull{};
        return physics::Box{extents};
    }();
    return ColliderShape{.Shape = shape, .MeshEntity = IsMeshBackedShape(shape) ? mesh_entity : null_entity};
}

void RefitAutoFitShape(entt::registry &r, entt::entity e) {
    const auto *cs = r.try_get<const ColliderShape>(e);
    if (!cs) return;
    const auto mesh_entity = cs->MeshEntity != null_entity ? cs->MeshEntity : FindMeshEntity(r, e);
    const auto *mesh = r.try_get<const Mesh>(mesh_entity);
    if (!mesh || mesh->VertexCount() == 0) return;
    auto fitted = cs->Shape;
    FitShapeToMesh(fitted, *mesh);
    r.patch<ColliderShape>(e, [&](ColliderShape &x) { x.Shape = std::move(fitted); });
}

void FitShapeToMesh(PhysicsShape &shape, const Mesh &mesh) {
    const auto bbox = ComputeBBox(mesh);
    const vec3 extents = bbox.Max - bbox.Min;

    std::visit(
        overloaded{
            [&](physics::Box &s) { s.Size = extents; },
            [&](physics::Sphere &s) { s.Radius = glm::compMax(extents) * 0.5f; },
            [&](physics::Cylinder &s) {
                const float radius = glm::max(extents.x, extents.z) * 0.5f;
                s.RadiusTop = radius;
                s.RadiusBottom = radius;
                s.Height = extents.y;
            },
            [&](physics::Capsule &s) {
                const float radius = glm::max(extents.x, extents.z) * 0.5f;
                s.RadiusTop = radius;
                s.RadiusBottom = radius;
                s.Height = glm::max(0.f, extents.y - 2.f * radius);
            },
            [](auto &) {}, // Plane, ConvexHull, TriangleMesh: no fittable dims
        },
        shape
    );
}
