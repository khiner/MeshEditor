#include "numeric/VectorMath.h"
#include "numeric/vec2.h"

#include "Variant.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "physics/ColliderUpdate.h"
#include "physics/PhysicsTypes.h"
#include "scene/Entity.h"
#include "scene/WorldTransform.h"
#include "state/Scene.h"
using numeric::Max;

void RederiveCollider(state::Scene &r, state::Entity e) {
    const auto *cs = r.try_get<const ColliderShape>(e);
    const auto *policy = r.try_get<const ColliderPolicy>(e);
    if (!cs || !policy) return;
    const auto mesh_entity = cs->MeshEntity != state::Null ? cs->MeshEntity : FindMeshEntity(r, e);
    const auto mesh = TryGetMesh(r, mesh_entity);
    if (!mesh) return;

    const bool has_verts = mesh->VertexCount()!=0u;

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

    const bool fit_dimensions=policy->AutoFitDims && !authored_primitive && has_verts &&
        (std::holds_alternative<physics::Box>(shape) || std::holds_alternative<physics::Sphere>(shape) ||
         std::holds_alternative<physics::Cylinder>(shape) || std::holds_alternative<physics::Capsule>(shape));
    if (fit_dimensions) {
        auto &meshes=r.Context.get<MeshStore>();
        meshes.EnsureSelectionState(r,std::array{mesh->GetStoreId()});
        const auto aabb=meshes.GetSelectionRoot(mesh->GetStoreId(),Element::Vertex).Bounds;
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

    const auto derived_mesh=IsMeshBackedShape(shape) ? mesh_entity : state::Null;
    ColliderShape derived{.Shape=std::move(shape),
                          .MeshEntity=derived_mesh,
                          .LocalOffset=local_offset};
    if (*cs==derived) return;
    r.patch<ColliderShape>(e,[&](ColliderShape &x) { x=std::move(derived); });
}
