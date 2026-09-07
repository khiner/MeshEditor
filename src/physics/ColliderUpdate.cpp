#include "physics/ColliderUpdate.h"
#include "Variant.h"
#include "mesh/Mesh.h"
#include "mesh/Primitives.h"
#include "physics/PhysicsTypes.h"
#include "scene/Entity.h"
#include "scene/WorldTransform.h"
#include <entt/entity/registry.hpp>
#include <numbers>
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
    // Preserve manually configured dimensions and offset.
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
            // Use a convex hull for imported and de-primitivized meshes.
            shape = physics::ConvexHull{};
        }
    }

    if (policy->AutoFitDims && has_verts) {
        // Ritter's algorithm uses two farthest-point passes and one expansion pass (Real-Time Collision Detection section 4.3.5).
        auto ritter = [&]() -> std::pair<vec3, float> {
            const auto farthest_from = [&](vec3 from) {
                vec3 best = from;
                float best_d2 = 0;
                for (const auto &v : verts) {
                    const vec3 delta = v.Position - from;
                    const float d2 = numeric::Dot(delta, delta);
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
            float radius = numeric::Length(ru - c);
            for (const auto &v : verts) {
                const float d = numeric::Length(v.Position - c);
                if (d > radius) {
                    const float new_r = (radius + d) * 0.5f;
                    c = c + ((d - radius) / (2.f * d)) * (v.Position - c);
                    radius = new_r;
                }
            }
            return {c, radius};
        };
        // Compute the tightest radius around the Y axis through aabb_center.
        const auto xz_radius = [&] {
            const vec2 c{aabb_center.x, aabb_center.z};
            float r = 0;
            for (const auto &v : verts) r = numeric::Max(r, numeric::Length(vec2{v.Position.x, v.Position.z} - c));
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
                    s.Height = numeric::Max(physics::MinShapeHeight, aabb_extents.y);
                    local_offset = aabb_center;
                },
                [&](physics::Capsule &s) {
                    const float radius = xz_radius();
                    s.RadiusTop = s.RadiusBottom = radius;
                    // 2r >= aabb.y degenerates toward a sphere, so clamp height to keep it spec-valid.
                    s.Height = numeric::Max(physics::MinShapeHeight, aabb_extents.y - 2.f * radius);
                    local_offset = aabb_center;
                },
                [](auto &) {},
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
