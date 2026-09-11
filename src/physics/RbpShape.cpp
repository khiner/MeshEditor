#include "RbpShape.h"
#include "RbpBody.h"

#include "Variant.h"
#include "World.h"
#include "mesh/Mesh.h"

#include <numbers>
#include <stdexcept>

namespace physics {
rbp::Index BuildRbpShape(rbp::World &world, const PhysicsShape &source, const Mesh *mesh, vec3 scale, rbp::Pose local) {
    const auto stretch = numeric::Abs(scale);
    if (stretch.x <= 0 || stretch.y <= 0 || stretch.z <= 0) throw std::invalid_argument("A physics collider must have nonzero scale.");
    const auto hull = [&](std::span<rbp::float3> points) {
        for (auto &point : points) point *= ToRbp(scale);
        return world.AddHull(points, nullptr, local);
    };
    // Support samples of the convex hull of two spheres, including each axis extremum.
    const auto round = [&](float half, float top, float bottom) {
        rbp::float3 points[64];
        uint32_t count = 0;
        const auto add = [&](rbp::float3 direction) {
            const auto a = rbp::float3{0, half, 0} + top * direction;
            const auto b = rbp::float3{0, -half, 0} + bottom * direction;
            points[count++] = simd::dot(a, direction) >= simd::dot(b, direction) ? a : b;
        };
        for (int axis = 0; axis < 3; ++axis) {
            rbp::float3 direction{};
            direction[axis] = 1;
            add(direction);
            add(-direction);
        }
        for (int i = 0; i < 58; ++i) {
            const float y = 1 - 2 * (float(i) + 0.5f) / 58;
            const float radius = std::sqrt(1 - y * y), angle = 2.39996323f * float(i);
            add(rbp::float3{radius * std::cos(angle), y, radius * std::sin(angle)});
        }
        return hull(points);
    };
    rbp::Shape shape{};
    shape.Local = local;
    const auto create = overloaded{
        [&](const physics::Box &box) {
            shape.Kind = rbp::ShapeBox;
            shape.HalfExtents = ToRbp(box.Size * stretch * 0.5f);
            return world.AddShape(shape);
        },
        [&](const Sphere &sphere) {
            if (stretch.x != stretch.y || stretch.y != stretch.z) return round(0, sphere.Radius, sphere.Radius);
            shape.Kind = rbp::ShapeSphere;
            shape.Radius = sphere.Radius * stretch.x;
            return world.AddShape(shape);
        },
        [&](const Capsule &capsule) {
            if (capsule.RadiusTop != capsule.RadiusBottom || stretch.x != stretch.y || stretch.y != stretch.z)
                return round(capsule.Height * 0.5f, capsule.RadiusTop, capsule.RadiusBottom);
            shape.Kind = rbp::ShapeCapsule;
            shape.Radius = capsule.RadiusTop * stretch.x;
            shape.HalfExtents.y = capsule.Height * stretch.y * 0.5f;
            return world.AddShape(shape);
        },
        [&](const Cylinder &cylinder) {
            if (cylinder.RadiusTop == cylinder.RadiusBottom && stretch.x == stretch.z) {
                shape.Kind = rbp::ShapeCylinder;
                shape.HalfExtents = {cylinder.RadiusTop * stretch.x, cylinder.Height * stretch.y * 0.5f, 0};
                return world.AddShape(shape);
            }
            rbp::float3 points[64];
            for (int i = 0; i < 32; ++i) {
                const float angle = 2 * std::numbers::pi_v<float> * float(i) / 32;
                for (int end = 0; end < 2; ++end) {
                    const float radius = end ? cylinder.RadiusTop : cylinder.RadiusBottom;
                    points[2 * i + end] = {radius * std::cos(angle), (end ? 0.5f : -0.5f) * cylinder.Height, radius * std::sin(angle)};
                }
            }
            return hull(points);
        },
        [&](const Plane &plane) {
            shape.Kind = rbp::ShapePlane;
            shape.Normal = {0, scale.y > 0 ? 1.f : -1.f, 0};
            shape.HalfExtents = {plane.SizeX * stretch.x * 0.5f, 0, plane.SizeZ * stretch.z * 0.5f};
            shape.DoubleSided = plane.DoubleSided;
            return world.AddShape(shape);
        },
        [&](const auto &kind) {
            if (!mesh) throw std::runtime_error("A mesh collider has no mesh geometry.");
            std::vector<rbp::float3> points;
            points.reserve(mesh->VertexCount());
            for (auto vertex : mesh->vertices()) points.push_back(ToRbp(mesh->GetPosition(vertex) * scale));
            if constexpr (std::is_same_v<std::decay_t<decltype(kind)>, ConvexHull>) return world.AddHull(points, nullptr, local);
            else {
                auto indices = mesh->CreateTriangleIndices();
                if (scale.x * scale.y * scale.z < 0)
                    for (size_t i = 0; i + 2 < indices.size(); i += 3) std::swap(indices[i + 1], indices[i + 2]);
                const auto mesh_shape = world.AddMesh(points, indices, local);
                if (mesh_shape != rbp::NoIndex) world.Shapes[mesh_shape].DoubleSided = true;
                return mesh_shape;
            }
        },
    };
    const auto result = std::visit(create, source);
    if (result == rbp::NoIndex) throw std::runtime_error("RBP could not cook or allocate a collider.");
    return result;
}
} // namespace physics
