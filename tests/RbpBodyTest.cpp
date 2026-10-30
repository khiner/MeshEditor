#include "physics/RbpBody.h"
#include "physics/RbpShape.h"

#include "Solver.h"
#include "TransformMath.h"
#include <boost/ut.hpp>

#include <random>

using namespace boost::ut;
using namespace physics;

namespace {
bool Near(rbp::float3 a, rbp::float3 b, float tolerance = 1e-4f) { return simd::length(a - b) < tolerance; }

rbp::Shape BoxShape(rbp::float3 half, rbp::Pose local = rbp::IdentityPose) {
    return {.HalfExtents = half, .Kind = rbp::ShapeBox, .Local = local};
}

// Reconstruct the tensor in node axes independently of the principal-axis ordering.
std::array<double, 9> Tensor(const rbp::World &world, const RbpBody &body) {
    const auto inv = world.Masses[body.Body].InvInertiaLocal;
    const auto q = body.Frame.Orientation;
    const rbp::float3 axes[]{rbp::Rotate(q, {1, 0, 0}), rbp::Rotate(q, {0, 1, 0}), rbp::Rotate(q, {0, 0, 1})};
    std::array<double, 9> result{};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            for (int axis = 0; axis < 3; ++axis)
                if (inv[axis] > 0) result[3 * row + col] += double(axes[axis][row]) * axes[axis][col] / inv[axis];
    return result;
}

void CheckTensor(const std::array<double, 9> &actual, const std::array<double, 9> &expected) {
    double largest = 1.0;
    for (const auto value : expected) largest = std::max(largest, std::abs(value));
    for (size_t i = 0; i < actual.size(); ++i) expect(std::abs(actual[i] - expected[i]) < 1e-4 * largest) << i << actual[i] << expected[i];
}

struct Box {
    rbp::float3 Half;
    rbp::Pose Local;
    double Volume() const { return 8.0 * Half.x * Half.y * Half.z; }
};
// The volume-weighted center of the boxes.
rbp::float3 Centroid(std::span<const Box> boxes) {
    rbp::float3 sum{};
    double volume = 0;
    for (const auto &box : boxes) {
        sum += box.Local.Position * float(box.Volume());
        volume += box.Volume();
    }
    return sum / float(volume);
}
// The inertia tensor of the boxes about `center` in node axes, with `mass` shared by volume, each box's own tensor turned by its pose and shifted by the parallel axis theorem.
std::array<double, 9> ExpectedTensor(std::span<const Box> boxes, double mass, rbp::float3 center) {
    double volume = 0;
    for (const auto &box : boxes) volume += box.Volume();
    std::array<double, 9> result{};
    for (const auto &box : boxes) {
        const double m = mass * box.Volume() / volume;
        const auto h = box.Half;
        const std::array<double, 3> diagonal{m / 3 * (h.y * h.y + h.z * h.z), m / 3 * (h.x * h.x + h.z * h.z), m / 3 * (h.x * h.x + h.y * h.y)};
        const rbp::float3 axes[]{rbp::Rotate(box.Local.Orientation, {1, 0, 0}), rbp::Rotate(box.Local.Orientation, {0, 1, 0}), rbp::Rotate(box.Local.Orientation, {0, 0, 1})};
        const auto d = box.Local.Position - center;
        const double d2 = double(simd::dot(d, d));
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) {
                double value = m * ((row == col ? d2 : 0.0) - double(d[row]) * d[col]);
                for (int axis = 0; axis < 3; ++axis) value += double(axes[axis][row]) * axes[axis][col] * diagonal[axis];
                result[3 * row + col] += value;
            }
    }
    return result;
}

const suite Bodies = [] {
    // Cylinders and planes keep their analytic geometry through signed and uneven scale, and only an elliptic or tapered cylinder becomes a hull.
    "shapes keep analytic geometry through scale"_test = [] {
        rbp::mtl::Context context;
        rbp::World world{context};
        const physics::Cylinder cylinder{.Height = 1.4f, .RadiusTop = 0.5f, .RadiusBottom = 0.5f};
        const auto local = rbp::At({1, 2, 3}, rbp::QuatFromRotationVector({0.2f, 0.3f, 0.1f}));
        const auto shape = BuildRbpShape(world, cylinder, nullptr, vec3{-2, -3, 2}, local);
        expect(world.Shapes[shape].Kind == rbp::ShapeCylinder);
        expect(Near(world.Shapes[shape].HalfExtents, rbp::float3{1, 2.1f, 0}));
        expect(Near(world.Shapes[shape].Local.Position, local.Position));
        expect(simd::all(world.Shapes[shape].Local.Orientation == local.Orientation));
        expect(world.Shapes[shape].VertexCount == 0_u);
        expect(world.Shapes[BuildRbpShape(world, cylinder, nullptr, vec3{2, 3, 1}, rbp::IdentityPose)].Kind == rbp::ShapeHull);
        auto tapered = cylinder;
        tapered.RadiusTop = 0.25f;
        expect(world.Shapes[BuildRbpShape(world, tapered, nullptr, vec3{1, 1, 1}, rbp::IdentityPose)].Kind == rbp::ShapeHull);
        for (const auto plane : {physics::Plane{.SizeX = 0.3f, .DoubleSided = true}, physics::Plane{.SizeZ = 0.2f, .DoubleSided = true}, physics::Plane{.SizeX = 0.3f, .SizeZ = 0.2f, .DoubleSided = true}}) {
            const auto placed = rbp::At({1, 1, 2}, rbp::QuatFromRotationVector({0, 0.35f, 0}));
            const auto floor = BuildRbpShape(world, plane, nullptr, vec3{-2, -1, 3}, placed);
            expect(world.Shapes[floor].Kind == rbp::ShapePlane);
            expect(Near(world.Shapes[floor].HalfExtents, rbp::float3{plane.SizeX.value_or(0), 0, 1.5f * plane.SizeZ.value_or(0)}));
            expect(Near(world.Shapes[floor].Normal, rbp::float3{0, -1, 0}));
            expect(world.Masses[BuildRbpBody(world, std::array{floor}, Transform{}, {}).Body].InvMass == 0);
        }
    };

    // Every body takes its mass, center and inertia from its colliders, shared by volume, or from the motion that authors them, and places each collider's corners where the node places them.
    "bodies take their mass properties from their colliders or their authored motion"_test = [] {
        rbp::mtl::Context context;
        std::mt19937 rng{11};
        const auto uniform = [&](float low, float high) { return std::uniform_real_distribution<float>{low, high}(rng); };
        const auto direction = [&] { return numeric::Normalize(vec3{uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)}); };
        for (int trial = 0; trial < 24; ++trial) {
            rbp::World world{context};
            std::vector<Box> boxes;
            std::vector<rbp::Index> shapes;
            for (uint32_t i = 0u, count = 1u + rng() % 3u; i < count; ++i) {
                boxes.push_back({{uniform(.2f, 3), uniform(.2f, 3), uniform(.2f, 3)}, rbp::At({uniform(-3, 3), uniform(-3, 3), uniform(-3, 3)}, rbp::QuatFromRotationVector({uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)}))});
                shapes.push_back(world.AddShape(BoxShape(boxes.back().Half, boxes.back().Local)));
            }
            const double mass = uniform(1, 20);
            // An authored center is in the node's scaled space, and authored inertia replaces the colliders' with locked axes.
            const bool authored_center = trial % 3 == 1, authored_inertia = trial % 3 == 2;
            const Transform node{.P = {uniform(-5, 5), uniform(-5, 5), uniform(-5, 5)}, .R = numeric::AngleAxis(uniform(0, 3), direction()), .S = authored_center ? vec3{uniform(.5f, 3), uniform(.5f, 3), uniform(.5f, 3)} : vec3{1}};
            const vec3 center_authored{uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)}, diagonal_authored{uniform(1, 5), uniform(1, 5), uniform(1, 5)};
            const auto turn = numeric::AngleAxis(uniform(0, 2), direction());
            PhysicsMotion motion{.Mass = float(mass)};
            if (authored_center || authored_inertia) motion.CenterOfMass = center_authored;
            if (authored_inertia) {
                motion.InertiaDiagonal = diagonal_authored;
                motion.InertiaOrientation = turn;
            }
            const auto body = BuildRbpBody(world, shapes, node, motion);
            expect(std::abs(world.Masses[body.Body].InvMass - 1.f / float(mass)) < 1e-6f);
            const auto center = authored_center || authored_inertia ? ToRbp(center_authored * node.S) : Centroid(boxes);
            expect(Near(body.Frame.Position, center, 1e-3f)) << trial;
            if (authored_inertia) {
                expect(Near(world.Masses[body.Body].InvInertiaLocal, 1.f / ToRbp(diagonal_authored), 1e-4f)) << trial;
                for (const auto axis : {vec3{1, 0, 0}, vec3{0, 1, 0}, vec3{0, 0, 1}}) expect(Near(rbp::Rotate(body.Frame.Orientation, ToRbp(axis)), ToRbp(turn * axis), 1e-4f)) << trial;
            } else CheckTensor(Tensor(world, body), ExpectedTensor(boxes, mass, center));
            const auto &compound = world.Shapes[body.Shape];
            for (uint32_t i = 0u; i < boxes.size(); ++i) {
                const auto &leaf = world.Shapes[world.Child(body.Shape, i)];
                const auto geometry = rbp::ComposePose(rbp::ComposePose(world.Poses[body.Body], compound.Local), leaf.Local);
                for (int corner = 0; corner < 8; ++corner) {
                    const rbp::float3 point{corner & 1 ? boxes[i].Half.x : -boxes[i].Half.x, corner & 2 ? boxes[i].Half.y : -boxes[i].Half.y, corner & 4 ? boxes[i].Half.z : -boxes[i].Half.z};
                    const auto actual = geometry.Position + rbp::Rotate(geometry.Orientation, point);
                    const auto expected = ToRbp(node.P + node.R * FromRbp(boxes[i].Local.Position + rbp::Rotate(boxes[i].Local.Orientation, point)));
                    expect(Near(actual, expected, 1e-3f)) << trial << i << corner;
                }
            }
            const auto sampled = RbpNodePose(world.Poses[body.Body], body.Frame);
            expect(Near(ToRbp(sampled.P), ToRbp(node.P), 1e-3f));
            expect(Near(ToRbp(sampled.R * vec3{1, 0, 0}), ToRbp(node.R * vec3{1, 0, 0}), 1e-3f));
        }
    };

    "body failures release owned colliders"_test = [] {
        rbp::mtl::Context context;
        rbp::World world{context, {.Bodies = 1, .Shapes = 4}};
        const auto shape = world.AddShape(BoxShape(rbp::float3{1, 1, 1}));
        world.AddBody({});
        const PhysicsMotion motion;
        expect(throws([&] { BuildRbpBody(world, std::array{shape}, Transform{}, motion); }));
        expect(world.ShapeCount() == 1_u);
        const PhysicsMotion invalid{.InertiaDiagonal = vec3{-1, 1, 1}};
        expect(throws([&] { BuildRbpBody(world, std::array{shape}, Transform{}, invalid); }));
        expect(world.ShapeCount() == 1_u);
        expect(world.RemoveShape(shape));
    };

    "shape replacements preserve identity and release temporary geometry"_test = [] {
        rbp::mtl::Context context;
        rbp::World world{context, {.Bodies = 1, .Shapes = 6, .CompoundChildren = 2}};
        const PhysicsMotion motion{.Mass = 12};
        auto source = world.AddShape(BoxShape(rbp::float3{1, 2, 3}));
        auto body = BuildRbpBody(world, std::array{source}, Transform{}, motion);
        world.RemoveShape(source);
        const auto id = world.IdOf(body.Body);
        for (int i = 0; i < 16; ++i) {
            const float half_x = i % 2 ? 1 : 2;
            source = world.AddShape(BoxShape(rbp::float3{half_x, 2, 3}, rbp::At({0.2f, 0, 0})));
            const auto old_shape = body.Shape;
            const PhysicsMotion invalid{.InertiaDiagonal = vec3{-1, 1, 1}};
            expect(throws([&] { BuildRbpBody(world, std::array{source}, Transform{}, invalid, {}, false, body); }));
            expect(throws([&] { BuildRbpBody(world, std::array{source, source}, Transform{}, motion, {}, false, body); }));
            expect(world.BodyShapes[body.Body] == old_shape);
            body = BuildRbpBody(world, std::array{source}, Transform{}, motion, {}, false, body);
            expect(world.IdOf(body.Body) == id);
            expect(world.BodyCount() == 1_u);
            expect(Near(body.Frame.Position, rbp::float3{0.2f, 0, 0}));
            CheckTensor(Tensor(world, body), {52, 0, 0, 0, 4 * (half_x * half_x + 9), 0, 0, 0, 4 * (half_x * half_x + 4)});
            expect(world.RemoveShape(source));
            expect(world.ShapeCount() <= 5_u);
        }
        expect(world.RemoveBody(body.Body));
        expect(world.RemoveShape(body.Shape));
        expect(world.ShapeCount() == 0_u);
    };
};
} // namespace
