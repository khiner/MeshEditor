#include "physics/RbpBody.h"
#include "physics/RbpShape.h"

#include "RunSuites.h"
#include "Solver.h"
#include "TransformMath.h"

#include <boost/ut.hpp>

#include <array>
#include <cmath>

using namespace boost::ut;
using namespace physics;

namespace {
bool Near(rbp::float3 a, rbp::float3 b, float tolerance = 1e-4f) { return simd::length(a - b) < tolerance; }

rbp::Shape BoxShape(rbp::float3 half, rbp::Pose local = rbp::IdentityPose) {
    rbp::Shape result{};
    result.Kind = rbp::ShapeBox;
    result.HalfExtents = half;
    result.Local = local;
    return result;
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
    for (size_t i = 0; i < actual.size(); ++i) expect(std::abs(actual[i] - expected[i]) < 1e-4) << i << actual[i] << expected[i];
}

const suite Bodies = [] {
    "RBP circular cylinders retain analytic geometry through signed scale"_test = [] {
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
        const auto elliptic = BuildRbpShape(world, cylinder, nullptr, vec3{2, 3, 1}, rbp::IdentityPose);
        expect(world.Shapes[elliptic].Kind == rbp::ShapeHull);
        auto tapered = cylinder;
        tapered.RadiusTop = 0.25f;
        const auto taper = BuildRbpShape(world, tapered, nullptr, vec3{1, 1, 1}, rbp::IdentityPose);
        expect(world.Shapes[taper].Kind == rbp::ShapeHull);
    };

    "RBP planes retain finite axes through scaling and body construction"_test = [] {
        rbp::mtl::Context context;
        rbp::Solver solver{context};
        for (const auto plane : {physics::Plane{.SizeX = 0.3f, .DoubleSided = true}, physics::Plane{.SizeZ = 0.2f, .DoubleSided = true}, physics::Plane{.SizeX = 0.3f, .SizeZ = 0.2f, .DoubleSided = true}}) {
            rbp::World world{context};
            const auto local = rbp::At({1, 1, 2}, rbp::QuatFromRotationVector({0, 0.35f, 0}));
            const auto shape = BuildRbpShape(world, plane, nullptr, vec3{-2, -1, 3}, local);
            expect(world.Shapes[shape].Kind == rbp::ShapePlane);
            expect(Near(world.Shapes[shape].HalfExtents, rbp::float3{plane.SizeX, 0, 1.5f * plane.SizeZ}));
            expect(Near(world.Shapes[shape].Normal, rbp::float3{0, -1, 0}));
            const auto floor = BuildRbpBody(world, std::array{shape}, Transform{}, nullptr);
            const auto cube = world.AddShape(BoxShape(rbp::float3{0.5f, 0.5f, 0.5f}));
            const PhysicsMotion motion{.Mass = 1};
            const auto body = BuildRbpBody(world, std::array{cube}, Transform{.P = {1, 3, 2}}, &motion);
            const auto beyond = rbp::WorldPoint(local, plane.SizeX > 0 ? rbp::float3{2, 2, 0} : rbp::float3{0, 2, 2});
            rbp::BodyDesc miss{};
            miss.Pose = rbp::At(beyond);
            miss.Shape = cube;
            const auto missed = world.AddBody(miss);
            solver.Advance(world, {}, 180);
            expect(std::abs(world.Poses[body.Body].Position.y - 1.5f) < 0.005f);
            expect(simd::length(world.Velocities[body.Body].Linear) < 0.005f);
            expect(world.Poses[missed].Position.y < -2);
            expect(world.Masses[floor.Body].InvMass == 0);
        }
    };

    "RBP body retains anisotropic box tensor and node geometry"_test = [] {
        rbp::mtl::Context context;
        rbp::World world{context};
        const auto shape = world.AddShape(BoxShape(rbp::float3{1, 2, 3}, rbp::At({2, -1, 3})));
        const Transform node{.P = {7, 8, 9}, .R = numeric::AngleAxis(0.7f, numeric::Normalize(vec3{1, 2, 3}))};
        const PhysicsMotion motion{.Mass = 12};
        const auto body = BuildRbpBody(world, std::array{shape}, node, &motion);
        expect(Near(body.Frame.Position, rbp::float3{2, -1, 3}));
        expect(std::abs(world.Masses[body.Body].InvMass - 1.f / 12) < 1e-6f);
        CheckTensor(Tensor(world, body), {52, 0, 0, 0, 40, 0, 0, 0, 20});
        const auto &compound = world.Shapes[body.Shape];
        const auto &leaf = world.Shapes[world.Child(body.Shape, 0)];
        const auto geometry = rbp::ComposePose(rbp::ComposePose(world.Poses[body.Body], compound.Local), leaf.Local);
        for (int corner = 0; corner < 8; ++corner) {
            const rbp::float3 point{corner & 1 ? 1.f : -1.f, corner & 2 ? 2.f : -2.f, corner & 4 ? 3.f : -3.f};
            const auto actual = geometry.Position + rbp::Rotate(geometry.Orientation, point);
            const auto expected = ToRbp(node.P + node.R * (FromRbp(point) + vec3{2, -1, 3}));
            expect(Near(actual, expected));
        }
        const auto sampled = RbpNodePose(world.Poses[body.Body], body.Frame);
        expect(Near(ToRbp(sampled.P), ToRbp(node.P)));
        expect(Near(ToRbp(sampled.R * vec3{1, 0, 0}), ToRbp(node.R * vec3{1, 0, 0})));
    };

    "RBP authored centre carries the full parallel-axis tensor"_test = [] {
        rbp::mtl::Context context;
        rbp::World world{context};
        const auto shape = world.AddShape(BoxShape(rbp::float3{1, 2, 3}));
        const PhysicsMotion motion{.Mass = 12, .CenterOfMass = vec3{0.5f, 0.5f, 0}};
        const auto body = BuildRbpBody(world, std::array{shape}, Transform{.S = {2, 4, 1}}, &motion);
        expect(Near(body.Frame.Position, rbp::float3{1, 2, 0}));
        CheckTensor(Tensor(world, body), {100, -24, 0, -24, 52, 0, 0, 0, 80});
    };

    "RBP authored inertia overrides geometry and preserves locked axes"_test = [] {
        rbp::mtl::Context context;
        rbp::World world{context};
        const auto shape = world.AddShape(BoxShape(rbp::float3{1, 2, 3}));
        const auto turn = numeric::AngleAxis(float(std::numbers::pi / 2), vec3{0, 0, 1});
        const PhysicsMotion motion{.Mass = 0, .CenterOfMass = vec3{2, 3, 4}, .InertiaDiagonal = vec3{2, 0, 8}, .InertiaOrientation = turn};
        const auto body = BuildRbpBody(world, std::array{shape}, Transform{}, &motion);
        expect(world.Masses[body.Body].InvMass == 0.f);
        expect(Near(world.Masses[body.Body].InvInertiaLocal, rbp::float3{0.5f, 0, 0.125f}));
        expect(Near(rbp::Rotate(body.Frame.Orientation, rbp::float3{1, 0, 0}), rbp::float3{0, 1, 0}));
        expect(Near(ToRbp(RbpNodePose(world.Poses[body.Body], body.Frame).P), rbp::float3{0, 0, 0}));
    };

    "RBP compound mass uses collider volume and offset"_test = [] {
        rbp::mtl::Context context;
        rbp::World world{context};
        const auto a = world.AddShape(BoxShape(rbp::float3{0.5f, 0.5f, 0.5f}, rbp::At({-2, 0, 0})));
        const auto b = world.AddShape(BoxShape(rbp::float3{0.5f, 0.5f, 0.5f}, rbp::At({2, 0, 0})));
        const PhysicsMotion motion{.Mass = 6};
        const auto body = BuildRbpBody(world, std::array{a, b}, Transform{}, &motion);
        expect(Near(body.Frame.Position, rbp::float3{0, 0, 0}));
        CheckTensor(Tensor(world, body), {1, 0, 0, 0, 25, 0, 0, 0, 25});
    };

    "RBP kinematic and collider-free bodies retain authored velocity"_test = [] {
        rbp::mtl::Context context;
        rbp::World world{context};
        const PhysicsVelocity velocity{.Linear = {1, 2, 3}, .Angular = {4, 5, 6}};
        for (bool kinematic : {false, true}) {
            const PhysicsMotion motion{.IsKinematic = kinematic, .Mass = 2};
            const auto body = BuildRbpBody(world, {}, Transform{}, &motion, &velocity);
            expect(body.Shape == rbp::NoIndex);
            expect(world.Masses[body.Body].InvMass == (kinematic ? 0.f : 0.5f));
            expect(rbp::Turns(world.Masses[body.Body]) == !kinematic);
            expect(Near(world.Velocities[body.Body].Linear, rbp::float3{1, 2, 3}));
            expect(Near(world.Velocities[body.Body].Angular, rbp::float3{4, 5, 6}));
        }
        const auto body = BuildRbpBody(world, {}, Transform{}, nullptr, &velocity);
        expect(!rbp::Moves(world.Masses[body.Body]));
        expect(Near(world.Velocities[body.Body].Linear, rbp::float3{0, 0, 0}));
    };

    "RBP body failures release owned colliders"_test = [] {
        rbp::mtl::Context context;
        rbp::World world{context, {.Bodies = 1, .Shapes = 4}};
        const auto shape = world.AddShape(BoxShape(rbp::float3{1, 1, 1}));
        world.AddBody({});
        const PhysicsMotion motion;
        expect(throws([&] { BuildRbpBody(world, std::array{shape}, Transform{}, &motion); }));
        expect(world.ShapeCount() == 1_u);
        const PhysicsMotion invalid{.InertiaDiagonal = vec3{-1, 1, 1}};
        expect(throws([&] { BuildRbpBody(world, std::array{shape}, Transform{}, &invalid); }));
        expect(world.ShapeCount() == 1_u);
        expect(world.RemoveShape(shape));
    };

    "RBP shape replacements preserve identity and release temporary geometry"_test = [] {
        rbp::mtl::Context context;
        rbp::World world{context, {.Bodies = 1, .Shapes = 6, .CompoundChildren = 2}};
        const PhysicsMotion motion{.Mass = 12};
        auto source = world.AddShape(BoxShape(rbp::float3{1, 2, 3}));
        auto body = BuildRbpBody(world, std::array{source}, Transform{}, &motion);
        world.RemoveShape(source);
        const auto id = world.IdOf(body.Body);
        for (int i = 0; i < 16; ++i) {
            const float half_x = i % 2 ? 1 : 2;
            source = world.AddShape(BoxShape(rbp::float3{half_x, 2, 3}, rbp::At({0.2f, 0, 0})));
            const auto old_shape = body.Shape;
            const PhysicsMotion invalid{.InertiaDiagonal = vec3{-1, 1, 1}};
            expect(throws([&] { BuildRbpBody(world, std::array{source}, Transform{}, &invalid, nullptr, false, &body); }));
            expect(throws([&] { BuildRbpBody(world, std::array{source, source}, Transform{}, &motion, nullptr, false, &body); }));
            expect(world.BodyShapes[body.Body] == old_shape);
            body = BuildRbpBody(world, std::array{source}, Transform{}, &motion, nullptr, false, &body);
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

    "RBP tapered capsules settle with an offset centre of mass"_test = [] {
        rbp::mtl::Context context;
        rbp::Solver solver{context};
        for (float dt : {1.f / 240, 1.f / 600}) {
            rbp::World world{context};
            const auto floor = world.AddShape(BoxShape(rbp::float3{5, 0.1f, 5}));
            BuildRbpBody(world, std::array{floor}, Transform{}, nullptr);
            const auto shape = BuildRbpShape(world, physics::Capsule{.Height = 0.5498f, .RadiusTop = 0.25f, .RadiusBottom = 0.389f}, nullptr, vec3{1}, rbp::IdentityPose);
            const Transform node{.P = {3.0006042f, 2.5743825f, 0}, .R = {0.7869047f, 0.2398302f, -0.1657568f, -0.5438632f}};
            const PhysicsMotion motion{.Mass = 1, .CenterOfMass = vec3{0, -0.35f, 0}};
            const auto body = BuildRbpBody(world, std::array{shape}, node, &motion);
            auto previous = world.Poses[body.Body].Position;
            float largest_step = 0;
            solver.Advance(world, {.DeltaTime = dt}, uint32_t(12 / dt), {}, [&](const rbp::StepResult &step) {
                const auto position = step.Poses[body.Body].Position;
                largest_step = std::max(largest_step, float(simd::length(position - previous)));
                previous = position;
            });
            const auto sampled = RbpNodePose(world.Poses[body.Body], body.Frame);
            expect(largest_step < 0.04f) << largest_step;
            expect(sampled.P.y > 0.6f && sampled.P.y < 0.8f);
            expect((sampled.R * vec3{0, 1, 0}).y > 0.9f);
            expect(simd::length(world.Velocities[body.Body].Linear) < 0.02f);
            expect(simd::length(world.Velocities[body.Body].Angular) < 0.02f);
        }
    };

    "RBP authored centre keeps collision geometry at the node pose"_test = [] {
        rbp::mtl::Context context;
        rbp::World world{context};
        rbp::Solver solver{context};
        world.TrackContacts = true;
        rbp::Shape plane{};
        plane.Normal = {0, 1, 0};
        plane.Kind = rbp::ShapePlane;
        rbp::BodyDesc ground{};
        ground.Shape = world.AddShape(plane);
        world.AddBody(ground);
        const auto box = world.AddShape(BoxShape(rbp::float3{0.5f, 0.5f, 0.5f}));
        const PhysicsMotion motion{.Mass = 2, .CenterOfMass = vec3{0, 0.3f, 0}, .InertiaDiagonal = vec3{1, 2, 3}, .InertiaOrientation = numeric::AngleAxis(0.4f, vec3{1, 0, 0})};
        const Transform node{.P = {2, 0.5f, 3}};
        const auto body = BuildRbpBody(world, std::array{box}, node, &motion);
        solver.Advance(world, {}, 60);
        world.TakeContactChanges();
        solver.Step(world);
        const auto sampled = RbpNodePose(world.Poses[body.Body], body.Frame);
        expect(Near(ToRbp(sampled.P), ToRbp(node.P), 0.005f));
        float support = 0;
        const auto contacts = world.TakeContactChanges();
        expect(!contacts.empty());
        for (const auto &contact : contacts) support += contact.ForceOnA().y;
        expect(std::abs(support - 2 * 9.81f) < 1.f) << support;
    };
};
} // namespace

int main() { return RunSuites(); }
