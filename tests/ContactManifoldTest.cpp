#include "Near.h"
#include "RunSuites.h"
#include "Solver.h"

#include <boost/ut.hpp>

#include <map>
#include <ranges>
#include <tuple>

using namespace boost::ut;
namespace {
using Key = std::tuple<uint64_t, uint32_t, uint32_t>;
struct Manifold {
    rbp::float3 Normal{};
    std::vector<rbp::float3> Points;
    float Impulse{};
};
struct Scene {
    rbp::mtl::Context Context;
    rbp::World World{Context, {.Bodies = 64, .Shapes = 128}};
    rbp::Solver Solver{Context};
    std::map<Key, Manifold> Manifolds;
    static constexpr float Dt = 1.f / 60;
    Scene() { World.TrackContacts = true; }
    rbp::Index Box(rbp::float3 half, rbp::Pose local = rbp::IdentityPose) {
        rbp::Shape shape{};
        shape.Kind = rbp::ShapeBox;
        shape.HalfExtents = half;
        shape.Local = local;
        return World.AddShape(shape);
    }
    rbp::Index Add(rbp::Index shape, rbp::float3 at, bool dynamic, float gravity = 1) {
        rbp::BodyDesc body{};
        body.Shape = shape;
        body.Pose = rbp::At(at);
        body.Density = dynamic ? 1000 : 0;
        body.GravityScale = gravity;
        return World.AddBody(body);
    }
    void Floor() { Add(Box(rbp::float3{50, 1, 50}), rbp::float3{0, -1, 0}, false); }
    void UnitBox() { Add(Box(rbp::float3{0.5f, 0.5f, 0.5f}), rbp::float3{0, 0.5f, 0}, true); }
    void Legs(std::span<const rbp::float3> centers, rbp::float3 half, bool dynamic) {
        std::vector<rbp::Index> shapes;
        for (auto at : centers) shapes.push_back(Box(half, rbp::At(at)));
        rbp::Pose frame;
        const auto compound = World.AddCompound(shapes, &frame);
        rbp::BodyDesc body{};
        body.Shape = compound;
        body.Pose = frame;
        body.Density = dynamic ? 1000 : 0;
        World.AddBody(body);
    }
    void Step() {
        Solver.Step(World);
        Manifolds.clear();
        for (const auto &c : World.TakeContactChanges()) {
            if (c.Kind == rbp::ContactRemoved) continue;
            auto &m = Manifolds[{c.Children, c.SubShapeA, c.SubShape}];
            m.Normal = c.Normal;
            m.Points.push_back(c.SideA.Point);
            m.Impulse += -c.Lambda.x * Dt;
        }
    }
    void Settle() {
        Solver.Advance(World, {}, 119);
        World.TakeContactChanges();
        Step();
    }
    float Impulse() const {
        float total = 0;
        for (const auto &[key, m] : Manifolds) total += m.Impulse;
        return total;
    }
};
const suite Manifolds = [] {
    "a box reports four spread points carrying its weight"_test = [] {
        Scene s;
        s.Floor();
        s.UnitBox();
        s.Settle();
        expect(s.Manifolds.size() == 1_ul);
        if (s.Manifolds.empty()) return;
        const auto &m = s.Manifolds.begin()->second;
        expect(m.Points.size() == 4_ul);
        for (auto p : m.Points) {
            expect(Near(std::abs(p.x), 0.5f, 0.01f));
            expect(Near(std::abs(p.z), 0.5f, 0.01f));
        }
        expect(Near(s.Impulse(), 1000 * 9.81f * Scene::Dt, 0.1f));
    };
    "separate compound feet retain all contact regions and their load"_test = [] {
        for (int count : {4, 6}) {
            Scene s;
            s.Floor();
            std::vector<rbp::float3> centers;
            for (int i = 0; i < count / 2; ++i)
                for (float z : {-0.4f, 0.4f}) centers.push_back(rbp::float3{float(i) * 0.4f - float(count / 2 - 1) * 0.2f, 0.3f, z});
            s.Legs(centers, rbp::float3{0.1f, 0.3f, 0.1f}, true);
            s.Settle();
            expect(s.Manifolds.size() == size_t(count));
            expect(Near(s.Impulse(), float(count) * 0.2f * 0.6f * 0.2f * 1000 * 9.81f * Scene::Dt, 0.15f));
        }
    };
    "opposed faces retain separate normals"_test = [] {
        Scene s;
        const rbp::float3 centers[]{{-0.55f, 0, 0}, {0.55f, 0, 0}};
        s.Legs(centers, rbp::float3{0.1f, 1, 1}, false);
        s.Add(s.Box(rbp::float3{0.5f, 0.5f, 0.5f}), rbp::float3{0, 0, 0}, true, 0);
        s.Settle();
        expect(s.Manifolds.size() == 2_ul);
        if (s.Manifolds.size() != 2) return;
        const auto &a = s.Manifolds.begin()->second, &b = std::next(s.Manifolds.begin())->second;
        expect(simd::dot(a.Normal, b.Normal) < -0.99f);
        if (a.Impulse + b.Impulse > 0) expect(simd::length((a.Normal * a.Impulse + b.Normal * b.Impulse) / (a.Impulse + b.Impulse)) < 0.2f);
    };
    "resting triangle manifolds keep their keys across steps"_test = [] {
        Scene s;
        std::vector<rbp::float3> vertices;
        std::vector<uint32_t> indices;
        for (float x = -2; x < 2; x += 0.25f)
            for (float z = -2; z < 2; z += 0.25f) {
                const auto first = uint32_t(vertices.size());
                vertices.insert(vertices.end(), {rbp::float3{x, 0, z}, rbp::float3{x, 0, z + 0.25f}, rbp::float3{x + 0.25f, 0, z + 0.25f}, rbp::float3{x + 0.25f, 0, z}});
                indices.insert(indices.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
            }
        s.Add(s.World.AddMesh(vertices, indices), rbp::float3{0, 0, 0}, false);
        s.UnitBox();
        s.Settle();
        const auto original = s.Manifolds | std::views::keys | std::ranges::to<std::vector>();
        expect(!original.empty());
        for (int i = 0; i < 30; ++i) {
            s.Step();
            expect(std::ranges::equal(s.Manifolds | std::views::keys, original));
        }
    };
};
} // namespace
int main() { return RunSuites(); }
