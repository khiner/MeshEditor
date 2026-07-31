// Pins what the physics step reports to the audio system: one sustained contact per manifold, with the identity,
// geometry, and load each carries. ContactManifoldTest pins Jolt's input to that step, this its output.

#include "Reactive.h"
#include "physics/PhysicsContact.h"
#include "physics/PhysicsSystem.h"
#include "physics/PhysicsTypes.h"
#include "scene/SceneGraph.h"
#include "scene/WorldTransform.h"

#include "Near.h"

#include <entt/entity/registry.hpp>

#include <boost/ut.hpp>

#include <algorithm>
#include <ranges>
#include <vector>

using namespace boost::ut;

namespace {
constexpr float Fps{60};
constexpr int RangeEnd{240};

// Just enough scene for physics: a viewport carrying the simulation settings, and bodies built by the
// reactive handlers physics::Init registers.
struct Scene {
    entt::registry R;
    entt::entity Viewport{};
    int Frame{0};

    Scene() {
        physics::Init(R);
        Viewport = R.create();
        R.emplace<PhysicsSimulationSettings>(Viewport);
    }
    ~Scene() { physics::Deinit(R); }

    // A body at `position`, static without motion and dynamic with it. Colliders on children make it a compound.
    entt::entity AddBody(vec3 position, std::optional<PhysicsShape> shape, std::optional<PhysicsMotion> motion, vec3 velocity = {}) {
        const auto e = R.create();
        R.emplace<Transform>(e, Transform{.P = position});
        R.emplace<WorldTransform>(e, Transform{.P = position});
        R.emplace<SceneNode>(e);
        R.emplace<ParentInverse>(e); // The world-transform walk reads it on every parented node.
        R.emplace<ReportContacts>(e);
        if (shape) R.emplace<ColliderShape>(e, ColliderShape{.Shape = *shape});
        if (motion) {
            R.emplace<PhysicsMotion>(e, *motion);
            R.emplace<PhysicsVelocity>(e, PhysicsVelocity{.Linear = velocity});
        }
        return e;
    }

    // Link `child` under `parent`, leaving the world transforms these scenes author directly.
    void Parent(entt::entity child, entt::entity parent) {
        auto &pn = R.get<SceneNode>(parent);
        R.get<SceneNode>(child).Parent = parent;
        R.get<SceneNode>(child).NextSibling = pn.FirstChild;
        pn.FirstChild = child;
    }

    // Build or update the bodies the components describe, as ProcessComponentEvents does in the app.
    void Sync() {
        for (auto &handler : R.ctx().get<std::vector<ComponentEventHandler>>()) handler(R);
    }

    void Step(int frames = 1) {
        for (int i = 0; i < frames; ++i) {
            physics::AdvancePlayback(R, Viewport, Frame, Frame + 1, 0, RangeEnd, Fps, false, false);
            ++Frame;
        }
    }

    // Frames the playhead does not move through, as a paused scene runs.
    void Hold(int frames = 1) {
        for (int i = 0; i < frames; ++i) physics::AdvancePlayback(R, Viewport, Frame, Frame, 0, RangeEnd, Fps, false, false);
    }

    const std::vector<SustainedContact> &Contacts() const { return R.ctx().get<const PhysicsSustainedContacts>().Active; }
    uint64_t ContactStep() const { return R.ctx().get<const PhysicsSustainedContacts>().Step; }
    const std::vector<ContactImpact> &Impacts() const { return R.ctx().get<const PhysicsContactImpacts>().Events; }
};

PhysicsShape Box(vec3 size) { return physics::Box{size}; }

// A floor wide enough that nothing reaches its edges, with its top face at y = 0.
entt::entity AddFloor(Scene &s) { return s.AddBody({0, -1, 0}, Box({100, 2, 100}), {}); }

// A unit box come to rest on a floor.
entt::entity AddRestingBox(Scene &s, PhysicsMotion motion = {}) {
    AddFloor(s);
    const auto box = s.AddBody({0, 0.5f, 0}, Box({1, 1, 1}), motion);
    s.Sync();
    s.Step(60);
    return box;
}

// Give a collider its own physics material, as a compound's feet each carry.
void SetFriction(Scene &s, entt::entity collider, float friction) {
    const auto material = s.R.create();
    s.R.emplace<PhysicsMaterial>(material, PhysicsMaterial{.DynamicFriction = friction});
    s.R.emplace<ColliderMaterial>(collider, ColliderMaterial{.PhysicsMaterialEntity = material});
}

const SustainedContact *ContactWithId(const Scene &s, uint64_t id) {
    const auto &active = s.Contacts();
    const auto it = std::ranges::find(active, id, &SustainedContact::Id);
    return it == active.end() ? nullptr : &*it;
}

// The side belonging to `e` first, the other second, since which side a body lands on is up to the pair.
std::pair<const SustainedContactSide &, const SustainedContactSide &> SidesOf(const SustainedContact &c, entt::entity e) {
    const bool first = c.Sides.front().Entity == e;
    return {first ? c.Sides.front() : c.Sides.back(), first ? c.Sides.back() : c.Sides.front()};
}

} // namespace

int main() {
    "a box resting on a floor is one contact"_test = [] {
        Scene s;
        AddRestingBox(s);

        // The four corners share a normal, so they are one manifold and one contact.
        expect(s.Contacts().size() == 1_ul);
        if (s.Contacts().size() != 1) return;
        const auto &c = s.Contacts().front();
        expect(Near(c.Point.y, 0.f, 0.05f));
        expect(Near(c.Normal.y, -1.f, 0.01f)); // directed into the second side, which is the floor
    };

    "a resting contact reports the load it carries and no travel"_test = [] {
        Scene s;
        AddRestingBox(s, PhysicsMotion{.Mass = 2.f});

        expect(s.Contacts().size() == 1_ul);
        if (s.Contacts().size() != 1) return;
        const auto &c = s.Contacts().front();
        expect(Near(c.NormalForce, 2.f * 9.81f, 0.1f));
        expect(c.Friction > 0.f); // The pair's combined coefficient.
        // Neither surface travels under a body at rest.
        expect(glm::length(c.Sides.front().SweepVelocity) < 1e-3f);
        expect(glm::length(c.Sides.back().SweepVelocity) < 1e-3f);
    };

    "a sliding box sweeps the floor and not itself"_test = [] {
        Scene s;
        AddFloor(s);
        const auto box = s.AddBody({0, 0.5f, 0}, Box({1, 1, 1}), PhysicsMotion{}, vec3{2, 0, 0});
        s.Sync();
        s.Step(10);

        expect(s.Contacts().size() == 1_ul);
        if (s.Contacts().size() != 1) return;
        const auto &c = s.Contacts().front();
        const auto [own, floor] = SidesOf(c, box);
        // The same material region of the box stays in contact while the floor streams past it.
        expect(glm::length(own.SweepVelocity) < 0.05f);
        expect(glm::length(floor.SweepVelocity) > 0.5f);
        expect(glm::length(c.Slip) > 0.5f);
        // The floor's sweep runs along the direction of travel, which a speed alone could not give.
        const auto floor_dir = glm::normalize(floor.SweepVelocity);
        expect(std::abs(floor_dir.x) > 0.99f);
        expect(std::abs(floor_dir.y) < 0.05f);
    };

    "a body wedged between opposed faces of one body is two contacts"_test = [] {
        Scene s;
        // Two walls of one kinematic body, their inner faces 0.9 apart, squeezing a box 1.0 wide.
        // The box slides along the slot, which also keeps it from sleeping.
        const auto walls = s.AddBody({0, 0, 0}, {}, PhysicsMotion{.IsKinematic = true});
        s.Parent(s.AddBody({-0.55f, 0, 0}, Box({0.2f, 2, 2}), {}), walls);
        s.Parent(s.AddBody({0.55f, 0, 0}, Box({0.2f, 2, 2}), {}), walls);
        s.AddBody({0, 0, 0}, Box({1, 1, 1}), PhysicsMotion{.GravityFactor = 0}, vec3{0, 0, 1});
        s.Sync();
        s.Step(10);

        // The two normals oppose, so a pair merged into one contact would average them to nothing and report silence.
        expect(s.Contacts().size() == 2_ul);
        if (s.Contacts().size() != 2) return;
        const auto &a = s.Contacts().front();
        const auto &b = s.Contacts().back();
        expect(a.Id != b.Id);
        expect(std::abs(glm::dot(a.Normal, b.Normal) + 1.f) < 0.01f);
        expect(a.NormalForce > 0.f);
        expect(b.NormalForce > 0.f);
    };

    "a box landing flat is struck at every corner it lands on"_test = [] {
        Scene s;
        AddFloor(s);
        // Dropped from just clear of the floor, so it lands within the first few frames and nothing drains it first.
        const auto box = s.AddBody({0, 0.6f, 0}, Box({1, 1, 1}), PhysicsMotion{.Mass = 2.f});
        s.Sync();
        for (int i = 0; i < 30 && s.Impacts().empty(); ++i) s.Step();

        // Each point produces an impact for both bodies, so count only the box's.
        const auto own = s.Impacts() | std::views::filter([box](const auto &c) { return c.Entity == box; }) | std::ranges::to<std::vector>();
        expect(own.size() > 1_ul); // A flat landing is more than one point, not one centre of pressure.
        expect(own.size() <= 4_ul); // Jolt's manifold reduction caps a face at four points.
        float total = 0;
        for (const auto &c : own) {
            expect(c.Impulse > 0.f);
            expect(c.Speed > 0.f);
            expect(Near(c.Point.y, 0.f, 0.1f)); // every point sits on the floor, not at one averaged centre
            total += c.Impulse;
        }
        // The points split one landing rather than each repeating it, so they sum to the pair's impulse.
        expect(total > 0.f && total < 2.f * 9.81f);
    };

    "each side names the collider node that is touching"_test = [] {
        Scene s;
        // A compound standing on two feet, each on its own floor, so the two feet touch as separate pairs.
        const auto floor_a = s.AddBody({-1, -1, 0}, Box({1, 2, 1}), {});
        const auto floor_b = s.AddBody({1, -1, 0}, Box({1, 2, 1}), {});
        const auto body = s.AddBody({0, 0.5f, 0}, {}, PhysicsMotion{});
        const auto foot_a = s.AddBody({-1, 0.5f, 0}, Box({0.5f, 1, 0.5f}), {});
        const auto foot_b = s.AddBody({1, 0.5f, 0}, Box({0.5f, 1, 0.5f}), {});
        s.Parent(foot_a, body);
        s.Parent(foot_b, body);
        // A slippery foot and a grippy one, combined separately by the pair each foot makes.
        SetFriction(s, floor_a, 0.2f);
        SetFriction(s, floor_b, 0.2f);
        SetFriction(s, foot_a, 0.1f);
        SetFriction(s, foot_b, 1.f);
        s.Sync();
        s.Step(10);

        expect(s.Contacts().size() == 2_ul);
        if (s.Contacts().size() != 2) return;
        // Both contacts are between the same two bodies, so only the collider nodes tell them apart.
        for (const auto &c : s.Contacts()) {
            const auto [own, floor] = SidesOf(c, body);
            expect(own.Entity == body);
            const bool is_a = own.ColliderEntity == foot_a;
            expect(is_a || own.ColliderEntity == foot_b);
            expect(floor.ColliderEntity == (is_a ? floor_a : floor_b));
            // The touching foot's material decides the pair's friction, so the two feet report different coefficients.
            expect(Near(c.Friction, is_a ? 0.15f : 0.6f, 0.01f));
        }
    };

    "a manifold keeps its id while it lasts"_test = [] {
        Scene s;
        AddRestingBox(s);

        expect(s.Contacts().size() == 1_ul);
        if (s.Contacts().empty()) return;
        const auto id = s.Contacts().front().Id;
        for (int i = 0; i < 30; ++i) {
            s.Step();
            expect(s.Contacts().size() == 1_ul);
            expect(ContactWithId(s, id) != nullptr);
        }
    };

    "a contact that stops touching stops being reported"_test = [] {
        Scene s;
        const auto box = AddRestingBox(s);
        expect(s.Contacts().size() == 1_ul);

        // Lift the box clear of the floor. A pose change rebuilds nothing, so the body follows its transform.
        s.R.patch<Transform>(box, [](auto &t) { t.P = vec3{0, 5, 0}; });
        s.R.patch<WorldTransform>(box, [](auto &t) { t.P = vec3{0, 5, 0}; });
        s.Sync();
        s.Step(5);

        expect(s.Contacts().empty());
    };

    "a resting contact reports no new step while the playhead is parked"_test = [] {
        Scene s;
        AddRestingBox(s);
        expect(s.Contacts().size() == 1_ul);

        // The set stands, since only a step rebuilds it, but its step number does not move.
        const auto step = s.ContactStep();
        s.Hold(5);
        expect(s.Contacts().size() == 1_ul);
        expect(s.ContactStep() == step);

        // Resuming reports a step again, so the same resting contact drives audio once more.
        s.Step();
        expect(s.ContactStep() > step);
        expect(s.Contacts().size() == 1_ul);
    };
}
