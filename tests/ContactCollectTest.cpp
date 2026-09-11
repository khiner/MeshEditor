
#include "Reactive.h"
#include "RunSuites.h"
#include "mesh/MeshBatch.h"
#include "mesh/MeshComponents.h"
#include "mesh/Primitives.h"
#include "physics/PhysicsContact.h"
#include "physics/PhysicsSystem.h"
#include "physics/PhysicsTypes.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "scene/WorldTransform.h"
#include "viewport/ViewportEvents.h"

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

struct Scene {
    entt::registry R;
    entt::entity Viewport{};
    int Frame{0};
    uint32_t BodyCreations = 0;
    void BodyCreated(entt::registry &, entt::entity) { ++BodyCreations; }

    Scene() {
        physics::Init(R);
        R.on_construct<PhysicsBodyHandle>().connect<&Scene::BodyCreated>(*this);
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

    // Link the translated test bodies while preserving the authored world pose.
    void Parent(entt::entity child, entt::entity parent) {
        auto &pn = R.get<SceneNode>(parent);
        R.get<SceneNode>(child).Parent = parent;
        R.get<SceneNode>(child).NextSibling = pn.FirstChild;
        pn.FirstChild = child;
        R.get<Transform>(child).P -= R.get<WorldTransform>(parent).P;
    }

    // Build or update the bodies the components describe, as ProcessComponentEvents does in the app.
    void Sync() {
        for (auto &handler : R.ctx().get<std::vector<ComponentEventHandler>>()) handler.Apply(R);
        physics::AdvancePlayback(R, Viewport, Frame, Frame, 0, RangeEnd, Fps, false);
        for (auto &&[id, storage] : R.storage()) {
            if (storage.info() == entt::type_id<entt::reactive>()) storage.clear();
        }
    }

    void Step(int frames = 1) {
        for (int i = 0; i < frames; ++i) {
            physics::AdvancePlayback(R, Viewport, Frame, Frame + 1, 0, RangeEnd, Fps, false);
            ++Frame;
        }
    }

    void Hold(int frames = 1) {
        for (int i = 0; i < frames; ++i) physics::AdvancePlayback(R, Viewport, Frame, Frame, 0, RangeEnd, Fps, false);
    }

    const std::vector<SustainedContact> &Contacts() const { return R.ctx().get<const PhysicsSustainedContacts>().Active; }
    uint64_t ContactStep() const { return R.ctx().get<const PhysicsSustainedContacts>().Step; }
    const std::vector<ContactImpact> &Impacts() const { return R.ctx().get<const PhysicsContactImpacts>().Events; }
};

PhysicsShape Box(vec3 size) { return physics::Box{size}; }
PhysicsShape Sphere(float radius) { return physics::Sphere{radius}; }

// A floor wide enough that nothing reaches its edges, with its top face at y = 0.
entt::entity AddFloor(Scene &s) { return s.AddBody({0, -1, 0}, Box({100, 2, 100}), {}); }

entt::entity AddRestingBox(Scene &s, PhysicsMotion motion = {}) {
    AddFloor(s);
    const auto box = s.AddBody({0, 0.5f, 0}, Box({1, 1, 1}), motion);
    s.Sync();
    s.Step(60);
    return box;
}

void SetMaterial(Scene &s, entt::entity collider, const PhysicsMaterial &material) {
    const auto e = s.R.create();
    s.R.emplace<PhysicsMaterial>(e, material);
    s.R.emplace<ColliderMaterial>(collider, ColliderMaterial{.PhysicsMaterialEntity = e});
}

const SustainedContact *ContactWithId(const Scene &s, uint64_t id) {
    const auto &active = s.Contacts();
    const auto it = std::ranges::find(active, id, &SustainedContact::Id);
    return it == active.end() ? nullptr : &*it;
}

const SustainedContact *OnlyContact(const Scene &s) {
    expect(s.Contacts().size() == 1_ul);
    return s.Contacts().size() == 1 ? &s.Contacts().front() : nullptr;
}

// The impacts belonging to one body. Each contact point produces an impact for both bodies of the pair.
std::vector<ContactImpact> ImpactsOn(const Scene &s, entt::entity e) {
    return s.Impacts() | std::views::filter([e](const auto &c) { return c.Entity == e; }) | std::ranges::to<std::vector>();
}

std::pair<const SustainedContactSide &, const SustainedContactSide &> SidesOf(const SustainedContact &c, entt::entity e) {
    const bool first = c.Sides.front().Entity == e;
    return {first ? c.Sides.front() : c.Sides.back(), first ? c.Sides.back() : c.Sides.front()};
}

} // namespace
int main() {
    "resting contacts report geometry and support without surface travel"_test = [] {
        for (bool sphere : {false, true}) {
            Scene s;
            AddFloor(s);
            s.AddBody({0, 0.5f, 0}, sphere ? Sphere(0.5f) : Box({1, 1, 1}), PhysicsMotion{.Mass = 2.f});
            s.Sync();
            s.Step(60);
            const auto *c = OnlyContact(s);
            if (!c) continue;
            expect(Near(c->Point.y, 0.f, 0.05f));
            expect(Near(c->Normal.y, -1.f, 0.01f));
            expect(sphere ? c->NominalArea == 0.f : Near(c->NominalArea, 1.f, 0.05f));
            expect(Near(c->NormalForce, 2.f * 9.81f, 0.1f));
            expect(c->Friction > 0.f);
            for (const auto &side : c->Sides) expect(numeric::Length(side.SweepVelocity) < 1e-3f);
        }
    };

    "a sliding box sweeps the floor and not itself"_test = [] {
        Scene s;
        AddFloor(s);
        const auto box = s.AddBody({0, 0.5f, 0}, Box({1, 1, 1}), PhysicsMotion{}, vec3{2, 0, 0});
        s.Sync();
        s.Step(10);

        const auto *c = OnlyContact(s);
        if (!c) return;
        const auto [own, floor] = SidesOf(*c, box);
        // The same material region of the box stays in contact while the floor streams past it.
        expect(numeric::Length(own.SweepVelocity) < 0.05f);
        expect(numeric::Length(floor.SweepVelocity) > 0.5f);
        expect(numeric::Length(c->Slip) > 0.5f);
        // The floor's sweep runs along the direction of travel, which a speed alone could not give.
        const auto floor_dir = numeric::Normalize(floor.SweepVelocity);
        expect(std::abs(floor_dir.x) > 0.99f);
        expect(std::abs(floor_dir.y) < 0.05f);
    };

    "a body wedged between opposed faces of one body is two contacts"_test = [] {
        Scene s;
        // A shallow V supports the box on two nearly opposed faces of one body.
        // Gravity supplies both normal loads; sliding along the slot keeps the box awake.
        const auto walls = s.AddBody({0, 0, 0}, {}, PhysicsMotion{.IsKinematic = true});
        constexpr float angle = 0.04f;
        const float offset = 0.5f + (0.1f + 0.5f * std::sin(angle)) / std::cos(angle);
        const PhysicsMaterial frictionless{.StaticFriction = 0, .DynamicFriction = 0};
        for (float side : {-1.f, 1.f}) {
            const auto wall = s.AddBody({side * offset, 0, 0}, Box({0.2f, 2, 2}), {});
            s.R.get<Transform>(wall).R = numeric::AngleAxis(-side * angle, vec3{0, 0, 1});
            SetMaterial(s, wall, frictionless);
            s.Parent(wall, walls);
        }
        const auto box = s.AddBody({0, 0, 0}, Box({1, 1, 1}), PhysicsMotion{.Mass = 1}, vec3{0, 0, 1});
        SetMaterial(s, box, frictionless);
        s.Sync();
        s.Step(10);

        // The two normals oppose, so a pair merged into one contact would average them to nothing and report silence.
        expect(s.Contacts().size() == 2_ul);
        if (s.Contacts().size() != 2) return;
        const auto &a = s.Contacts().front();
        const auto &b = s.Contacts().back();
        expect(a.Id != b.Id);
        expect(std::abs(numeric::Dot(a.Normal, b.Normal) + 1.f) < 0.01f);
        expect(a.NormalForce > 0.f);
        expect(b.NormalForce > 0.f);
        const float support = std::abs(a.Normal.y) * a.NormalForce + std::abs(b.Normal.y) * b.NormalForce;
        expect(Near(support, 9.81f, 0.2f));
    };

    "a box landing flat is struck at every corner it lands on"_test = [] {
        Scene s;
        AddFloor(s);
        const auto box = s.AddBody({0, 0.6f, 0}, Box({1, 1, 1}), PhysicsMotion{.Mass = 2.f});
        s.Sync();
        for (int i = 0; i < 30 && s.Impacts().empty(); ++i) s.Step();

        const auto own = ImpactsOn(s, box);
        expect(own.size() > 1_ul); // A flat landing is more than one point, not one centre of pressure.
        expect(own.size() <= 4_ul); // A face manifold is reduced to at most four points.
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

    "a body authored at rest lands as silence"_test = [] {
        Scene s;
        AddFloor(s);
        // Authored touching, so the first solved impulse is pure support and the excess convention renders nothing.
        s.AddBody({0, 0.5f, 0}, Box({1, 1, 1}), PhysicsMotion{.Mass = 50.f});
        s.Sync();
        s.Step(60);
        expect(s.Impacts().empty());
    };

    "a bouncing body reports the strike its bounce arrested"_test = [] {
        Scene s;
        const auto floor = AddFloor(s);
        constexpr float restitution = 0.9f, mass = 0.5f;
        const auto ball = s.AddBody({0, 1.f, 0}, Sphere(0.5f), PhysicsMotion{.Mass = mass});
        for (const auto collider : {floor, ball}) SetMaterial(s, collider, {.Restitution = restitution});
        s.Sync();
        for (int i = 0; i < 30 && s.Impacts().empty(); ++i) s.Step();

        const auto own = ImpactsOn(s, ball);
        expect(own.size() >= 1_ul);
        // A 0.5 m fall arrives at sqrt(2g*0.5), and the strike reports the impulse that arrested and reversed it.
        const float speed = std::sqrt(2.f * 9.81f * 0.5f);
        float total = 0;
        for (const auto &c : own) {
            expect(Near(c.Speed, speed, 0.2f));
            total += c.Impulse;
        }
        expect(Near(total, mass * (1.f + restitution) * speed, 0.3f));
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
        SetMaterial(s, floor_a, {.StaticFriction = 0.2f, .DynamicFriction = 0.2f});
        SetMaterial(s, floor_b, {.StaticFriction = 0.2f, .DynamicFriction = 0.2f});
        SetMaterial(s, foot_a, {.StaticFriction = 0.1f, .DynamicFriction = 0.1f});
        SetMaterial(s, foot_b, {.StaticFriction = 1.f, .DynamicFriction = 1.f});
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

        const auto *first = OnlyContact(s);
        if (!first) return;
        const auto id = first->Id;
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

        // Raise the authored starting pose enough to remain clear at the current timeline frame.
        s.R.patch<Transform>(box, [](auto &t) { t.P = vec3{0, 10, 0}; });
        s.R.patch<WorldTransform>(box, [](auto &t) { t.P = vec3{0, 10, 0}; });
        s.Sync();
        s.Step(5);

        expect(s.Contacts().empty());
    };

    "a resting contact reports no new step while the playhead is parked"_test = [] {
        Scene s;
        AddRestingBox(s);
        expect(s.Contacts().size() == 1_ul);

        const auto step = s.ContactStep();
        s.Hold(5);
        expect(s.Contacts().size() == 1_ul);
        expect(s.ContactStep() == step);

        // Resuming reports a step again, so the same resting contact drives audio once more.
        s.Step();
        expect(s.ContactStep() > step);
        expect(s.Contacts().size() == 1_ul);
    };

    "seeking reconstructs physics from the authored starting pose"_test = [] {
        Scene stepped, sought;
        const auto setup = [](Scene &s) {
            s.R.get<PhysicsSimulationSettings>(s.Viewport).Gravity = {};
            const auto body = s.AddBody({}, Sphere(0.25f), PhysicsMotion{}, {1, 0, 0});
            s.Sync();
            return body;
        };
        const auto a = setup(stepped), b = setup(sought);
        expect(sought.R.get<WorldTransform>(b).P == vec3{});
        stepped.Step(30);
        physics::AdvancePlayback(sought.R, sought.Viewport, 0, 30, 0, RangeEnd, Fps, false);
        expect(stepped.R.get<WorldTransform>(a).P == sought.R.get<WorldTransform>(b).P);
        physics::AdvancePlayback(sought.R, sought.Viewport, 30, 5, 0, RangeEnd, Fps, false);
        physics::AdvancePlayback(sought.R, sought.Viewport, 5, 31, 0, RangeEnd, Fps, false);
        stepped.Step();
        expect(stepped.R.get<WorldTransform>(a).P == sought.R.get<WorldTransform>(b).P);
    };

    "shutter prediction retains the displayed contact frame"_test = [] {
        Scene s;
        AddRestingBox(s);
        const auto step = s.ContactStep();
        const auto force = s.Contacts().front().NormalForce;
        const auto impacts = s.Impacts().size();
        physics::BakeThrough(s.R, s.Viewport, s.Frame + 3, Fps);
        expect(s.ContactStep() == step);
        expect(s.Contacts().front().NormalForce == force);
        expect(s.Impacts().size() == impacts);
        s.Step();
        expect(s.ContactStep() == step + 1);
        expect(physics::BakedThrough(s.R) == std::optional{uint32_t(s.Frame + 2)});
    };

    "nested rigid bodies keep independent cached world poses"_test = [] {
        Scene s;
        const PhysicsMotion motion{.GravityFactor = 0, .LinearDamping = 0, .AngularDamping = 0};
        const auto parent = s.AddBody({0, 0, 0}, {}, motion, {1, 0, 0});
        const auto child = s.AddBody({0, 2, 0}, {}, motion, {-1, 0, 0});
        s.Parent(child, parent);
        s.Sync();
        s.Step(60);
        expect(Near(s.R.get<WorldTransform>(parent).P.x, 1.f, 0.005f));
        expect(Near(s.R.get<WorldTransform>(child).P.x, -1.f, 0.005f));
        physics::SamplePosesAtFrame(s.R, 30.f);
        expect(Near(s.R.get<WorldTransform>(child).P.x, -0.5f, 0.005f));
        physics::SamplePosesAtFrame(s.R, 30.5f);
        expect(Near(s.R.get<WorldTransform>(child).P.x, -30.5f / Fps, 0.005f));
        physics::SamplePosesAtFrame(s.R, -1.f);
        expect(Near(s.R.get<WorldTransform>(child).P.x, 0.f, 0.005f));
        physics::SamplePosesAtFrame(s.R, 61.f);
        expect(Near(s.R.get<WorldTransform>(child).P.x, -1.f, 0.005f));
    };

    "a joint drive measures the connected frame in the owner frame"_test = [] {
        Scene s;
        const auto owner = s.AddBody({0, 0, 0}, {}, {});
        const auto connected = s.AddBody({0, 0, 0}, {}, PhysicsMotion{.GravityFactor = 0, .LinearDamping = 0});
        // A collider-free static attachment is anchored to the world.
        const auto def = s.R.create();
        PhysicsJointDef joint;
        joint.Drives.push_back({.Type = PhysicsDriveType::Linear, .Axis = 0, .PositionTarget = 1, .Stiffness = 100, .Damping = 20});
        s.R.emplace<PhysicsJointDef>(def, joint);
        s.R.emplace<PhysicsJoint>(connected, PhysicsJoint{.ConnectedNode = owner, .JointDefEntity = def});
        s.Sync();
        s.Step(120);
        expect(Near(s.R.get<WorldTransform>(connected).P.x, -1.f, 0.02f));
    };

    "body-local authored velocity follows node rotation"_test = [] {
        Scene s;
        const auto body = s.AddBody({0, 0, 0}, {}, PhysicsMotion{.IsKinematic = true}, {1, 0, 0});
        const auto turn = numeric::AngleAxis(float(std::numbers::pi / 2), vec3{0, 0, 1});
        s.R.patch<Transform>(body, [&](auto &t) { t.R = turn; });
        s.Sync();
        s.Step(60);
        const auto position = s.R.get<WorldTransform>(body).P;
        expect(Near(position.x, 0.f, 0.005f));
        expect(Near(position.y, 1.f, 0.005f));
    };

    "editing initial velocity rebuilds cached motion"_test = [] {
        Scene s;
        const auto body = s.AddBody({}, {}, PhysicsMotion{.IsKinematic = true}, {1, 0, 0});
        s.Sync();
        s.Step(60);
        expect(Near(s.R.get<WorldTransform>(body).P.x, 1.f, 0.005f));
        s.R.patch<PhysicsVelocity>(body, [](auto &v) { v.Linear = {2, 0, 0}; });
        s.Sync();
        s.Step();
        expect(Near(s.R.get<WorldTransform>(body).P.x, 2.f * 61 / Fps, 0.005f));
    };

    "authored edits reuse bodies and match a fresh trajectory at a parked playhead"_test = [] {
        enum Edit { Pose,
                    GroupPose,
                    Velocity,
                    Damping,
                    MotionType,
                    Mass,
                    Center,
                    MassFrame,
                    Material,
                    Filter,
                    Settings,
                    Joint,
                    JointAdd,
                    JointRemove,
                    JointRetarget,
                    JointDisable,
                    JointDeleteDefinition,
                    JointDeleteEndpoint,
                    JointDeleteNode,
                    ReparentBody,
                    ReparentKeepWorld,
                    ReparentCollider,
                    ReparentJoint,
                    ReparentScaled,
                    ReparentOwner,
                    Rename,
                    Unrelated,
                    NoOp,
                    Offset,
                    BulkGeometry,
                    Geometry };
        for (int edit = Pose; edit <= Geometry; ++edit) {
            Scene changed, fresh;
            struct Objects {
                entt::entity Body, Material, Filter, Joint, Unrelated, Group, Floor, Child, ChildGroup, JointNode;
            };
            const auto setup = [edit](Scene &s) {
                const auto floor = s.AddBody({0, -0.5f, 0}, Box(vec3{10, 1, 10}), std::nullopt);
                const auto body = s.AddBody({0, 0.7f, 0}, Box(vec3{1}), PhysicsMotion{}, {0.5f, 0, 0});
                const auto child = s.AddBody({0.75f, 0.7f, 0}, Sphere(0.2f), std::nullopt);
                s.Parent(child, body);
                const auto child_group = s.AddBody({0, 0.7f, 0}, {}, {});
                s.Parent(child_group, body);
                const auto joint_node = s.AddBody({0, 0.7f, 0}, {}, {});
                s.Parent(joint_node, body);
                const auto material = s.R.create(), filter = s.R.create(), joint = s.R.create(), unrelated = s.R.create();
                s.R.emplace<PhysicsMaterial>(material);
                s.R.emplace<CollisionFilter>(filter);
                s.R.emplace<ColliderMaterial>(body, ColliderMaterial{material, null_entity});
                s.R.emplace<ColliderMaterial>(child, ColliderMaterial{material, null_entity});
                PhysicsJointDef definition;
                definition.Drives.push_back({.Type = PhysicsDriveType::Linear, .Mode = PhysicsDriveMode::Acceleration, .Axis = 0, .PositionTarget = 0.25f, .Stiffness = 2, .Damping = 0.5f});
                s.R.emplace<PhysicsJointDef>(joint, definition);
                if (edit != JointAdd) s.R.emplace<PhysicsJoint>(joint_node, PhysicsJoint{edit == JointDeleteEndpoint ? unrelated : floor, joint, true});
                s.R.emplace<Transform>(unrelated);
                s.R.emplace<WorldTransform>(unrelated);
                s.R.emplace<SceneNode>(unrelated);
                const auto group = s.R.create();
                s.R.emplace<Transform>(group);
                s.R.emplace<WorldTransform>(group);
                s.R.emplace<SceneNode>(group);
                s.Parent(body, group);
                return Objects{body, material, filter, joint, unrelated, group, floor, child, child_group, joint_node};
            };
            const auto a = setup(changed), b = setup(fresh);
            changed.Sync();
            changed.Step(24);
            const auto created = changed.BodyCreations;
            const auto contact_step = changed.ContactStep();
            const auto joint_index = edit == JointAdd ? UINT32_MAX : changed.R.get<PhysicsConstraintHandle>(a.JointNode).ConstraintIndex;
            const auto apply = [edit](Scene &s, Objects e) {
                switch (edit) {
                    case Pose: s.R.patch<Transform>(e.Body, [](auto &t) { t.P.x = 0.2f; t.R = numeric::AngleAxis(0.3f, vec3{0, 1, 0}); }); break;
                    case GroupPose: s.R.patch<Transform>(e.Group, [](auto &t) { t.P.x = 0.2f; t.R = numeric::AngleAxis(0.3f, vec3{0, 1, 0}); }); break;
                    case Velocity: s.R.patch<PhysicsVelocity>(e.Body, [](auto &v) { v.Linear.x = 1; }); break;
                    case Damping: s.R.patch<PhysicsMotion>(e.Body, [](auto &m) { m.LinearDamping = 0.7f; m.GravityFactor = 0.8f; }); break;
                    case MotionType: s.R.patch<PhysicsMotion>(e.Body, [](auto &m) { m.IsKinematic = true; }); break;
                    case Center: s.R.patch<PhysicsMotion>(e.Body, [](auto &m) { m.CenterOfMass = vec3{0.1f, 0, 0}; }); break;
                    case Mass: s.R.patch<PhysicsMotion>(e.Body, [](auto &m) { m.Mass = 3; }); break;
                    case MassFrame: s.R.patch<PhysicsMotion>(e.Body, [](auto &m) { m.CenterOfMass = vec3{0.2f, 0, 0}; m.InertiaDiagonal = vec3{0.3f, 0.5f, 0.7f}; m.InertiaOrientation = numeric::AngleAxis(0.4f, vec3{0, 0, 1}); }); break;
                    case Material: s.R.patch<PhysicsMaterial>(e.Material, [](auto &m) { m.StaticFriction = m.DynamicFriction = 0.15f; m.Restitution = 0.3f; }); break;
                    case Filter: s.R.patch<ColliderMaterial>(e.Body, [e](auto &m) { m.CollisionFilterEntity = e.Filter; }); break;
                    case Settings: s.R.patch<PhysicsSimulationSettings>(s.Viewport, [](auto &v) { v.Gravity.y = -5; v.SubstepsPerFrame = 4; v.TimeScale = 0.75f; }); break;
                    case Joint: s.R.patch<PhysicsJointDef>(e.Joint, [](auto &j) { j.Drives[0].PositionTarget = 1; }); break;
                    case JointAdd: s.R.emplace<PhysicsJoint>(e.JointNode, PhysicsJoint{e.Floor, e.Joint, true}); break;
                    case JointRemove: s.R.remove<PhysicsJoint>(e.JointNode); break;
                    case JointRetarget: s.R.patch<PhysicsJoint>(e.JointNode, [e](auto &j) { j.ConnectedNode = e.Unrelated; }); break;
                    case JointDisable: s.R.patch<PhysicsJoint>(e.JointNode, [e](auto &j) { j.ConnectedNode = e.Child; }); break;
                    case JointDeleteDefinition: s.R.destroy(e.Joint); break;
                    case JointDeleteEndpoint: s.R.destroy(e.Unrelated); break;
                    case JointDeleteNode:
                        ClearParent(s.R, e.JointNode);
                        s.R.destroy(e.JointNode);
                        break;
                    case ReparentBody:
                        SetParent(s.R, e.Body, e.Unrelated);
                        s.R.patch<Transform>(e.Unrelated, [](auto &t) { t.P.x = 0.2f; t.R = numeric::AngleAxis(0.3f, vec3{0, 1, 0}); });
                        break;
                    case ReparentKeepWorld: SetParentKeepWorld(s.R, e.Body, e.Unrelated); break;
                    case ReparentCollider: SetParent(s.R, e.Child, e.ChildGroup); break;
                    case ReparentJoint:
                        SetParent(s.R, e.JointNode, e.Floor);
                        s.R.patch<PhysicsJoint>(e.JointNode, [e](auto &j) { j.ConnectedNode = e.Body; });
                        break;
                    case ReparentScaled:
                        SetParent(s.R, e.Body, e.Unrelated);
                        s.R.patch<Transform>(e.Unrelated, [](auto &t) { t.S = vec3{2}; });
                        break;
                    case ReparentOwner: SetParent(s.R, e.Child, e.Floor); break;
                    case Rename: s.R.patch<PhysicsMaterial>(e.Material, [](auto &m) { m.Name = "renamed"; }); break;
                    case Unrelated: s.R.patch<Transform>(e.Unrelated, [](auto &t) { t.P.x = 3; }); break;
                    case NoOp: s.R.patch<PhysicsMotion>(e.Body); break;
                    case Offset: s.R.patch<ColliderShape>(e.Child, [](auto &c) { c.LocalOffset = {0.2f, 0.1f, 0}; }); break;
                    case BulkGeometry:
                        s.R.patch<Transform>(e.Body, [](auto &t) { t.S = vec3{0}; });
                        s.R.patch<ColliderShape>(e.Body, [](auto &c) { c.Shape = Box(vec3{0.6f}); });
                        s.R.patch<ColliderShape>(e.Child, [](auto &c) { c.Shape = Sphere(0.3f); c.LocalOffset = {0.2f, 0, 0}; });
                        s.R.patch<Transform>(e.Child, [](auto &t) { t.P.x = 0.4f; });
                        s.R.patch<ColliderShape>(e.Floor, [](auto &c) { c.Shape = Box(vec3{10, 0.8f, 10}); });
                        s.R.patch<PhysicsMotion>(e.Body, [](auto &m) { m.CenterOfMass = vec3{0.2f, 0, 0}; m.InertiaDiagonal = vec3{1, 2, 3}; m.LinearDamping = 0.3f; });
                        s.R.patch<Transform>(e.Body, [](auto &t) { t.S = {1.1f, 0.9f, 1.2f}; });
                        break;
                    case Geometry: s.R.patch<ColliderShape>(e.Body, [](auto &c) { c.Shape = Box(vec3{0.8f}); }); break;
                }
            };
            apply(changed, a);
            changed.Sync();
            apply(fresh, b);
            if (edit == ReparentKeepWorld) fresh.R.replace<Transform>(b.Body, changed.R.get<Transform>(a.Body));
            fresh.Sync();
            fresh.Step(24);
            expect(edit == ReparentOwner ? changed.BodyCreations > created : changed.BodyCreations == created) << int(edit);
            expect(changed.R.view<const PhysicsConstraintHandle>().size() == fresh.R.view<const PhysicsConstraintHandle>().size()) << int(edit);
            if (edit == Joint || edit == JointRetarget || edit == ReparentJoint) expect(changed.R.get<PhysicsConstraintHandle>(a.JointNode).ConstraintIndex == joint_index);
            if (edit == Rename || edit == Unrelated || edit == NoOp) expect(changed.ContactStep() == contact_step);
            const auto &actual = changed.R.get<WorldTransform>(a.Body), &expected = fresh.R.get<WorldTransform>(b.Body);
            expect(Near(actual.P.x, expected.P.x, 1e-5f) && Near(actual.P.y, expected.P.y, 1e-5f) && Near(actual.P.z, expected.P.z, 1e-5f)) << int(edit) << actual.P.x << actual.P.y << actual.P.z << expected.P.x << expected.P.y << expected.P.z;
            expect(std::abs(numeric::Dot(actual.R, expected.R)) > 0.99999f) << int(edit);
            expect(changed.Contacts().size() == fresh.Contacts().size()) << int(edit);
            for (size_t i = 0; i < std::min(changed.Contacts().size(), fresh.Contacts().size()); ++i) {
                expect(Near(changed.Contacts()[i].NormalForce, fresh.Contacts()[i].NormalForce, 0.001f)) << int(edit);
                expect(changed.Contacts()[i].Friction == fresh.Contacts()[i].Friction) << int(edit);
            }
        }
    };

    "bulk shared mesh edits reuse worlds and match fresh trajectories"_test = [] {
        mtl::Context context;
        mtl::BindlessSet slots{context};
        mtl::BufferContext buffers{context, slots};
        for (bool triangles : {false, true}) {
            Scene changed, fresh;
            struct Objects {
                entt::entity Mesh, Sensor;
                std::array<entt::entity, 4> Bodies;
                std::vector<vec3> Positions;
            };
            const auto setup = [&](Scene &s) {
                s.R.ctx().emplace<MeshStore>(buffers);
                const auto floor = AddFloor(s);
                Objects result;
                result.Mesh = s.R.create();
                auto data = primitive::CreateMesh(primitive::Cuboid{});
                result.Positions = data.Positions;
                const auto mesh = CreateMesh(s.R, {.Data = std::move(data)});
                s.R.emplace<MeshHandle>(result.Mesh, MeshHandle{mesh.StoreId});
                for (uint32_t i = 0; i < result.Bodies.size(); ++i) {
                    result.Bodies[i] = s.AddBody({float(i) * 3, 0.7f, 0}, triangles ? PhysicsShape{physics::TriangleMesh{}} : PhysicsShape{physics::ConvexHull{}}, i == 3 ? std::nullopt : std::optional{PhysicsMotion{}});
                    s.R.patch<ColliderShape>(result.Bodies[i], [&](auto &c) { c.MeshEntity = result.Mesh; });
                }
                result.Sensor = s.AddBody({0.2f, 0.7f, 0}, Sphere(0.2f), {});
                s.R.emplace<TriggerTag>(result.Sensor);
                s.Parent(result.Sensor, result.Bodies[0]);
                const auto definition = s.R.create();
                PhysicsJointDef joint;
                joint.Drives.push_back({.Type = PhysicsDriveType::Linear, .Mode = PhysicsDriveMode::Acceleration, .Axis = 0, .PositionTarget = 0.25f, .Stiffness = 2, .Damping = 0.5f});
                s.R.emplace<PhysicsJointDef>(definition, joint);
                s.R.emplace<PhysicsJoint>(result.Bodies[1], PhysicsJoint{floor, definition, true});
                return result;
            };
            auto a = setup(changed), b = setup(fresh);
            changed.Sync();
            changed.Step(12);
            for (int edit = 0; edit < 8; ++edit) {
                const bool grow = triangles && edit == 6;
                const auto apply = [&](Scene &s, Objects &objects) {
                    if (grow) {
                        const auto cube = primitive::CreateMesh(primitive::Cuboid{});
                        MeshData data;
                        for (int copy = 0; copy < 8; ++copy) {
                            const auto base = uint32_t(data.Positions.size());
                            for (auto p : cube.Positions) data.Positions.push_back(p + vec3{float(copy) * 0.1f, 0, 0});
                            for (uint32_t face = 0; face < cube.FaceCount(); ++face) {
                                std::vector<uint32_t> corners;
                                for (auto vertex : cube.Face(face)) corners.push_back(base + vertex);
                                data.AddFace(corners);
                            }
                        }
                        objects.Positions = data.Positions;
                        const auto previous = s.R.get<MeshHandle>(objects.Mesh).StoreId;
                        const auto mesh = CreateMesh(s.R, {.Data = std::move(data)});
                        s.R.replace<MeshHandle>(objects.Mesh, mesh.StoreId);
                        s.R.ctx().get<MeshStore>().Release(previous);
                    }
                    auto vertices = s.R.ctx().get<MeshStore>().GetMutableVertices(s.R.get<MeshHandle>(objects.Mesh).StoreId);
                    for (size_t i = 0; i < vertices.size(); ++i) vertices[i].Position = objects.Positions[i] * vec3{1, 0.6f + 0.05f * edit, 1} + vec3{0.1f, 0, 0};
                    s.R.emplace_or_replace<MeshPositionsChanged>(objects.Mesh);
                    s.R.emplace_or_replace<MeshGeometryDirty>(objects.Mesh);
                    for (auto entity : objects.Bodies) {
                        s.R.patch<Transform>(entity, [](auto &t) { t.S = vec3{0}; });
                        s.R.patch<Transform>(entity, [edit](auto &t) { t.S = {1, 1, 1 + 0.02f * edit}; });
                        s.R.patch<ColliderShape>(entity, [edit](auto &c) { c.LocalOffset.y = 0.01f * edit; });
                    }
                };
                const auto created = changed.BodyCreations, bodies = physics::BodyCount(changed.R);
                apply(changed, a);
                apply(fresh, b);
                expect(changed.BodyCreations == created);
                changed.Frame = fresh.Frame = 0;
                const auto step = changed.ContactStep();
                changed.Sync();
                expect(changed.BodyCreations == created + (grow ? bodies : 0)) << triangles << edit;
                expect(changed.ContactStep() == step + (grow ? 2 : 1)) << triangles << edit;
                changed.Step(12);
                physics::Clear(fresh.R);
                fresh.Sync();
                fresh.Step(12);
                for (size_t i = 0; i <= a.Bodies.size(); ++i) {
                    const auto &actual = changed.R.get<WorldTransform>(i == a.Bodies.size() ? a.Sensor : a.Bodies[i]);
                    const auto &expected = fresh.R.get<WorldTransform>(i == b.Bodies.size() ? b.Sensor : b.Bodies[i]);
                    expect(numeric::Length(actual.P - expected.P) < 1e-5f) << triangles << edit << i;
                    expect(std::abs(numeric::Dot(actual.R, expected.R)) > 0.99999f) << triangles << edit << i;
                }
                expect(changed.Contacts().size() == fresh.Contacts().size()) << triangles << edit;
                for (size_t i = 0; i < std::min(changed.Contacts().size(), fresh.Contacts().size()); ++i)
                    expect(Near(changed.Contacts()[i].NormalForce, fresh.Contacts()[i].NormalForce, 0.001f)) << triangles << edit;
            }
        }
    };

    "joint edits reuse slots and rebuild only when capacity is exceeded"_test = [] {
        Scene s;
        const auto body = s.AddBody({0, 2, 0}, Box(vec3{1}), PhysicsMotion{});
        const auto anchor = s.AddBody({0, 2, 0}, {}, {});
        const auto definition = s.R.create();
        PhysicsJointDef fixed;
        fixed.Limits.push_back({.LinearAxes = {0, 1, 2}, .Min = 0, .Max = 0});
        s.R.emplace<PhysicsJointDef>(definition, fixed);
        s.Sync();
        const auto created = s.BodyCreations;
        const auto add = [&] {
            const auto node = s.AddBody({0, 2, 0}, {}, {});
            s.Parent(node, body);
            s.R.emplace<PhysicsJoint>(node, PhysicsJoint{anchor, definition});
            return node;
        };
        std::vector<entt::entity> nodes;
        for (int i = 0; i < 8; ++i) nodes.push_back(add());
        s.Sync();
        expect(s.BodyCreations == created);
        expect(s.R.view<const PhysicsConstraintHandle>().size() == 8_u);
        std::vector<uint32_t> released;
        for (int i = 0; i < 4; ++i) {
            released.push_back(s.R.get<PhysicsConstraintHandle>(nodes[i]).ConstraintIndex);
            s.R.remove<PhysicsJoint>(nodes[i]);
            nodes[i] = add();
        }
        s.Sync();
        expect(s.BodyCreations == created);
        for (int i = 0; i < 4; ++i) expect(std::ranges::contains(released, s.R.get<PhysicsConstraintHandle>(nodes[i]).ConstraintIndex));
        nodes.push_back(add());
        s.Sync();
        expect(s.BodyCreations == created + 1);
        expect(s.R.view<const PhysicsConstraintHandle>().size() == 9_u);
        s.Step(12);
        expect(Near(s.R.get<WorldTransform>(body).P.y, 2.f, 1e-5f));
        s.R.remove<PhysicsJointDef>(definition);
        s.Sync();
        expect(s.R.view<const PhysicsConstraintHandle>().empty());
        expect(s.R.get<WorldTransform>(body).P.y < 1.9f);
        s.R.emplace<PhysicsJointDef>(definition, fixed);
        for (auto node : nodes) s.R.patch<PhysicsJoint>(node, [&](auto &j) { j.JointDefEntity = definition; });
        s.Sync();
        expect(s.BodyCreations == created + 1);
        expect(s.R.view<const PhysicsConstraintHandle>().size() == 9_u);
        expect(Near(s.R.get<WorldTransform>(body).P.y, 2.f, 1e-5f));
    };

    "timeline extension and rewind preserve bodies"_test = [] {
        Scene s;
        const auto body = s.AddBody({}, {}, PhysicsMotion{.IsKinematic = true}, {1, 0, 0});
        s.Sync();
        s.Step(12);
        const auto created = s.BodyCreations;
        const auto contact_step = s.ContactStep();
        physics::AdvancePlayback(s.R, s.Viewport, 12, 12, 0, RangeEnd + 120, Fps, false);
        expect(s.BodyCreations == created);
        expect(physics::BakedThrough(s.R) == std::optional{12u});
        expect(s.ContactStep() == contact_step);
        physics::AdvancePlayback(s.R, s.Viewport, 12, 0, 0, RangeEnd + 120, Fps, true);
        expect(s.BodyCreations == created);
        expect(s.R.get<WorldTransform>(body).P == vec3{});
        physics::AdvancePlayback(s.R, s.Viewport, 0, 12, 0, RangeEnd + 120, Fps, false);
        expect(Near(s.R.get<WorldTransform>(body).P.x, 12.f / Fps, 1e-6f));
    };

    return RunSuites();
}
