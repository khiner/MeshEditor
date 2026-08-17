// Pins the contact reporting Jolt delivers, which the sustained-contact audio path consumes.
// Every assertion is a property of the engine rather than of this repo, and the audio design rests on all of them:
// manifolds per body pair, points per manifold, what the per-point impulses sum to, and whether a manifold's
// sub-shape identity holds while the contact does.

#include "RunSuites.h"

#include "Jolt/Jolt.h"

#include "Jolt/Core/Factory.h"
#include "Jolt/Core/JobSystemSingleThreaded.h"
#include "Jolt/Core/TempAllocator.h"
#include "Jolt/Physics/Body/BodyCreationSettings.h"
#include "Jolt/Physics/Collision/Shape/BoxShape.h"
#include "Jolt/Physics/Collision/Shape/MeshShape.h"
#include "Jolt/Physics/Collision/Shape/StaticCompoundShape.h"
#include "Jolt/Physics/PhysicsSystem.h"
#include "Jolt/RegisterTypes.h"

#include "JoltTestLayers.h"
#include "Near.h"

#include <boost/ut.hpp>

#include <cmath>
#include <set>
#include <vector>

using namespace JPH;
using namespace boost::ut;

namespace {
// One persisting manifold as the audio path reads it: the solver's per-point impulses and where they act.
struct Manifold {
    uint64 SubShapeKey{0};
    Vec3 Normal{Vec3::sZero()};
    std::vector<Vec3> Points; // In body 1's centre of mass space, which is the frame the sweep is differenced in.
    std::vector<float> Impulses;
    float TotalImpulse{0};
};

// Records one collision step's reports, reading applied impulses as the physics system's own listener does.
class Recorder final : public ContactListener {
public:
    const PhysicsSystem *System{nullptr};
    std::vector<Manifold> Manifolds;

    void OnContactPersisted(const Body &b1, const Body &b2, const ContactManifold &manifold, ContactSettings &) override {
        ContactConstraintManager::AppliedContactImpulses applied;
        if (!System->GetAppliedContactImpulses(SubShapeIDPair{b1.GetID(), manifold.mSubShapeID1, b2.GetID(), manifold.mSubShapeID2}, applied)) return;
        Manifold m;
        m.SubShapeKey = (uint64(manifold.mSubShapeID1.GetValue()) << 32) | manifold.mSubShapeID2.GetValue();
        m.Normal = manifold.mWorldSpaceNormal;
        for (const auto &p : applied.mPoints) {
            m.Points.push_back(p.mPosition1);
            m.Impulses.push_back(p.mNormalImpulse);
            m.TotalImpulse += p.mNormalImpulse;
        }
        Manifolds.push_back(std::move(m));
    }
};

Ref<Shape> Box(Vec3 half_extent) { return Ref<Shape>{new BoxShape{half_extent}}; }

// A world with one dynamic body against one static body, stepped single threaded so the reported set is reproducible.
struct World {
    static constexpr float Dt{1.f / 60.f};
    static constexpr uint CollisionSteps{1}; // One report per manifold per frame, so a frame's set is one collision step's.

    TempAllocatorImpl TempAllocator{16 * 1024 * 1024};
    JobSystemSingleThreaded JobSystem{cMaxPhysicsJobs};
    BPLayerInterface BPLayerIface;
    ObjectVsBPFilter ObjectVsBP;
    ObjectPairFilter ObjectPair;
    PhysicsSystem System;
    Recorder Listener;

    World() {
        System.Init(64, 0, 256, 256, BPLayerIface, ObjectVsBP, ObjectPair);
        Listener.System = &System;
        System.SetContactListener(&Listener);
    }

    BodyID AddStatic(const Ref<Shape> &shape, RVec3 position) {
        return System.GetBodyInterface().CreateAndAddBody(
            BodyCreationSettings{shape, position, Quat::sIdentity(), EMotionType::Static, Layers::NonMoving}, EActivation::DontActivate
        );
    }
    BodyID AddDynamic(const Ref<Shape> &shape, RVec3 position, float gravity_factor = 1.f) {
        BodyCreationSettings settings{shape, position, Quat::sIdentity(), EMotionType::Dynamic, Layers::Moving};
        settings.mAllowSleeping = false; // Contacts are reported only between active bodies.
        settings.mGravityFactor = gravity_factor;
        return System.GetBodyInterface().CreateAndAddBody(settings, EActivation::Activate);
    }

    // A floor wide enough that nothing reaches its edges, with its top face at y = 0.
    BodyID AddFloor() { return AddStatic(Box({50, 1, 50}), RVec3(0, -1, 0)); }
    // A unit box standing on that floor.
    BodyID AddUnitBox() { return AddDynamic(Box({0.5f, 0.5f, 0.5f}), RVec3(0, 0.5f, 0)); }

    void Step() {
        Listener.Manifolds.clear();
        System.Update(Dt, CollisionSteps, &TempAllocator, &JobSystem);
    }
    // Settle the contact and warm start the solver, then return what the last frame reported.
    const std::vector<Manifold> &Settle(uint frames = 120) {
        for (uint i = 0; i < frames; ++i) Step();
        return Listener.Manifolds;
    }
};

// A compound of identical legs, each a box whose bottom face sits at the compound origin's height.
Ref<Shape> Legs(const std::vector<Vec3> &centers, Vec3 half_extent) {
    StaticCompoundShapeSettings settings;
    for (const auto &c : centers) settings.AddShape(c, Quat::sIdentity(), new BoxShape{half_extent});
    return settings.Create().Get();
}

// A flat floor tessellated finely enough that one resting box spans many triangles.
Ref<Shape> TriangleFloor(float extent, float cell) {
    TriangleList triangles;
    for (float x = -extent; x < extent - 0.5f * cell; x += cell) {
        for (float z = -extent; z < extent - 0.5f * cell; z += cell) {
            const Float3 a{x, 0, z}, b{x + cell, 0, z}, c{x + cell, 0, z + cell}, d{x, 0, z + cell};
            triangles.emplace_back(a, d, c);
            triangles.emplace_back(a, c, b);
        }
    }
    return MeshShapeSettings{triangles}.Create().Get();
}

// The load the solver carried over one collision step, which every manifold's impulses must add up to.
float SupportedImpulse(float mass) { return mass * 9.81f * World::Dt / float(World::CollisionSteps); }

// The manifolds' impulse-weighted mean normal, which is the direction a merge over a body pair excites along.
Vec3 MergedNormal(const std::vector<Manifold> &manifolds) {
    Vec3 sum = Vec3::sZero();
    float impulse = 0;
    for (const auto &m : manifolds) {
        sum += m.Normal * m.TotalImpulse;
        impulse += m.TotalImpulse;
    }
    return impulse > 0 ? sum / impulse : Vec3::sZero();
}

// The sub-shape keys a box settled on `floor` reports over 30 steps, which says whether that identity is
// stable enough to track a region across steps.
std::set<uint64> KeysWhileResting(const Ref<Shape> &floor, RVec3 floor_position) {
    World w;
    w.AddStatic(floor, floor_position);
    w.AddUnitBox();
    w.Settle();

    std::set<uint64> keys;
    for (uint i = 0; i < 30; ++i) {
        w.Step();
        for (const auto &m : w.Listener.Manifolds) keys.insert(m.SubShapeKey);
    }
    return keys;
}
} // namespace

int main() {
    RegisterDefaultAllocator();
    Factory::sInstance = new Factory();
    RegisterTypes();

    "a box resting flat reports one manifold of four spread points"_test = [] {
        World w;
        w.AddFloor();
        w.AddUnitBox();
        const auto &manifolds = w.Settle();

        expect(manifolds.size() == 1_ul);
        expect(manifolds[0].Points.size() == 4_ul);
        // The four points are the face's own corners, so the load arrives distributed, not reduced to a centre.
        for (const auto &p : manifolds[0].Points) {
            expect(Near(std::abs(p.GetX()), 0.5f, 0.05f));
            expect(Near(std::abs(p.GetZ()), 0.5f, 0.05f));
        }
    };

    "per-point impulses add up to the load the step carried"_test = [] {
        World w;
        w.AddFloor();
        w.AddUnitBox();
        const auto &manifolds = w.Settle();

        expect(manifolds.size() == 1_ul);
        // Weights summing to the resultant are what make any partition of the manifold conservative.
        expect(Near(manifolds[0].TotalImpulse, SupportedImpulse(1000.f), 0.1f));
    };

    "coplanar legs of one compound arrive as a single manifold"_test = [] {
        World w;
        w.AddFloor();
        w.AddDynamic(Legs({{-0.4f, 0.3f, -0.4f}, {0.4f, 0.3f, -0.4f}, {-0.4f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.4f}}, {0.1f, 0.3f, 0.1f}), RVec3(0, 0, 0));
        const auto &manifolds = w.Settle();

        // Manifold reduction groups by normal, so four spatially separate legs share one manifold and one sub-shape key.
        // Sub-shape identity therefore cannot tell the legs apart. The points can.
        expect(manifolds.size() == 1_ul);
        expect(manifolds[0].Points.size() == 4_ul);
        std::set<std::pair<bool, bool>> quadrants;
        for (const auto &p : manifolds[0].Points) {
            expect(std::abs(p.GetX()) > 0.3f);
            expect(std::abs(p.GetZ()) > 0.3f);
            quadrants.emplace(p.GetX() > 0, p.GetZ() > 0);
        }
        expect(quadrants.size() == 4_ul); // one point per leg
    };

    "more than four coplanar regions lose the ones that are not reported"_test = [] {
        World w;
        w.AddFloor();
        const std::vector<Vec3> six{
            {-0.4f, 0.3f, -0.4f}, {0.f, 0.3f, -0.4f}, {0.4f, 0.3f, -0.4f}, {-0.4f, 0.3f, 0.4f}, {0.f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.4f}
        };
        w.AddDynamic(Legs(six, {0.1f, 0.3f, 0.1f}), RVec3(0, 0, 0));
        const auto &manifolds = w.Settle();

        // MaxContactPoints is four, so a sixth region cannot be represented however the load is distributed.
        // The reported points still carry the whole load: the resultant stays right, the distribution does not.
        expect(manifolds.size() == 1_ul);
        expect(manifolds[0].Points.size() <= 4_ul);
        expect(manifolds[0].Points.size() < six.size());
        float total = 0;
        for (const auto &m : manifolds) total += m.TotalImpulse;
        expect(Near(total, SupportedImpulse(6.f * 0.2f * 0.6f * 0.2f * 1000.f), 0.15f));
    };

    "opposed faces of one body cancel the direction a merge would excite along"_test = [] {
        World w;
        // Two walls of one static body, their inner faces 0.9 apart, squeezing a box 1.0 wide.
        w.AddStatic(Legs({{-0.55f, 0, 0}, {0.55f, 0, 0}}, {0.1f, 1.f, 1.f}), RVec3(0, 0, 0));
        w.AddDynamic(Box({0.5f, 0.5f, 0.5f}), RVec3(0, 0, 0), 0.f);
        const auto &manifolds = w.Settle();

        // Both walls press along opposing normals, so their impulse-weighted mean has no direction left.
        // Merging a body pair into one contact would read this as silence rather than as two contacts.
        expect(manifolds.size() == 2_ul);
        expect(std::abs(manifolds[0].Normal.Dot(manifolds[1].Normal) + 1.f) < 0.01f);
        expect(MergedNormal(manifolds).Length() < 0.2f);
    };

    "a resting contact keeps one sub-shape key across steps"_test = [] {
        // A reduced manifold takes the sub-shape ids of whichever hit seeded it.
        // The box's face spans a whole box floor and many triangles of a tessellated one, and both report one key throughout.
        expect(KeysWhileResting(Box({50, 1, 50}), RVec3(0, -1, 0)).size() == 1_ul);
        expect(KeysWhileResting(TriangleFloor(2.f, 0.25f), RVec3(0, 0, 0)).size() == 1_ul);
    };

    return RunSuites();
}
