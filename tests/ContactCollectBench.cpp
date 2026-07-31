// Times the contact reporting the audio system needs, against the physics step it rides on.
// Two ways of obtaining a contact's impulses, compared on the same scene.

#include "Jolt/Jolt.h"

#include "Jolt/Core/Factory.h"
#include "Jolt/Core/JobSystemThreadPool.h"
#include "Jolt/Core/TempAllocator.h"
#include "Jolt/Physics/Body/BodyCreationSettings.h"
#include "Jolt/Physics/Collision/EstimateCollisionResponse.h"
#include "Jolt/Physics/Collision/Shape/BoxShape.h"
#include "Jolt/Physics/PhysicsSystem.h"
#include "Jolt/RegisterTypes.h"

#include "JoltTestLayers.h"

#include <entt/entity/registry.hpp>

#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace JPH;

namespace {
// What the drain consumes, matching the fields the collectors append per sub-shape entry.
struct RawSustained {
    uint32 Body1Index, Body2Index;
    uint64 SubShapeKey;
    Vec3 Point, Normal, Slip, Local1, Local2;
    float NormalImpulse, Restitution;
};

// Tags a body whose contacts are reported in detail, standing in for the real component of the same name.
struct ReportContacts {};

// How a callback decides whether a pair is worth reporting: an array indexed by body, or the body's entity.
enum class Gate { Array,
                  Entity };

// Common to both collectors: everything except how the impulses and contact point are obtained.
class CollectorBase : public ContactListener {
public:
    const PhysicsSystem *System{nullptr}; // Set once the scene's bodies are in.
    Gate ReportGate{Gate::Array};
    std::vector<uint8_t> Reporting; // Gate::Array
    const entt::registry *Registry{nullptr}; // Gate::Entity
    std::unordered_map<uint32, entt::entity> EntityByBodyIndex; // Gate::Entity
    std::mutex Mutex;
    std::vector<RawSustained> Raw;
    size_t Entries{0};

    // A pair is worth solving for detail when at least one of its bodies reports contacts.
    bool PairReports(const Body &b1, const Body &b2) const { return Reports(b1) || Reports(b2); }

    void Reset() {
        Raw.clear();
        Entries = 0;
    }

private:
    bool Reports(const Body &b) const {
        const auto index = b.GetID().GetIndex();
        if (ReportGate == Gate::Array) return index < Reporting.size() && Reporting[index] != 0;
        const auto it = EntityByBodyIndex.find(index);
        return it != EntityByBodyIndex.end() && Registry->all_of<ReportContacts>(it->second);
    }

protected:
    void Append(const Body &b1, const Body &b2, const ContactManifold &manifold, const ContactSettings &s, RVec3Arg world_point, Vec3Arg local1, Vec3Arg local2, float normal_impulse) {
        const Vec3 relative_velocity = b1.GetPointVelocity(world_point) - b2.GetPointVelocity(world_point);
        const Vec3 normal = manifold.mWorldSpaceNormal;
        const Vec3 slip = relative_velocity - normal * relative_velocity.Dot(normal);
        const uint64 key = (uint64(manifold.mSubShapeID1.GetValue()) << 32) | manifold.mSubShapeID2.GetValue();
        const std::scoped_lock lock{Mutex};
        Raw.emplace_back(b1.GetID().GetIndex(), b2.GetID().GetIndex(), key, Vec3{world_point}, normal, slip, local1, local2, normal_impulse, s.mCombinedRestitution);
        ++Entries;
    }
};

// Predict the impulses with a two-body solve, then weigh the manifold's points by them.
class EstimateCollector final : public CollectorBase {
public:
    void OnContactPersisted(const Body &b1, const Body &b2, const ContactManifold &manifold, ContactSettings &s) override {
        if (!PairReports(b1, b2) || manifold.mRelativeContactPointsOn1.empty()) return;
        CollisionEstimationResult est;
        EstimateCollisionResponse(b1, b2, manifold, est, s.mCombinedFriction, s.mCombinedRestitution, 1.0f, 4);
        float normal_impulse = 0;
        for (const float impulse : est.mContactImpulse) normal_impulse += impulse;
        if (normal_impulse <= 0) return;
        Vec3 point = Vec3::sZero();
        for (uint i = 0; i < manifold.mRelativeContactPointsOn1.size(); ++i) point += manifold.mRelativeContactPointsOn1[i] * est.mContactImpulse[i];
        const RVec3 world_point = manifold.mBaseOffset + point / normal_impulse;
        Append(b1, b2, manifold, s, world_point, b1.GetInverseCenterOfMassTransform() * world_point, b2.GetInverseCenterOfMassTransform() * world_point, normal_impulse);
    }
};

// Read back what the solver applied, which already carries the per-point positions.
class AppliedCollector final : public CollectorBase {
public:
    void OnContactPersisted(const Body &b1, const Body &b2, const ContactManifold &manifold, ContactSettings &s) override {
        if (!PairReports(b1, b2) || manifold.mRelativeContactPointsOn1.empty()) return;
        ContactConstraintManager::AppliedContactImpulses applied;
        if (!System->GetAppliedContactImpulses(SubShapeIDPair{b1.GetID(), manifold.mSubShapeID1, b2.GetID(), manifold.mSubShapeID2}, applied)) return;
        float normal_impulse = 0;
        Vec3 local1 = Vec3::sZero(), local2 = Vec3::sZero();
        for (const auto &p : applied.mPoints) {
            normal_impulse += p.mNormalImpulse;
            local1 += p.mPosition1 * p.mNormalImpulse;
            local2 += p.mPosition2 * p.mNormalImpulse;
        }
        if (normal_impulse <= 0) return;
        local1 /= normal_impulse;
        local2 /= normal_impulse;
        Append(b1, b2, manifold, s, b1.GetCenterOfMassTransform() * local1, local1, local2, normal_impulse);
    }
};

// A pile of boxes resting on a floor, reporting contacts every step.
// A null collector leaves the step uninstrumented, the baseline the two collectors are measured against.
double Bench(uint32 rows, CollectorBase *collector, uint32 frames) {
    TempAllocatorImpl temp_allocator{32 * 1024 * 1024};
    JobSystemThreadPool job_system{cMaxPhysicsJobs, cMaxPhysicsBarriers, int(std::max(1u, std::thread::hardware_concurrency() - 1))};
    BPLayerInterface bp_layer_interface;
    ObjectVsBPFilter object_vs_bp_filter;
    ObjectPairFilter object_pair_filter;

    PhysicsSystem system;
    system.Init(4096, 0, 8192, 8192, bp_layer_interface, object_vs_bp_filter, object_pair_filter);
    entt::registry registry;
    if (collector != nullptr) {
        collector->System = &system;
        collector->Registry = &registry;
        system.SetContactListener(collector);
    }
    // Every body in the scene reports, so both gates do the same work and reach the same result.
    const auto mark_reporting = [&](BodyID id) {
        if (collector == nullptr) return;
        const auto index = id.GetIndex();
        if (index >= collector->Reporting.size()) collector->Reporting.resize(index + 1, 0);
        collector->Reporting[index] = 1;
        const auto e = registry.create();
        registry.emplace<ReportContacts>(e);
        collector->EntityByBodyIndex.emplace(index, e);
    };

    BodyInterface &bi = system.GetBodyInterface();
    auto *const floor_shape = new BoxShape(Vec3(100, 1, 100));
    mark_reporting(bi.CreateAndAddBody(BodyCreationSettings(floor_shape, RVec3(0, -1, 0), Quat::sIdentity(), EMotionType::Static, Layers::NonMoving), EActivation::DontActivate));

    // A grid of boxes just touching, so each rests on the floor and against its neighbours.
    auto *const box_shape = new BoxShape(Vec3(0.5f, 0.5f, 0.5f));
    for (uint32 x = 0; x < rows; ++x) {
        for (uint32 z = 0; z < rows; ++z) {
            BodyCreationSettings settings{box_shape, RVec3(Real(x) * 1.01, 0.5, Real(z) * 1.01), Quat::sIdentity(), EMotionType::Dynamic, Layers::Moving};
            settings.mAllowSleeping = false; // Contacts are only reported between active bodies.
            mark_reporting(bi.CreateAndAddBody(settings, EActivation::Activate));
        }
    }
    system.OptimizeBroadPhase();

    const auto frame = [&] {
        if (collector != nullptr) collector->Reset();
        system.Update(1.0f / 60.0f, 10, &temp_allocator, &job_system);
    };
    for (uint32 i = 0; i < 60; ++i) frame(); // let the pile settle and the solver warm start

    const auto start = std::chrono::steady_clock::now();
    for (uint32 i = 0; i < frames; ++i) frame();
    const auto elapsed = std::chrono::duration<double, std::micro>{std::chrono::steady_clock::now() - start}.count();
    return elapsed / double(frames);
}
} // namespace

int main() {
    RegisterDefaultAllocator();
    Factory::sInstance = new Factory();
    RegisterTypes();

    constexpr uint32 Frames{300};
    std::printf("%-8s %9s %12s %12s %12s %12s\n", "pile", "contacts", "no collect", "estimate", "applied", "entity gate");
    for (const uint32 rows : {4u, 8u, 16u}) {
        EstimateCollector estimate;
        AppliedCollector applied, entity_gated;
        entity_gated.ReportGate = Gate::Entity;
        const double none = Bench(rows, nullptr, Frames);
        const double est = Bench(rows, &estimate, Frames);
        const double app = Bench(rows, &applied, Frames);
        const double gated = Bench(rows, &entity_gated, Frames);
        std::printf("%-8u %9zu %9.2fus %9.2fus %9.2fus %9.2fus\n", rows * rows, estimate.Entries, none, est, app, gated);
    }
    return 0;
}
