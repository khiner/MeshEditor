// All Jolt includes are isolated to this file.

#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/ScaledShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/Shape/TaperedCapsuleShape.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

JPH_SUPPRESS_WARNINGS

#include "PhysicsWorld.h"
#include "Transform.h"

#include <entt/entity/registry.hpp>

#include <thread>

using namespace JPH;
using namespace JPH::literals;

// --- Coordinate conversion helpers ---

static inline Vec3 ToJolt(vec3 v) { return {v.x, v.y, v.z}; }
static inline Quat ToJoltQuat(quat q) { return {q.x, q.y, q.z, q.w}; }
static inline vec3 FromJoltVec3(Vec3 v) { return {v.GetX(), v.GetY(), v.GetZ()}; }
static inline vec3 FromJoltRVec3(RVec3 v) { return {float(v.GetX()), float(v.GetY()), float(v.GetZ())}; }
static inline quat FromJoltQuat(Quat q) { return {q.GetW(), q.GetX(), q.GetY(), q.GetZ()}; }

// --- Layer configuration ---

namespace Layers {
static constexpr ObjectLayer NonMoving = 0;
static constexpr ObjectLayer Moving = 1;
static constexpr ObjectLayer NumLayers = 2;
} // namespace Layers

namespace BPLayers {
static constexpr BroadPhaseLayer NonMoving(0);
static constexpr BroadPhaseLayer Moving(1);
static constexpr uint NumLayers = 2;
} // namespace BPLayers

class BPLayerInterface final : public BroadPhaseLayerInterface {
public:
    BPLayerInterface() {
        mMapping[Layers::NonMoving] = BPLayers::NonMoving;
        mMapping[Layers::Moving] = BPLayers::Moving;
    }
    uint GetNumBroadPhaseLayers() const override { return BPLayers::NumLayers; }
    BroadPhaseLayer GetBroadPhaseLayer(ObjectLayer inLayer) const override { return mMapping[inLayer]; }
    const char *GetBroadPhaseLayerName(BroadPhaseLayer inLayer) const override {
        switch (static_cast<BroadPhaseLayer::Type>(inLayer)) {
            case static_cast<BroadPhaseLayer::Type>(0): return "NON_MOVING";
            case static_cast<BroadPhaseLayer::Type>(1): return "MOVING";
            default: return "INVALID";
        }
    }

private:
    BroadPhaseLayer mMapping[Layers::NumLayers];
};

class ObjectLayerPairFilterImpl : public ObjectLayerPairFilter {
public:
    bool ShouldCollide(ObjectLayer inLayer1, ObjectLayer inLayer2) const override {
        if (inLayer1 == Layers::NonMoving) return inLayer2 == Layers::Moving;
        return true; // Moving collides with everything
    }
};

class ObjectVsBPLayerFilterImpl : public ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(ObjectLayer inLayer1, BroadPhaseLayer inLayer2) const override {
        if (inLayer1 == Layers::NonMoving) return inLayer2 == BPLayers::Moving;
        return true;
    }
};

// --- Snapshot data ---

struct BodySnapshot {
    entt::entity Entity;
    vec3 Position;
    quat Rotation;
    vec3 Scale;
    vec3 LinearVelocity;
    vec3 AngularVelocity;
};

// --- Impl ---

struct PhysicsWorld::Impl {
    TempAllocatorImpl TempAllocator{10 * 1024 * 1024};
    JobSystemThreadPool JobSystem{cMaxPhysicsJobs, cMaxPhysicsBarriers, int(std::max(1u, std::thread::hardware_concurrency() - 1))};
    BPLayerInterface BPLayerIface;
    ObjectLayerPairFilterImpl ObjectPairFilter;
    ObjectVsBPLayerFilterImpl ObjectVsBPFilter;
    PhysicsSystem System;

    std::vector<BodySnapshot> Snapshots;

    Impl() {
        System.Init(
            65536, // max bodies
            0, // num body mutexes (0 = default)
            65536, // max body pairs
            65536, // max contact constraints
            BPLayerIface,
            ObjectVsBPFilter,
            ObjectPairFilter
        );
    }
};

// --- Shape creation ---

static Ref<Shape> CreateJoltShape(const PhysicsShape &shape) {
    switch (shape.Type) {
        case PhysicsShapeType::Box: {
            // KHR spec uses full size, Jolt uses half-extents
            return new BoxShape(ToJolt(shape.Size * 0.5f));
        }
        case PhysicsShapeType::Sphere: {
            return new SphereShape(shape.Radius);
        }
        case PhysicsShapeType::Capsule: {
            if (std::abs(shape.RadiusTop - shape.RadiusBottom) < 1e-6f) {
                return new CapsuleShape(shape.Height * 0.5f, shape.RadiusBottom);
            }
            auto result = TaperedCapsuleShapeSettings(shape.Height * 0.5f, shape.RadiusTop, shape.RadiusBottom).Create();
            return result.IsValid() ? result.Get() : Ref<Shape>(new CapsuleShape(shape.Height * 0.5f, shape.RadiusBottom));
        }
        case PhysicsShapeType::Cylinder: {
            return new CylinderShape(shape.Height * 0.5f, std::max(shape.RadiusTop, shape.RadiusBottom));
        }
        case PhysicsShapeType::ConvexHull:
        case PhysicsShapeType::TriangleMesh: {
            // Mesh-based shapes require vertex data — handled separately.
            // For now return a unit box as placeholder.
            return new BoxShape(Vec3(0.5f, 0.5f, 0.5f));
        }
    }
    return new BoxShape(Vec3(0.5f, 0.5f, 0.5f));
}

// --- Jolt one-time init/shutdown ---

static struct JoltInit {
    JoltInit() {
        RegisterDefaultAllocator();
        Factory::sInstance = new Factory();
        RegisterTypes();
    }
    ~JoltInit() {
        UnregisterTypes();
        delete Factory::sInstance;
        Factory::sInstance = nullptr;
    }
} sJoltInit;

// --- PhysicsWorld ---

PhysicsWorld::PhysicsWorld() : P(std::make_unique<Impl>()) {}
PhysicsWorld::~PhysicsWorld() = default;

bool PhysicsWorld::HasBodies() const { return P->System.GetNumBodies() > 0; }
uint32_t PhysicsWorld::BodyCount() const { return P->System.GetNumBodies(); }

void PhysicsWorld::Rebuild(entt::registry &r) {
    // Remove all existing bodies
    auto &bi = P->System.GetBodyInterface();
    for (auto [entity, handle] : r.view<PhysicsBodyHandle>().each()) {
        bi.RemoveBody(BodyID(handle.BodyId));
        bi.DestroyBody(BodyID(handle.BodyId));
    }
    r.clear<PhysicsBodyHandle>();

    // Batch-add all bodies with physics components
    std::vector<BodyCreationSettings> settings_list;
    std::vector<entt::entity> entities;

    for (auto [entity, collider] : r.view<const PhysicsCollider>().each()) {
        auto shape = CreateJoltShape(collider.Shape);
        if (!shape) continue;

        const auto *transform = r.try_get<const Transform>(entity);
        RVec3 pos = transform ? RVec3(transform->P.x, transform->P.y, transform->P.z) : RVec3::sZero();
        Quat rot = transform ? ToJoltQuat(transform->R) : Quat::sIdentity();

        // Wrap in ScaledShape if non-unit scale
        if (transform && (transform->S.x != 1.0f || transform->S.y != 1.0f || transform->S.z != 1.0f)) {
            shape = new ScaledShape(shape, ToJolt(transform->S));
        }

        const auto *motion = r.try_get<const PhysicsMotion>(entity);
        EMotionType motion_type = EMotionType::Static;
        ObjectLayer layer = Layers::NonMoving;
        if (motion) {
            motion_type = motion->IsKinematic ? EMotionType::Kinematic : EMotionType::Dynamic;
            layer = Layers::Moving;
        }

        BodyCreationSettings bcs(shape, pos, rot, motion_type, layer);

        if (motion) {
            if (motion->Mass.has_value()) {
                if (motion_type == EMotionType::Dynamic) {
                    bcs.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
                    bcs.mMassPropertiesOverride.mMass = *motion->Mass;
                }
            }
            bcs.mLinearVelocity = ToJolt(motion->LinearVelocity);
            bcs.mAngularVelocity = ToJolt(motion->AngularVelocity);
            bcs.mGravityFactor = motion->GravityFactor;
        }

        // Physics material
        if (collider.PhysicsMaterialIndex.has_value() && *collider.PhysicsMaterialIndex < Materials.size()) {
            const auto &mat = Materials[*collider.PhysicsMaterialIndex];
            bcs.mFriction = mat.DynamicFriction;
            bcs.mRestitution = mat.Restitution;
        }

        settings_list.push_back(std::move(bcs));
        entities.push_back(entity);
    }

    if (settings_list.empty()) return;

    // Batch create and add
    for (size_t i = 0; i < settings_list.size(); ++i) {
        Body *body = bi.CreateBody(settings_list[i]);
        if (!body) continue;
        bi.AddBody(body->GetID(), settings_list[i].mMotionType == EMotionType::Static ? EActivation::DontActivate : EActivation::Activate);
        r.emplace_or_replace<PhysicsBodyHandle>(entities[i], PhysicsBodyHandle{body->GetID().GetIndexAndSequenceNumber()});
    }

    P->System.OptimizeBroadPhase();
}

void PhysicsWorld::AddBody(entt::registry &r, entt::entity entity) {
    const auto *collider = r.try_get<const PhysicsCollider>(entity);
    if (!collider) return;

    auto shape = CreateJoltShape(collider->Shape);
    if (!shape) return;

    const auto *transform = r.try_get<const Transform>(entity);
    RVec3 pos = transform ? RVec3(transform->P.x, transform->P.y, transform->P.z) : RVec3::sZero();
    Quat rot = transform ? ToJoltQuat(transform->R) : Quat::sIdentity();

    if (transform && (transform->S.x != 1.0f || transform->S.y != 1.0f || transform->S.z != 1.0f)) {
        shape = new ScaledShape(shape, ToJolt(transform->S));
    }

    const auto *motion = r.try_get<const PhysicsMotion>(entity);
    EMotionType motion_type = EMotionType::Static;
    ObjectLayer layer = Layers::NonMoving;
    if (motion) {
        motion_type = motion->IsKinematic ? EMotionType::Kinematic : EMotionType::Dynamic;
        layer = Layers::Moving;
    }

    BodyCreationSettings bcs(shape, pos, rot, motion_type, layer);
    if (motion) {
        if (motion->Mass.has_value()) {
            if (motion_type == EMotionType::Dynamic) {
                bcs.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
                bcs.mMassPropertiesOverride.mMass = *motion->Mass;
            }
        }
        bcs.mLinearVelocity = ToJolt(motion->LinearVelocity);
        bcs.mAngularVelocity = ToJolt(motion->AngularVelocity);
        bcs.mGravityFactor = motion->GravityFactor;
    }

    if (collider->PhysicsMaterialIndex.has_value() && *collider->PhysicsMaterialIndex < Materials.size()) {
        const auto &mat = Materials[*collider->PhysicsMaterialIndex];
        bcs.mFriction = mat.DynamicFriction;
        bcs.mRestitution = mat.Restitution;
    }

    auto &bi = P->System.GetBodyInterface();
    Body *body = bi.CreateBody(bcs);
    if (!body) return;
    bi.AddBody(body->GetID(), motion_type == EMotionType::Static ? EActivation::DontActivate : EActivation::Activate);
    r.emplace_or_replace<PhysicsBodyHandle>(entity, PhysicsBodyHandle{body->GetID().GetIndexAndSequenceNumber()});
}

void PhysicsWorld::RemoveBody(entt::registry &r, entt::entity entity) {
    const auto *handle = r.try_get<const PhysicsBodyHandle>(entity);
    if (!handle || handle->BodyId == UINT32_MAX) return;

    auto &bi = P->System.GetBodyInterface();
    BodyID id(handle->BodyId);
    bi.RemoveBody(id);
    bi.DestroyBody(id);
    r.remove<PhysicsBodyHandle>(entity);
}

void PhysicsWorld::Step(entt::registry &r, float dt) {
    P->System.SetGravity(ToJolt(Gravity));
    P->System.Update(dt, SubSteps, &P->TempAllocator, &P->JobSystem);

    // Sync Jolt → ECS for dynamic bodies
    auto &bi = P->System.GetBodyInterface();
    for (auto [entity, motion, handle] : r.view<PhysicsMotion, PhysicsBodyHandle>().each()) {
        if (motion.IsKinematic) continue;
        BodyID id(handle.BodyId);
        if (!bi.IsActive(id)) continue;

        r.patch<Transform>(entity, [&](Transform &t) {
            t.P = FromJoltRVec3(bi.GetPosition(id));
            t.R = FromJoltQuat(bi.GetRotation(id));
        });
        motion.LinearVelocity = FromJoltVec3(bi.GetLinearVelocity(id));
        motion.AngularVelocity = FromJoltVec3(bi.GetAngularVelocity(id));
    }
}

void PhysicsWorld::SaveSnapshot(entt::registry &r) {
    P->Snapshots.clear();
    auto &bi = P->System.GetBodyInterface();
    for (auto [entity, motion, handle] : r.view<PhysicsMotion, PhysicsBodyHandle>().each()) {
        BodyID id(handle.BodyId);
        const auto *t = r.try_get<const Transform>(entity);
        P->Snapshots.push_back({
            entity,
            t ? t->P : vec3{0},
            t ? t->R : quat{1, 0, 0, 0},
            t ? t->S : vec3{1},
            FromJoltVec3(bi.GetLinearVelocity(id)),
            FromJoltVec3(bi.GetAngularVelocity(id)),
        });
    }
}

void PhysicsWorld::RestoreSnapshot(entt::registry &r) {
    auto &bi = P->System.GetBodyInterface();
    for (const auto &snap : P->Snapshots) {
        if (!r.valid(snap.Entity)) continue;
        auto *handle = r.try_get<PhysicsBodyHandle>(snap.Entity);
        if (!handle || handle->BodyId == UINT32_MAX) continue;

        BodyID id(handle->BodyId);
        bi.SetPositionAndRotation(id, RVec3(snap.Position.x, snap.Position.y, snap.Position.z), ToJoltQuat(snap.Rotation), EActivation::DontActivate);
        bi.SetLinearVelocity(id, ToJolt(snap.LinearVelocity));
        bi.SetAngularVelocity(id, ToJolt(snap.AngularVelocity));

        r.patch<Transform>(snap.Entity, [&](Transform &t) {
            t.P = snap.Position;
            t.R = snap.Rotation;
            t.S = snap.Scale;
        });
    }
}
