// All Jolt includes are isolated to this file.

#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyLock.h>
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
#include "SceneTree.h"
#include "mesh/Mesh.h"

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

    std::vector<Ref<Constraint>> Constraints;
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

static Ref<Shape> CreateJoltShape(const PhysicsShape &shape, const Mesh *mesh) {
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
        case PhysicsShapeType::ConvexHull: {
            if (!mesh || mesh->VertexCount() == 0) break;
            auto verts = mesh->GetVerticesSpan();
            // Jolt Vec3 is a 16-byte SIMD type — must convert from interleaved Vertex positions
            Array<Vec3> points;
            points.reserve(int(verts.size()));
            for (const auto &v : verts) points.push_back(Vec3(v.Position.x, v.Position.y, v.Position.z));
            ConvexHullShapeSettings settings(points.data(), int(points.size()));
            auto result = settings.Create();
            if (result.IsValid()) return result.Get();
            break;
        }
        case PhysicsShapeType::TriangleMesh: {
            if (!mesh || mesh->FaceCount() == 0) break;
            auto verts = mesh->GetVerticesSpan();
            VertexList vertices;
            vertices.reserve(verts.size());
            for (const auto &v : verts) vertices.push_back(Float3(v.Position.x, v.Position.y, v.Position.z));
            auto indices = mesh->CreateTriangleIndices();
            IndexedTriangleList triangles;
            triangles.reserve(indices.size() / 3);
            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                triangles.push_back(IndexedTriangle(indices[i], indices[i + 1], indices[i + 2]));
            }
            MeshShapeSettings settings(std::move(vertices), std::move(triangles));
            auto result = settings.Create();
            if (result.IsValid()) return result.Get();
            break;
        }
    }
    return new BoxShape(Vec3(0.5f, 0.5f, 0.5f));
}

// Find the nearest ancestor (or self) that has a PhysicsBodyHandle.
static entt::entity FindBodyAncestor(const entt::registry &r, entt::entity e) {
    for (; e != entt::null; e = GetParentEntity(r, e)) {
        if (r.all_of<PhysicsBodyHandle>(e)) return e;
    }
    return entt::null;
}

// Configure a SixDOFConstraintSettings from a KHR joint definition.
static void ConfigureJointSettings(SixDOFConstraintSettings &settings, const PhysicsJointDef &def) {
    // Default: all axes fixed (locked to the attachment frame)
    for (int a = 0; a < SixDOFConstraintSettings::EAxis::Num; ++a)
        settings.MakeFixedAxis(static_cast<SixDOFConstraintSettings::EAxis>(a));

    // Apply limits — each limit entry may cover multiple axes
    for (const auto &limit : def.Limits) {
        auto configure_axis = [&](SixDOFConstraintSettings::EAxis axis) {
            if (!limit.Min && !limit.Max) {
                settings.MakeFreeAxis(axis);
            } else {
                float lo = limit.Min.value_or(-FLT_MAX);
                float hi = limit.Max.value_or(FLT_MAX);
                settings.SetLimitedAxis(axis, lo, hi);
            }
            // Soft spring limits (translation axes only in Jolt)
            if (limit.Stiffness && axis < SixDOFConstraintSettings::EAxis::NumTranslation) {
                settings.mLimitsSpringSettings[axis] = SpringSettings(ESpringMode::StiffnessAndDamping, *limit.Stiffness, limit.Damping);
            }
        };
        for (uint8_t a : limit.LinearAxes) {
            if (a < 3) configure_axis(static_cast<SixDOFConstraintSettings::EAxis>(a));
        }
        for (uint8_t a : limit.AngularAxes) {
            if (a < 3) configure_axis(static_cast<SixDOFConstraintSettings::EAxis>(a + 3));
        }
    }

    // Apply drives as motor settings
    for (const auto &drive : def.Drives) {
        int axis_index = (drive.Type == PhysicsDriveType::Linear ? 0 : 3) + drive.Axis;
        auto axis = static_cast<SixDOFConstraintSettings::EAxis>(axis_index);
        auto &motor = settings.mMotorSettings[axis_index];
        if (drive.Stiffness > 0.0f || drive.Damping > 0.0f) {
            motor.mSpringSettings = SpringSettings(ESpringMode::StiffnessAndDamping, drive.Stiffness, drive.Damping);
        }
        if (drive.Type == PhysicsDriveType::Linear) {
            motor.SetForceLimit(drive.MaxForce);
        } else {
            motor.SetTorqueLimit(drive.MaxForce);
        }
        // Ensure the axis isn't fixed so the motor can act
        if (settings.IsFixedAxis(axis)) settings.MakeFreeAxis(axis);
    }
}

// After constraint creation, set motor states and targets from drives.
static void ApplyDriveTargets(SixDOFConstraint &constraint, const PhysicsJointDef &def) {
    Vec3 target_pos = Vec3::sZero(), target_vel = Vec3::sZero();
    Vec3 target_ang_vel = Vec3::sZero();
    Quat target_orient = Quat::sIdentity();
    bool has_orient_target = false;

    for (const auto &drive : def.Drives) {
        int axis_index = (drive.Type == PhysicsDriveType::Linear ? 0 : 3) + drive.Axis;
        auto axis = static_cast<SixDOFConstraintSettings::EAxis>(axis_index);

        bool has_position = drive.Stiffness > 0.0f || drive.PositionTarget != 0.0f;
        bool has_velocity = drive.VelocityTarget != 0.0f;

        if (has_position) {
            constraint.SetMotorState(axis, EMotorState::Position);
            if (drive.Type == PhysicsDriveType::Linear) {
                target_pos.SetComponent(drive.Axis, drive.PositionTarget);
            } else {
                // Compose per-axis angular position targets into a quaternion
                static const Vec3 axes[3] = {Vec3::sAxisX(), Vec3::sAxisY(), Vec3::sAxisZ()};
                target_orient = target_orient * Quat::sRotation(axes[drive.Axis], drive.PositionTarget);
                has_orient_target = true;
            }
        } else if (has_velocity) {
            constraint.SetMotorState(axis, EMotorState::Velocity);
            if (drive.Type == PhysicsDriveType::Linear)
                target_vel.SetComponent(drive.Axis, drive.VelocityTarget);
            else
                target_ang_vel.SetComponent(drive.Axis, drive.VelocityTarget);
        }
    }
    constraint.SetTargetPositionCS(target_pos);
    constraint.SetTargetVelocityCS(target_vel);
    constraint.SetTargetAngularVelocityCS(target_ang_vel);
    if (has_orient_target) constraint.SetTargetOrientationCS(target_orient);
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
    // Remove all existing constraints
    for (auto &c : P->Constraints) P->System.RemoveConstraint(c);
    P->Constraints.clear();

    // Remove all existing bodies
    auto &bi = P->System.GetBodyInterface();
    for (auto [entity, handle] : r.view<PhysicsBodyHandle>().each()) {
        bi.RemoveBody(BodyID(handle.BodyId));
        bi.DestroyBody(BodyID(handle.BodyId));
    }
    r.clear<PhysicsBodyHandle>();

    // Create all bodies
    for (auto [entity, collider] : r.view<const PhysicsCollider>().each()) {
        const auto *mesh = collider.Shape.MeshEntity ? r.try_get<const Mesh>(*collider.Shape.MeshEntity) : nullptr;
        auto shape = CreateJoltShape(collider.Shape, mesh);
        if (!shape) continue;

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
            if (motion->Mass.has_value() && motion_type == EMotionType::Dynamic) {
                bcs.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
                bcs.mMassPropertiesOverride.mMass = *motion->Mass;
            }
            bcs.mLinearVelocity = ToJolt(motion->LinearVelocity);
            bcs.mAngularVelocity = ToJolt(motion->AngularVelocity);
            bcs.mGravityFactor = motion->GravityFactor;
        }

        if (collider.PhysicsMaterialIndex.has_value() && *collider.PhysicsMaterialIndex < Materials.size()) {
            const auto &mat = Materials[*collider.PhysicsMaterialIndex];
            bcs.mFriction = mat.DynamicFriction;
            bcs.mRestitution = mat.Restitution;
        }

        Body *body = bi.CreateBody(bcs);
        if (!body) continue;
        bi.AddBody(body->GetID(), motion_type == EMotionType::Static ? EActivation::DontActivate : EActivation::Activate);
        r.emplace_or_replace<PhysicsBodyHandle>(entity, PhysicsBodyHandle{body->GetID().GetIndexAndSequenceNumber()});
    }

    // Create constraints from PhysicsJoint components
    const auto &lock_iface = P->System.GetBodyLockInterfaceNoLock();
    for (auto [entity, joint] : r.view<const PhysicsJoint>().each()) {
        if (joint.ConnectedNode == entt::null) continue;
        if (joint.JointDefIndex >= JointDefs.size()) continue;

        // Body 1: nearest ancestor (or self) with a physics body
        entt::entity body1_entity = FindBodyAncestor(r, entity);
        // Body 2: the connected node (should itself be a body, or find its body ancestor)
        entt::entity body2_entity = FindBodyAncestor(r, joint.ConnectedNode);
        if (body1_entity == entt::null || body2_entity == entt::null) continue;
        if (body1_entity == body2_entity) continue;

        const auto *h1 = r.try_get<const PhysicsBodyHandle>(body1_entity);
        const auto *h2 = r.try_get<const PhysicsBodyHandle>(body2_entity);
        if (!h1 || !h2) continue;

        const auto &def = JointDefs[joint.JointDefIndex];

        SixDOFConstraintSettings settings;
        settings.mSpace = EConstraintSpace::WorldSpace;

        // Joint anchor at the joint node's world position
        const auto *jt = r.try_get<const Transform>(entity);
        if (jt) {
            RVec3 anchor(jt->P.x, jt->P.y, jt->P.z);
            settings.mPosition1 = settings.mPosition2 = anchor;
            // Derive constraint frame axes from joint node rotation
            glm::mat3 rot_mat = glm::mat3_cast(glm::normalize(jt->R));
            settings.mAxisX1 = settings.mAxisX2 = Vec3(rot_mat[0].x, rot_mat[0].y, rot_mat[0].z);
            settings.mAxisY1 = settings.mAxisY2 = Vec3(rot_mat[1].x, rot_mat[1].y, rot_mat[1].z);
        }

        ConfigureJointSettings(settings, def);

        BodyLockWrite lock1(lock_iface, BodyID(h1->BodyId));
        BodyLockWrite lock2(lock_iface, BodyID(h2->BodyId));
        if (!lock1.Succeeded() || !lock2.Succeeded()) continue;

        auto *constraint = static_cast<SixDOFConstraint *>(settings.Create(lock1.GetBody(), lock2.GetBody()));
        ApplyDriveTargets(*constraint, def);
        P->System.AddConstraint(constraint);
        P->Constraints.push_back(constraint);
    }

    P->System.OptimizeBroadPhase();
}

void PhysicsWorld::AddBody(entt::registry &r, entt::entity entity) {
    const auto *collider = r.try_get<const PhysicsCollider>(entity);
    if (!collider) return;

    const auto *mesh = collider->Shape.MeshEntity ? r.try_get<const Mesh>(*collider->Shape.MeshEntity) : nullptr;
    auto shape = CreateJoltShape(collider->Shape, mesh);
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
    // Restore ECS transforms and velocities from snapshot
    for (const auto &snap : P->Snapshots) {
        if (!r.valid(snap.Entity)) continue;
        r.patch<Transform>(snap.Entity, [&](Transform &t) {
            t.P = snap.Position;
            t.R = snap.Rotation;
            t.S = snap.Scale;
        });
        if (auto *motion = r.try_get<PhysicsMotion>(snap.Entity)) {
            motion->LinearVelocity = snap.LinearVelocity;
            motion->AngularVelocity = snap.AngularVelocity;
        }
    }
    // Rebuild all Jolt bodies and constraints from the restored ECS state.
    // This guarantees deterministic replay by eliminating stale solver warm-start cache.
    // Ideally we'd just clear the solver cache, but Jolt doesn't expose such an API.
    Rebuild(r);
}
