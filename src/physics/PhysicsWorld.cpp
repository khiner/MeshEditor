// All Jolt includes are isolated to this file.

#include "Jolt/Jolt.h"

#include "Jolt/Core/Factory.h"
#include "Jolt/Core/JobSystemThreadPool.h"
#include "Jolt/Core/TempAllocator.h"
#include "Jolt/Physics/Body/BodyCreationSettings.h"
#include "Jolt/Physics/Body/BodyLock.h"
#include "Jolt/Physics/Collision/CollideShape.h"
#include "Jolt/Physics/Collision/Shape/BoxShape.h"
#include "Jolt/Physics/Collision/Shape/CapsuleShape.h"
#include "Jolt/Physics/Collision/Shape/ConvexHullShape.h"
#include "Jolt/Physics/Collision/Shape/CylinderShape.h"
#include "Jolt/Physics/Collision/Shape/MeshShape.h"
#include "Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h"
#include "Jolt/Physics/Collision/Shape/ScaledShape.h"
#include "Jolt/Physics/Collision/Shape/SphereShape.h"
#include "Jolt/Physics/Collision/Shape/StaticCompoundShape.h"
#include "Jolt/Physics/Collision/Shape/TaperedCapsuleShape.h"
#include "Jolt/Physics/Constraints/SixDOFConstraint.h"
#include "Jolt/Physics/PhysicsSettings.h"
#include "Jolt/Physics/PhysicsSystem.h"
#include "Jolt/RegisterTypes.h"

JPH_SUPPRESS_WARNINGS

#include "PhysicsWorld.h"
#include "SceneTree.h"
#include "mesh/Mesh.h"

#include <entt/entity/registry.hpp>

#include <thread>
#include <unordered_map>

using namespace JPH;
using namespace JPH::literals;

namespace {
inline Vec3 ToJolt(vec3 v) { return {v.x, v.y, v.z}; }
inline Quat ToJoltQuat(quat q) { return {q.x, q.y, q.z, q.w}; }
inline vec3 FromJoltVec3(Vec3 v) { return {v.GetX(), v.GetY(), v.GetZ()}; }
inline vec3 FromJoltRVec3(RVec3 v) { return {float(v.GetX()), float(v.GetY()), float(v.GetZ())}; }
inline quat FromJoltQuat(Quat q) { return {q.GetW(), q.GetX(), q.GetY(), q.GetZ()}; }

namespace Layers {
constexpr ObjectLayer NonMoving{0}, Moving{1}, NumLayers{2};
} // namespace Layers

namespace BPLayers {
constexpr BroadPhaseLayer NonMoving{0}, Moving{1};
constexpr uint NumLayers = 2;
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

// Custom GroupFilter implementing KHR_physics_rigid_bodies collision semantics.
// Collision filtering using bitmasks, matching the Babylon reference implementation.
// Each collision system name maps to a bit. A filter's membershipMask encodes which systems
// the body belongs to, and collideMask encodes which systems it can collide with.
// Two bodies collide iff (A.membership & B.collide) != 0 && (B.membership & A.collide) != 0.
struct CollisionMask {
    uint32_t Membership = 0, Collide = ~0u; // default: collide with everything
};

class KHRCollisionFilter : public GroupFilter {
    std::vector<CollisionMask> Masks;

public:
    explicit KHRCollisionFilter(const std::vector<CollisionFilter> &filters) { Update(filters); }

    void Update(const std::vector<CollisionFilter> &filters) {
        // Assign a bit index to each unique system name.
        std::unordered_map<std::string, uint32_t> bitIndex;
        auto getBit = [&](const std::string &name) -> uint32_t {
            auto [it, inserted] = bitIndex.try_emplace(name, uint32_t(bitIndex.size()));
            return 1u << it->second;
        };
        auto namesToMask = [&](const std::vector<std::string> &names) {
            uint32_t m = 0;
            for (auto &n : names) m |= getBit(n);
            return m;
        };

        Masks.resize(filters.size());
        for (size_t i = 0; i < filters.size(); ++i) {
            Masks[i].Membership = namesToMask(filters[i].CollisionSystems);
            if (!filters[i].CollideWithSystems.empty()) Masks[i].Collide = namesToMask(filters[i].CollideWithSystems);
            else if (!filters[i].NotCollideWithSystems.empty()) Masks[i].Collide = ~namesToMask(filters[i].NotCollideWithSystems);
            else Masks[i].Collide = ~0u;
        }
    }

    // Body-level filtering via Jolt CollisionGroup (SubGroupID = filter index).
    bool CanCollide(const CollisionGroup &g1, const CollisionGroup &g2) const override {
        if (g1.GetGroupID() == CollisionGroup::cInvalidGroup || g2.GetGroupID() == CollisionGroup::cInvalidGroup) return true;
        return MasksCollide(g1.GetSubGroupID(), g2.GetSubGroupID());
    }

    bool MasksCollide(uint32_t a, uint32_t b) const {
        if (a >= Masks.size() || b >= Masks.size()) return true;
        return (Masks[a].Membership & Masks[b].Collide) != 0 && (Masks[b].Membership & Masks[a].Collide) != 0;
    }
};

// We store the material index in Body::UserData. UINT32_MAX = no material assigned.
constexpr uint64_t NoMaterialSentinel = UINT32_MAX;

// KHR combine mode priority: Maximum(2) > Multiply(3) > Average(0) > Minimum(1).
// When two materials use different combine modes, pick the higher-priority one.
int CombineModePriority(PhysicsCombineMode m) {
    switch (m) {
        case PhysicsCombineMode::Maximum: return 3;
        case PhysicsCombineMode::Multiply: return 2;
        case PhysicsCombineMode::Average: return 1;
        case PhysicsCombineMode::Minimum: return 0;
    }
    return 1; // default = Average
}

float ApplyCombineMode(PhysicsCombineMode mode, float a, float b) {
    switch (mode) {
        case PhysicsCombineMode::Average: return (a + b) * 0.5f;
        case PhysicsCombineMode::Minimum: return std::min(a, b);
        case PhysicsCombineMode::Maximum: return std::max(a, b);
        case PhysicsCombineMode::Multiply: return a * b;
    }
    return (a + b) * 0.5f;
}

// Contact listener that:
// 1. Per-sub-shape collision filtering for compound bodies — Shape::mUserData stores filter index + 1
//    (0 = no filter), looked up in the KHRCollisionFilter's mask table. Jolt's CollisionGroup handles
//    body-level filtering; this covers compound bodies where sub-shapes have different filters.
// 2. KHR_physics_rigid_bodies friction/restitution combine modes.
class KHRContactListener : public ContactListener {
public:
    const std::vector<::PhysicsMaterial> *Materials = nullptr;
    const KHRCollisionFilter *Filter = nullptr;

    ValidateResult OnContactValidate(const Body &b1, const Body &b2, RVec3Arg, const CollideShapeResult &result) override {
        if (!Filter) return ValidateResult::AcceptAllContactsForThisBodyPair;
        uint64_t f1 = b1.GetShape()->GetSubShapeUserData(result.mSubShapeID1);
        uint64_t f2 = b2.GetShape()->GetSubShapeUserData(result.mSubShapeID2);
        if (f1 == 0 || f2 == 0) return ValidateResult::AcceptContact;
        if (!Filter->MasksCollide(uint32_t(f1 - 1), uint32_t(f2 - 1))) return ValidateResult::RejectContact;
        return ValidateResult::AcceptContact;
    }

    void OnContactAdded(const Body &b1, const Body &b2, const ContactManifold &, ContactSettings &s) override {
        CombineMaterials(b1, b2, s);
    }
    void OnContactPersisted(const Body &b1, const Body &b2, const ContactManifold &, ContactSettings &s) override {
        CombineMaterials(b1, b2, s);
    }

private:
    void CombineMaterials(const Body &b1, const Body &b2, ContactSettings &s) const {
        if (!Materials) return;
        uint64_t ud1 = b1.GetUserData(), ud2 = b2.GetUserData();
        bool has1 = ud1 != NoMaterialSentinel && ud1 < Materials->size();
        bool has2 = ud2 != NoMaterialSentinel && ud2 < Materials->size();
        if (!has1 && !has2) return; // both use Jolt defaults

        // Default combine mode is Average per KHR spec.
        auto fc1 = PhysicsCombineMode::Average, fc2 = fc1;
        auto rc1 = PhysicsCombineMode::Average, rc2 = rc1;
        float f1 = b1.GetFriction(), f2 = b2.GetFriction();
        float r1 = b1.GetRestitution(), r2 = b2.GetRestitution();
        if (has1) {
            fc1 = (*Materials)[ud1].FrictionCombine;
            rc1 = (*Materials)[ud1].RestitutionCombine;
        }
        if (has2) {
            fc2 = (*Materials)[ud2].FrictionCombine;
            rc2 = (*Materials)[ud2].RestitutionCombine;
        }

        auto pick = [](PhysicsCombineMode a, PhysicsCombineMode b) { return CombineModePriority(a) >= CombineModePriority(b) ? a : b; };
        s.mCombinedFriction = ApplyCombineMode(pick(fc1, fc2), f1, f2);
        s.mCombinedRestitution = ApplyCombineMode(pick(rc1, rc2), r1, r2);
    }
};

struct BodySnapshot {
    entt::entity Entity;
    vec3 Position;
    quat Rotation;
    vec3 Scale;
    vec3 LinearVelocity, AngularVelocity;
};
} // namespace

struct PhysicsWorld::Impl {
    TempAllocatorImpl TempAllocator{64 * 1024 * 1024};
    JobSystemThreadPool JobSystem{cMaxPhysicsJobs, cMaxPhysicsBarriers, int(std::max(1u, std::thread::hardware_concurrency() - 1))};
    BPLayerInterface BPLayerIface;
    ObjectLayerPairFilterImpl ObjectPairFilter;
    ObjectVsBPLayerFilterImpl ObjectVsBPFilter;
    PhysicsSystem System;

    std::vector<Ref<Constraint>> Constraints;
    std::vector<BodySnapshot> Snapshots;
    Ref<KHRCollisionFilter> FilterRef;
    KHRContactListener ContactListener;

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
        System.SetContactListener(&ContactListener);
    }
};

namespace {
Ref<Shape> CreateJoltShape(const PhysicsShape &shape, const Mesh *mesh) {
    switch (shape.Type) {
        case PhysicsShapeType::Box: {
            return new BoxShape(ToJolt(shape.Size * 0.5f)); // KHR spec uses full size, Jolt uses half-extents
        }
        case PhysicsShapeType::Sphere: {
            return new SphereShape(shape.Radius);
        }
        case PhysicsShapeType::Capsule: {
            if (std::abs(shape.RadiusTop - shape.RadiusBottom) < 1e-6f) return new CapsuleShape(shape.Height * 0.5f, shape.RadiusBottom);
            if (const auto result = TaperedCapsuleShapeSettings(shape.Height * 0.5f, shape.RadiusTop, shape.RadiusBottom).Create(); result.IsValid()) return result.Get();
            return Ref<Shape>(new CapsuleShape(shape.Height * 0.5f, shape.RadiusBottom));
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
            if (const auto result = settings.Create(); result.IsValid()) return result.Get();
            break;
        }
        case PhysicsShapeType::TriangleMesh: {
            if (!mesh || mesh->FaceCount() == 0) break;
            const auto verts = mesh->GetVerticesSpan();
            VertexList vertices;
            vertices.reserve(verts.size());
            for (const auto &v : verts) vertices.push_back(Float3(v.Position.x, v.Position.y, v.Position.z));
            const auto indices = mesh->CreateTriangleIndices();
            IndexedTriangleList triangles;
            triangles.reserve(indices.size() / 3);
            for (size_t i = 0; i + 2 < indices.size(); i += 3) triangles.emplace_back(indices[i], indices[i + 1], indices[i + 2]);
            MeshShapeSettings settings{std::move(vertices), std::move(triangles)};
            if (const auto result = settings.Create(); result.IsValid()) return result.Get();
            break;
        }
    }
    return new BoxShape(Vec3(0.5f, 0.5f, 0.5f));
}

// Apply motion and material properties to body creation settings.
void ApplyPhysicsProperties(BodyCreationSettings &bcs, const PhysicsMotion *motion, const PhysicsCollider *collider, const std::vector<::PhysicsMaterial> &materials) {
    bcs.mUserData = NoMaterialSentinel;
    if (motion) {
        if (motion->Mass.has_value() && bcs.mMotionType == EMotionType::Dynamic) {
            bcs.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
            bcs.mMassPropertiesOverride.mMass = *motion->Mass;
        }
        bcs.mLinearVelocity = ToJolt(motion->LinearVelocity);
        bcs.mAngularVelocity = ToJolt(motion->AngularVelocity);
        bcs.mGravityFactor = motion->GravityFactor;
    }
    if (collider && collider->PhysicsMaterialIndex.has_value() && *collider->PhysicsMaterialIndex < materials.size()) {
        const auto &mat = materials[*collider->PhysicsMaterialIndex];
        bcs.mFriction = mat.DynamicFriction;
        bcs.mRestitution = mat.Restitution;
        bcs.mUserData = *collider->PhysicsMaterialIndex;
    }
}

// Find the nearest ancestor (or self) that has a PhysicsBodyHandle.
entt::entity FindBodyAncestor(const entt::registry &r, entt::entity e) {
    for (; e != entt::null;) {
        if (r.all_of<PhysicsBodyHandle>(e)) return e;
        const auto parent = GetParentEntity(r, e);
        if (parent == e) break; // root node — stop
        e = parent;
    }
    return entt::null;
}

void ConfigureJointSettings(SixDOFConstraintSettings &settings, const PhysicsJointDef &def) {
    // Default: all axes fixed (locked to the attachment frame)
    for (int a = 0; a < SixDOFConstraintSettings::EAxis::Num; ++a) {
        settings.MakeFixedAxis(static_cast<SixDOFConstraintSettings::EAxis>(a));
    }

    // Apply limits — each limit entry may cover multiple axes
    for (const auto &limit : def.Limits) {
        const auto configure_axis = [&](SixDOFConstraintSettings::EAxis axis) {
            if (!limit.Min && !limit.Max) settings.MakeFreeAxis(axis);
            else settings.SetLimitedAxis(axis, limit.Min.value_or(-FLT_MAX), limit.Max.value_or(FLT_MAX));
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
        const int axis_index = (drive.Type == PhysicsDriveType::Linear ? 0 : 3) + drive.Axis;
        const auto axis = static_cast<SixDOFConstraintSettings::EAxis>(axis_index);
        auto &motor = settings.mMotorSettings[axis_index];
        if (drive.Stiffness > 0.0f || drive.Damping > 0.0f) {
            motor.mSpringSettings = SpringSettings(ESpringMode::StiffnessAndDamping, drive.Stiffness, drive.Damping);
        }
        if (drive.Type == PhysicsDriveType::Linear) motor.SetForceLimit(drive.MaxForce);
        else motor.SetTorqueLimit(drive.MaxForce);
        // Ensure the axis isn't fixed so the motor can act
        if (settings.IsFixedAxis(axis)) settings.MakeFreeAxis(axis);
    }
}

// After constraint creation, set motor states and targets from drives.
void ApplyDriveTargets(SixDOFConstraint &constraint, const PhysicsJointDef &def) {
    auto target_pos = Vec3::sZero(), target_vel = Vec3::sZero();
    auto target_ang_vel = Vec3::sZero();
    auto target_orient = Quat::sIdentity();
    bool has_orient_target = false;
    for (const auto &drive : def.Drives) {
        const int axis_index = (drive.Type == PhysicsDriveType::Linear ? 0 : 3) + drive.Axis;
        const auto axis = static_cast<SixDOFConstraintSettings::EAxis>(axis_index);
        const bool has_position = drive.Stiffness > 0.0f || drive.PositionTarget != 0.0f;
        const bool has_velocity = drive.VelocityTarget != 0.0f;
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
            if (drive.Type == PhysicsDriveType::Linear) target_vel.SetComponent(drive.Axis, drive.VelocityTarget);
            else target_ang_vel.SetComponent(drive.Axis, drive.VelocityTarget);
        }
    }
    constraint.SetTargetPositionCS(target_pos);
    constraint.SetTargetVelocityCS(target_vel);
    constraint.SetTargetAngularVelocityCS(target_ang_vel);
    if (has_orient_target) constraint.SetTargetOrientationCS(target_orient);
}

struct JoltInit {
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
} // namespace

PhysicsWorld::PhysicsWorld() : P(std::make_unique<Impl>()) {}
PhysicsWorld::~PhysicsWorld() = default;

bool PhysicsWorld::HasBodies() const { return P->System.GetNumBodies() > 0; }
uint32_t PhysicsWorld::BodyCount() const { return P->System.GetNumBodies(); }

void PhysicsWorld::UpdateFilterTable() {
    if (P->FilterRef) P->FilterRef->Update(Filters);
}

bool PhysicsWorld::DoFiltersCollide(uint32_t a, uint32_t b) const {
    return !P->FilterRef || P->FilterRef->MasksCollide(a, b);
}

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

    // Point the contact listener at our materials for KHR friction/restitution combining.
    P->ContactListener.Materials = &Materials;

    // Build collision filter bitmasks from string-based filter definitions.
    P->FilterRef = Filters.empty() ? nullptr : new KHRCollisionFilter(Filters);
    P->ContactListener.Filter = P->FilterRef.GetPtr();

    // Per the KHR_physics_rigid_bodies spec, each collider belongs to its nearest ancestor
    // (or self) with PhysicsMotion. Colliders with no motion ancestor are static.
    // Group colliders by their owning motion entity to build compound shapes.
    auto find_motion_owner = [&](entt::entity e) -> entt::entity {
        for (auto cur = e; cur != entt::null;) {
            if (r.all_of<PhysicsMotion>(cur)) return cur;
            auto parent = GetParentEntity(r, cur);
            if (parent == cur) break; // root node
            cur = parent;
        }
        return entt::null;
    };

    std::unordered_map<entt::entity, std::vector<entt::entity>> motion_colliders; // motion entity → collider entities
    std::vector<entt::entity> static_colliders;
    for (auto [entity, collider] : r.view<const PhysicsCollider>().each()) {
        if (auto owner = find_motion_owner(entity); owner != entt::null) motion_colliders[owner].push_back(entity);
        else static_colliders.push_back(entity);
    }

    auto make_jolt_shape = [&](const PhysicsCollider &collider, entt::entity entity, bool is_dynamic) -> Ref<Shape> {
        auto effective_shape = collider.Shape;
        // Jolt doesn't support MeshShape vs MeshShape collision; promote to ConvexHull for dynamic bodies.
        if (effective_shape.Type == PhysicsShapeType::TriangleMesh && is_dynamic) effective_shape.Type = PhysicsShapeType::ConvexHull;
        const auto *mesh = effective_shape.MeshEntity ? r.try_get<const Mesh>(*effective_shape.MeshEntity) : nullptr;
        auto shape = CreateJoltShape(effective_shape, mesh);
        if (!shape) return {};
        // Store filter index + 1 on the leaf shape (before decorator wrapping) so
        // GetSubShapeUserData returns it for per-sub-shape filtering in OnContactValidate.
        if (collider.CollisionFilterIndex.has_value()) shape->SetUserData(*collider.CollisionFilterIndex + 1);
        const auto *t = r.try_get<const Transform>(entity);
        if (t && (t->S.x != 1.0f || t->S.y != 1.0f || t->S.z != 1.0f)) shape = new ScaledShape(shape, ToJolt(t->S));
        return shape;
    };

    auto create_body = [&](Ref<Shape> shape, const Transform *t, EMotionType motion_type,
                           const PhysicsMotion *motion, const PhysicsCollider *collider, entt::entity body_entity) {
        if (!shape) return;
        const auto pos = t ? RVec3(t->P.x, t->P.y, t->P.z) : RVec3::sZero();
        const auto rot = t ? ToJoltQuat(t->R) : Quat::sIdentity();
        const auto layer = motion_type == EMotionType::Static ? Layers::NonMoving : Layers::Moving;

        BodyCreationSettings bcs(shape, pos, rot, motion_type, layer);
        ApplyPhysicsProperties(bcs, motion, collider, Materials);
        if (P->FilterRef && collider && collider->CollisionFilterIndex.has_value() && *collider->CollisionFilterIndex < Filters.size()) {
            bcs.mCollisionGroup = CollisionGroup(P->FilterRef, 0, *collider->CollisionFilterIndex);
        }
        if (const auto *body = bi.CreateBody(bcs)) {
            bi.AddBody(body->GetID(), motion_type == EMotionType::Static ? EActivation::DontActivate : EActivation::Activate);
            r.emplace_or_replace<PhysicsBodyHandle>(body_entity, PhysicsBodyHandle{body->GetID().GetIndexAndSequenceNumber()});
        }
    };

    // Static colliders: one body each.
    for (auto entity : static_colliders) {
        const auto &collider = r.get<const PhysicsCollider>(entity);
        create_body(make_jolt_shape(collider, entity, false), r.try_get<const Transform>(entity), EMotionType::Static, nullptr, &collider, entity);
    }

    // Dynamic/kinematic bodies: one body per motion entity, compound if multiple colliders.
    for (auto &[motion_entity, colliders] : motion_colliders) {
        const auto &motion = r.get<const PhysicsMotion>(motion_entity);
        const auto motion_type = motion.IsKinematic ? EMotionType::Kinematic : EMotionType::Dynamic;
        const bool is_dynamic = motion_type != EMotionType::Static;
        const auto *body_transform = r.try_get<const Transform>(motion_entity);

        Ref<Shape> shape;
        const PhysicsCollider *single_collider = nullptr;
        if (colliders.size() == 1 && colliders[0] == motion_entity) {
            // Single collider on the motion node itself — use directly.
            single_collider = &r.get<const PhysicsCollider>(motion_entity);
            shape = make_jolt_shape(*single_collider, motion_entity, is_dynamic);
        } else {
            // Compound: gather child collider shapes relative to the motion node.
            const auto inv_parent = body_transform ? glm::inverse(glm::translate(mat4{1}, body_transform->P) * glm::mat4_cast(glm::normalize(body_transform->R))) : mat4{1};

            StaticCompoundShapeSettings compound;
            for (auto ce : colliders) {
                const auto &child_collider = r.get<const PhysicsCollider>(ce);
                const auto sub = make_jolt_shape(child_collider, ce, is_dynamic);
                if (!sub) continue;
                // Compute collider transform relative to the motion node.
                const auto *wt = r.try_get<const WorldTransform>(ce);
                const auto world = wt ? ToMatrix(*wt) : mat4{1};
                const auto rel = inv_parent * world;
                compound.AddShape(ToJolt(vec3(rel[3])), ToJoltQuat(glm::normalize(glm::quat_cast(glm::mat3(rel)))), sub);
            }
            if (compound.mSubShapes.empty()) continue;

            if (const auto result = compound.Create(); result.IsValid()) shape = result.Get();
        }

        create_body(shape, body_transform, motion_type, &motion, single_collider, motion_entity);
    }

    // Create trigger sensor bodies (for entities without a collider body)
    for (auto [entity, trigger] : r.view<const PhysicsTrigger>().each()) {
        if (!trigger.Shape.has_value()) continue; // Compound trigger — no own geometry
        if (r.all_of<PhysicsBodyHandle>(entity)) continue; // Already has body from collider loop

        const auto *mesh = trigger.Shape->MeshEntity ? r.try_get<const Mesh>(*trigger.Shape->MeshEntity) : nullptr;
        auto shape = CreateJoltShape(*trigger.Shape, mesh);
        if (!shape) continue;

        const auto *transform = r.try_get<const Transform>(entity);
        const auto pos = transform ? RVec3(transform->P.x, transform->P.y, transform->P.z) : RVec3::sZero();
        const auto rot = transform ? ToJoltQuat(transform->R) : Quat::sIdentity();
        if (transform && (transform->S.x != 1.0f || transform->S.y != 1.0f || transform->S.z != 1.0f)) {
            shape = new ScaledShape(shape, ToJolt(transform->S));
        }

        const auto *motion = r.try_get<const PhysicsMotion>(entity);
        auto motion_type = EMotionType::Static;
        auto layer = Layers::NonMoving;
        if (motion) {
            motion_type = motion->IsKinematic ? EMotionType::Kinematic : EMotionType::Dynamic;
            layer = Layers::Moving;
        }

        BodyCreationSettings bcs{shape, pos, rot, motion_type, layer};
        bcs.mIsSensor = true;
        ApplyPhysicsProperties(bcs, motion, nullptr, Materials);
        if (P->FilterRef && trigger.CollisionFilterIndex.has_value() && *trigger.CollisionFilterIndex < Filters.size()) {
            bcs.mCollisionGroup = CollisionGroup(P->FilterRef, 0, *trigger.CollisionFilterIndex);
        }
        if (const auto *body = bi.CreateBody(bcs)) {
            bi.AddBody(body->GetID(), motion_type == EMotionType::Static ? EActivation::DontActivate : EActivation::Activate);
            r.emplace_or_replace<PhysicsBodyHandle>(entity, PhysicsBodyHandle{body->GetID().GetIndexAndSequenceNumber()});
        }
    }

    // Create constraints from PhysicsJoint components
    const auto &lock_iface = P->System.GetBodyLockInterfaceNoLock();
    for (auto [entity, joint] : r.view<const PhysicsJoint>().each()) {
        if (joint.ConnectedNode == entt::null) continue;
        if (joint.JointDefIndex >= JointDefs.size()) continue;

        // Body 1: nearest ancestor (or self) with a physics body
        const auto body1_entity = FindBodyAncestor(r, entity);
        // Body 2: the connected node (should itself be a body, or find its body ancestor)
        const auto body2_entity = FindBodyAncestor(r, joint.ConnectedNode);
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
            const RVec3 anchor{jt->P.x, jt->P.y, jt->P.z};
            settings.mPosition1 = settings.mPosition2 = anchor;
            // Derive constraint frame axes from joint node rotation
            const auto rot_mat = glm::mat3_cast(glm::normalize(jt->R));
            settings.mAxisX1 = settings.mAxisX2 = Vec3(rot_mat[0].x, rot_mat[0].y, rot_mat[0].z);
            settings.mAxisY1 = settings.mAxisY2 = Vec3(rot_mat[1].x, rot_mat[1].y, rot_mat[1].z);
        }

        ConfigureJointSettings(settings, def);

        const BodyLockWrite lock1{lock_iface, BodyID(h1->BodyId)};
        const BodyLockWrite lock2{lock_iface, BodyID(h2->BodyId)};
        if (!lock1.Succeeded() || !lock2.Succeeded()) continue;

        auto *constraint = static_cast<SixDOFConstraint *>(settings.Create(lock1.GetBody(), lock2.GetBody()));
        ApplyDriveTargets(*constraint, def);
        P->System.AddConstraint(constraint);
        P->Constraints.emplace_back(constraint);
    }

    P->System.OptimizeBroadPhase();
}

void PhysicsWorld::AddBody(entt::registry &r, entt::entity entity) {
    const auto *collider = r.try_get<const PhysicsCollider>(entity);
    const auto *trigger = r.try_get<const PhysicsTrigger>(entity);

    // Collider takes priority; trigger-only needs a shape
    const bool is_sensor = !collider;
    if (!collider && (!trigger || !trigger->Shape.has_value())) return;

    const auto &shape_ref = collider ? collider->Shape : *trigger->Shape;
    const auto *mesh = shape_ref.MeshEntity ? r.try_get<const Mesh>(*shape_ref.MeshEntity) : nullptr;
    auto shape = CreateJoltShape(shape_ref, mesh);
    if (!shape) return;

    const auto *transform = r.try_get<const Transform>(entity);
    const auto pos = transform ? RVec3(transform->P.x, transform->P.y, transform->P.z) : RVec3::sZero();
    const auto rot = transform ? ToJoltQuat(transform->R) : Quat::sIdentity();
    if (transform && (transform->S.x != 1.0f || transform->S.y != 1.0f || transform->S.z != 1.0f)) {
        shape = new ScaledShape(shape, ToJolt(transform->S));
    }

    const auto *motion = r.try_get<const PhysicsMotion>(entity);
    auto motion_type = EMotionType::Static;
    auto layer = Layers::NonMoving;
    if (motion) {
        motion_type = motion->IsKinematic ? EMotionType::Kinematic : EMotionType::Dynamic;
        layer = Layers::Moving;
    }

    BodyCreationSettings bcs{shape, pos, rot, motion_type, layer};
    bcs.mIsSensor = is_sensor;
    ApplyPhysicsProperties(bcs, motion, collider, Materials);
    const auto filter_idx = collider ? collider->CollisionFilterIndex : trigger->CollisionFilterIndex;
    if (P->FilterRef && filter_idx.has_value() && *filter_idx < Filters.size()) {
        bcs.mCollisionGroup = CollisionGroup(P->FilterRef, 0, *filter_idx);
    }

    auto &bi = P->System.GetBodyInterface();
    if (auto *body = bi.CreateBody(bcs)) {
        bi.AddBody(body->GetID(), motion_type == EMotionType::Static ? EActivation::DontActivate : EActivation::Activate);
        r.emplace_or_replace<PhysicsBodyHandle>(entity, PhysicsBodyHandle{body->GetID().GetIndexAndSequenceNumber()});
    }
}

void PhysicsWorld::RemoveBody(entt::registry &r, entt::entity entity) {
    const auto *handle = r.try_get<const PhysicsBodyHandle>(entity);
    if (!handle || handle->BodyId == UINT32_MAX) return;

    auto &bi = P->System.GetBodyInterface();
    BodyID id{handle->BodyId};
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
        BodyID id{handle.BodyId};
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
        BodyID id{handle.BodyId};
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
