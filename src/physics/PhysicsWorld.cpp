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
#include "Jolt/Physics/Collision/Shape/DecoratedShape.h"
#include "Jolt/Physics/Collision/Shape/MeshShape.h"
#include "Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h"
#include "Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h"
#include "Jolt/Physics/Collision/Shape/ScaledShape.h"
#include "Jolt/Physics/Collision/Shape/SphereShape.h"
#include "Jolt/Physics/Collision/Shape/StaticCompoundShape.h"
#include "Jolt/Physics/Collision/Shape/TaperedCapsuleShape.h"
#include "Jolt/Physics/Collision/Shape/TaperedCylinderShape.h"
#include "Jolt/Physics/Collision/SimShapeFilter.h"
#include "Jolt/Physics/Constraints/SixDOFConstraint.h"
#include "Jolt/Physics/PhysicsSettings.h"
#include "Jolt/Physics/PhysicsSystem.h"
#include "Jolt/RegisterTypes.h"

JPH_SUPPRESS_WARNINGS

#include "PhysicsWorld.h"
#include "SceneTree.h"
#include "mesh/Mesh.h"

#include <entt/entity/registry.hpp>

#include <algorithm>
#include <functional>
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
        Mapping[Layers::NonMoving] = BPLayers::NonMoving;
        Mapping[Layers::Moving] = BPLayers::Moving;
    }
    uint GetNumBroadPhaseLayers() const override { return BPLayers::NumLayers; }
    BroadPhaseLayer GetBroadPhaseLayer(ObjectLayer layer) const override { return Mapping[layer]; }
    const char *GetBroadPhaseLayerName(BroadPhaseLayer layer) const override {
        switch (static_cast<BroadPhaseLayer::Type>(layer)) {
            case static_cast<BroadPhaseLayer::Type>(0): return "NON_MOVING";
            case static_cast<BroadPhaseLayer::Type>(1): return "MOVING";
            default: return "INVALID";
        }
    }

private:
    BroadPhaseLayer Mapping[Layers::NumLayers];
};

class ObjectLayerPairFilterImpl : public ObjectLayerPairFilter {
public:
    bool ShouldCollide(ObjectLayer l1, ObjectLayer l2) const override {
        if (l1 == Layers::NonMoving) return l2 == Layers::Moving;
        return true; // Moving collides with everything
    }
};

class ObjectVsBPLayerFilterImpl : public ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(ObjectLayer l1, BroadPhaseLayer l2) const override {
        if (l1 == Layers::NonMoving) return l2 == BPLayers::Moving;
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
    // Each body gets a unique SubGroupID via RegisterBody(). This maps SubGroupID → KHR filter index.
    std::vector<uint32_t> BodyFilterIndices; // UINT32_MAX = no KHR filter
    // Pairs of SubGroupIDs that should not collide (from joints with EnableCollision=false).
    std::vector<std::pair<uint32_t, uint32_t>> DisabledPairs; // sorted after FinalizeDisabledPairs()

public:
    explicit KHRCollisionFilter(const std::vector<CollisionFilter> &filters) { Update(filters); }

    void Update(const std::vector<CollisionFilter> &filters) {
        // Assign a bit index to each unique system name.
        std::unordered_map<std::string, uint32_t> bit_index;
        auto get_bit = [&](const std::string &name) -> uint32_t {
            auto [it, inserted] = bit_index.try_emplace(name, bit_index.size());
            return 1u << it->second;
        };
        auto names_to_mask = [&](const std::vector<std::string> &names) {
            uint32_t m = 0;
            for (auto &n : names) m |= get_bit(n);
            return m;
        };

        Masks.resize(filters.size());
        for (size_t i = 0; i < filters.size(); ++i) {
            Masks[i].Membership = names_to_mask(filters[i].CollisionSystems);
            if (!filters[i].CollideWithSystems.empty()) Masks[i].Collide = names_to_mask(filters[i].CollideWithSystems);
            else if (!filters[i].NotCollideWithSystems.empty()) Masks[i].Collide = ~names_to_mask(filters[i].NotCollideWithSystems);
            else Masks[i].Collide = ~0u;
        }
    }

    // Register a body and return its unique SubGroupID. filter_index is the KHR collision filter index.
    uint32_t RegisterBody(std::optional<uint32_t> filter_index = {}) {
        const uint32_t id = BodyFilterIndices.size();
        BodyFilterIndices.emplace_back(filter_index.value_or(UINT32_MAX));
        return id;
    }

    void SetBodyFilter(uint32_t sub_group_id, std::optional<uint32_t> filter_index) {
        if (sub_group_id < BodyFilterIndices.size()) BodyFilterIndices[sub_group_id] = filter_index.value_or(UINT32_MAX);
    }

    void Reset() {
        BodyFilterIndices.clear();
        DisabledPairs.clear();
    }

    // Wipe DisabledPairs without invalidating SubGroupIDs of existing bodies.
    void ResetDisabledPairs() { DisabledPairs.clear(); }
    // Mark a pair of bodies (by SubGroupID) as non-colliding. Call FinalizeDisabledPairs() when done.
    void DisableCollision(uint32_t a, uint32_t b) { DisabledPairs.emplace_back(std::min(a, b), std::max(a, b)); }
    void FinalizeDisabledPairs() { std::sort(DisabledPairs.begin(), DisabledPairs.end()); }

    bool CanCollide(const CollisionGroup &g1, const CollisionGroup &g2) const override {
        if (g1.GetGroupID() == CollisionGroup::cInvalidGroup || g2.GetGroupID() == CollisionGroup::cInvalidGroup) return true;
        const uint32_t s1 = g1.GetSubGroupID(), s2 = g2.GetSubGroupID();
        // Check joint pair exclusion
        if (!DisabledPairs.empty()) {
            const auto key = std::make_pair(std::min(s1, s2), std::max(s1, s2));
            if (std::binary_search(DisabledPairs.begin(), DisabledPairs.end(), key)) return false;
        }
        // Check KHR collision filter masks
        const uint32_t f1 = s1 < BodyFilterIndices.size() ? BodyFilterIndices[s1] : UINT32_MAX;
        const uint32_t f2 = s2 < BodyFilterIndices.size() ? BodyFilterIndices[s2] : UINT32_MAX;
        if (f1 == UINT32_MAX || f2 == UINT32_MAX) return true;
        return MasksCollide(f1, f2);
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

// Jolt doesn't support MeshShape vs MeshShape collision — reject before dispatch.
class MeshVsMeshShapeFilter : public SimShapeFilter {
public:
    bool ShouldCollide(const Body &, const Shape *shape1, const SubShapeID &, const Body &, const Shape *shape2, const SubShapeID &) const override {
        return !(shape1->GetSubType() == EShapeSubType::Mesh && shape2->GetSubType() == EShapeSubType::Mesh);
    }
};

// Contact listener that:
// 1. Per-sub-shape collision filtering for compound bodies — Shape::mUserData stores filter index + 1
//    (0 = no filter), looked up in the KHRCollisionFilter's mask table. Jolt's CollisionGroup handles
//    body-level filtering; this covers compound bodies where sub-shapes have different filters.
// 2. KHR_physics_rigid_bodies friction/restitution combine modes.
class KHRContactListener : public ContactListener {
public:
    const std::vector<::PhysicsMaterial> *Materials{nullptr};
    const KHRCollisionFilter *Filter{nullptr};

    ValidateResult OnContactValidate(const Body &b1, const Body &b2, RVec3Arg, const CollideShapeResult &result) override {
        if (!Filter) return ValidateResult::AcceptAllContactsForThisBodyPair;
        const uint64_t f1 = b1.GetShape()->GetSubShapeUserData(result.mSubShapeID1);
        const uint64_t f2 = b2.GetShape()->GetSubShapeUserData(result.mSubShapeID2);
        if (f1 == 0 || f2 == 0) return ValidateResult::AcceptContact;
        if (!Filter->MasksCollide(f1 - 1, f2 - 1)) return ValidateResult::RejectContact;
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
        const uint64_t ud1 = b1.GetUserData(), ud2 = b2.GetUserData();
        const bool has1 = ud1 != NoMaterialSentinel && ud1 < Materials->size();
        const bool has2 = ud2 != NoMaterialSentinel && ud2 < Materials->size();
        if (!has1 && !has2) return; // both use Jolt defaults

        // Default combine mode is Average per KHR spec.
        const auto fc1 = has1 ? (*Materials)[ud1].FrictionCombine : PhysicsCombineMode::Average;
        const auto rc1 = has1 ? (*Materials)[ud1].RestitutionCombine : PhysicsCombineMode::Average;
        const auto fc2 = has2 ? (*Materials)[ud2].FrictionCombine : PhysicsCombineMode::Average;
        const auto rc2 = has2 ? (*Materials)[ud2].RestitutionCombine : PhysicsCombineMode::Average;
        const float f1 = b1.GetFriction(), f2 = b2.GetFriction();
        const float r1 = b1.GetRestitution(), r2 = b2.GetRestitution();

        const auto pick = [](PhysicsCombineMode a, PhysicsCombineMode b) { return CombineModePriority(a) >= CombineModePriority(b) ? a : b; };
        s.mCombinedFriction = ApplyCombineMode(pick(fc1, fc2), f1, f2);
        s.mCombinedRestitution = ApplyCombineMode(pick(rc1, rc2), r1, r2);
    }
};

struct BodySnapshot {
    entt::entity Entity;
    vec3 Position;
    quat Rotation;
    vec3 Scale;
    PhysicsVelocity Velocity;
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
    MeshVsMeshShapeFilter MeshFilter;

    std::unordered_map<entt::entity, uint32_t> BodySubGroups; // entity -> KHRCollisionFilter SubGroupID.

    bool JointsDirty{false};

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
        System.SetSimShapeFilter(&MeshFilter);
        // Jolt MeshShape is single-sided by default.
        // Enable back-face collision so mesh colliders block from both sides.
        System.SetSimCollideBodyVsBody([](const Body &b1, const Body &b2, Mat44Arg t1, Mat44Arg t2,
                                          CollideShapeSettings &settings, CollideShapeCollector &collector,
                                          const ShapeFilter &filter) {
            settings.mBackFaceMode = EBackFaceMode::CollideWithBackFaces;
            PhysicsSystem::sDefaultSimCollideBodyVsBody(b1, b2, t1, t2, settings, collector, filter);
        });
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
            if (std::abs(shape.RadiusTop - shape.RadiusBottom) < 1e-6f) return new CylinderShape(shape.Height * 0.5f, shape.RadiusBottom);
            if (const auto result = TaperedCylinderShapeSettings(shape.Height * 0.5f, shape.RadiusTop, shape.RadiusBottom).Create(); result.IsValid()) return result.Get();
            return new CylinderShape(shape.Height * 0.5f, std::max(shape.RadiusTop, shape.RadiusBottom));
        }
        case PhysicsShapeType::ConvexHull: {
            if (!mesh || mesh->VertexCount() == 0) break;
            auto verts = mesh->GetVerticesSpan();
            // Jolt Vec3 is a 16-byte SIMD type — must convert from interleaved Vertex positions
            Array<Vec3> points;
            points.reserve(int(verts.size()));
            for (const auto &v : verts) points.emplace_back(v.Position.x, v.Position.y, v.Position.z);
            ConvexHullShapeSettings settings(points.data(), int(points.size()));
            if (const auto result = settings.Create(); result.IsValid()) return result.Get();
            break;
        }
        case PhysicsShapeType::TriangleMesh: {
            if (!mesh || mesh->FaceCount() == 0) break;
            const auto verts = mesh->GetVerticesSpan();
            VertexList vertices;
            vertices.reserve(verts.size());
            for (const auto &v : verts) vertices.emplace_back(v.Position.x, v.Position.y, v.Position.z);
            const auto indices = mesh->CreateTriangleIndices();
            IndexedTriangleList triangles;
            triangles.reserve(indices.size() / 3);
            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                triangles.emplace_back(indices[i], indices[i + 1], indices[i + 2]);
            }
            MeshShapeSettings settings{std::move(vertices), std::move(triangles)};
            if (const auto result = settings.Create(); result.IsValid()) return result.Get();
            break;
        }
    }
    return new BoxShape(Vec3(0.5f, 0.5f, 0.5f));
}

// Apply motion and material properties to body creation settings.
void ApplyPhysicsProperties(BodyCreationSettings &bcs, const PhysicsMotion *motion, const PhysicsVelocity *velocity, std::optional<PhysicsShapeType> shape_type, const ColliderMaterial *material, const std::vector<::PhysicsMaterial> &materials) {
    bcs.mUserData = NoMaterialSentinel;
    if (motion) {
        if (bcs.mMotionType == EMotionType::Dynamic) {
            // KHR spec §128: an explicit mass of 0 means infinite (lock translation).
            if (motion->Mass == 0.0f) {
                bcs.mAllowedDOFs = EAllowedDOFs::All & ~(EAllowedDOFs::TranslationX | EAllowedDOFs::TranslationY | EAllowedDOFs::TranslationZ);
            }
            const float mass = motion->Mass.value_or(DefaultMass);
            bcs.mMassPropertiesOverride.mMass = mass > 0.0f ? mass : DefaultMass;
            bcs.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
        }
        // MeshShape can't compute mass properties — provide placeholders for any non-static body.
        const bool is_mesh = shape_type == PhysicsShapeType::TriangleMesh;
        if (is_mesh && bcs.mMotionType != EMotionType::Static) {
            const float m = motion->Mass.value_or(DefaultMass);
            bcs.mOverrideMassProperties = EOverrideMassProperties::MassAndInertiaProvided;
            bcs.mMassPropertiesOverride.mMass = m > 0.0f ? m : DefaultMass;
            bcs.mMassPropertiesOverride.mInertia = Mat44::sScale(Vec3::sReplicate(bcs.mMassPropertiesOverride.mMass / 6.0f));
        }
        if (velocity) {
            bcs.mLinearVelocity = ToJolt(velocity->Linear);
            bcs.mAngularVelocity = ToJolt(velocity->Angular);
        }
        bcs.mGravityFactor = motion->GravityFactor;
        bcs.mLinearDamping = motion->LinearDamping;
        bcs.mAngularDamping = motion->AngularDamping;
    }
    if (material && material->PhysicsMaterialIndex.has_value() && *material->PhysicsMaterialIndex < materials.size()) {
        const auto &mat = materials[*material->PhysicsMaterialIndex];
        bcs.mFriction = mat.DynamicFriction;
        bcs.mRestitution = mat.Restitution;
        bcs.mUserData = *material->PhysicsMaterialIndex;
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
    // Per KHR_physics_rigid_bodies, only axes mentioned in limits are constrained; all others are free.
    for (int a = 0; a < SixDOFConstraintSettings::EAxis::Num; ++a) {
        settings.MakeFreeAxis(static_cast<SixDOFConstraintSettings::EAxis>(a));
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
                static const Vec3 axes[]{Vec3::sAxisX(), Vec3::sAxisY(), Vec3::sAxisZ()};
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

PhysicsWorld::PhysicsWorld() : P(std::make_unique<Impl>()) {
    // Materials and FilterRef are stable for the PhysicsWorld lifetime, so the contact listener
    // can hold raw pointers to them.
    P->ContactListener.Materials = &Materials;
    P->FilterRef = new KHRCollisionFilter(Filters);
    P->ContactListener.Filter = P->FilterRef.GetPtr();
}
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
    // Clear existing constraints and bodies.
    for (auto &c : P->Constraints) P->System.RemoveConstraint(c);
    P->Constraints.clear();
    auto &bi = P->System.GetBodyInterface();
    for (auto [entity, handle] : r.view<PhysicsBodyHandle>().each()) {
        bi.RemoveBody(BodyID{handle.BodyId});
        bi.DestroyBody(BodyID{handle.BodyId});
    }
    r.clear<PhysicsBodyHandle>();

    // Refresh the filter's mask table and clear per-body state in place; the FilterRef pointer
    // (held by the contact listener) must not be reassigned.
    P->FilterRef->Update(Filters);
    P->FilterRef->Reset();
    P->BodySubGroups.clear();

    RecomputeSceneScale(r);

    // AddBody dedups by PhysicsBodyHandle and skips colliders absorbed into a parent compound,
    // so iterating all three views is safe.
    for (auto entity : r.view<PhysicsMotion>()) AddBody(r, entity);
    for (auto entity : r.view<ColliderShape>()) AddBody(r, entity);
    for (auto entity : r.view<PhysicsTrigger>()) AddBody(r, entity);
    // The joint loop below absorbs all current joint state; reset the dirty bit so the next
    // FlushJoints tick starts clean.
    P->JointsDirty = false;

    for (auto entity : r.view<const PhysicsJoint>()) BuildJoint(r, entity);
    P->FilterRef->FinalizeDisabledPairs();

    P->System.OptimizeBroadPhase();
}

void PhysicsWorld::RecomputeSceneScale(const entt::registry &r) {
    // Scale physics tolerances to scene size — default 0.02m slop is too large for small scenes.
    float min_dim = std::numeric_limits<float>::max();
    for (auto [entity, collider] : r.view<const ColliderShape>().each()) {
        const auto &s = collider.Shape;
        switch (s.Type) {
            case PhysicsShapeType::Sphere: min_dim = std::min(min_dim, s.Radius); break;
            case PhysicsShapeType::Box: min_dim = std::min(min_dim, std::min({s.Size.x, s.Size.y, s.Size.z})); break;
            case PhysicsShapeType::Capsule:
            case PhysicsShapeType::Cylinder: min_dim = std::min(min_dim, std::min({s.RadiusTop, s.RadiusBottom, s.Height})); break;
            default: break; // Mesh shapes — no analytic size
        }
    }
    auto settings = P->System.GetPhysicsSettings();
    if (min_dim < std::numeric_limits<float>::max()) {
        settings.mPenetrationSlop = std::min(settings.mPenetrationSlop, min_dim * 0.02f);
        settings.mSpeculativeContactDistance = std::min(settings.mSpeculativeContactDistance, min_dim * 0.02f);
    }
    P->System.SetPhysicsSettings(settings);
}

void PhysicsWorld::BuildJoint(const entt::registry &r, entt::entity entity) {
    const auto *joint_p = r.try_get<const PhysicsJoint>(entity);
    if (!joint_p) return;
    const auto &joint = *joint_p;
    if (joint.ConnectedNode == entt::null) return;
    if (joint.JointDefIndex >= JointDefs.size()) return;

    // Body 1: nearest ancestor (or self) with a physics body
    const auto body1_entity = FindBodyAncestor(r, entity);
    if (body1_entity == entt::null) return;
    const auto *h1 = r.try_get<const PhysicsBodyHandle>(body1_entity);
    if (!h1) return;

    // Body 2: find ancestor body, or constrain to world if none exists.
    const auto body2_entity = FindBodyAncestor(r, joint.ConnectedNode);
    const auto *h2 = body2_entity != entt::null ? r.try_get<const PhysicsBodyHandle>(body2_entity) : nullptr;

    const auto &def = JointDefs[joint.JointDefIndex];

    SixDOFConstraintSettings settings;
    settings.mSpace = EConstraintSpace::WorldSpace;
    // Default Cone swing couples Y/Z into an ellipse, which degenerates when one
    // axis is locked. Pyramid gives independent per-axis angular limits.
    settings.mSwingType = ESwingType::Pyramid;

    // Attachment frames from the joint node (A) and connected node (B) world transforms.
    if (const auto *jt = r.try_get<const Transform>(entity)) {
        const RVec3 anchor{jt->P.x, jt->P.y, jt->P.z};
        settings.mPosition1 = settings.mPosition2 = anchor;
        const auto rot_mat = glm::mat3_cast(glm::normalize(jt->R));
        settings.mAxisX1 = settings.mAxisX2 = Vec3(rot_mat[0].x, rot_mat[0].y, rot_mat[0].z);
        settings.mAxisY1 = settings.mAxisY2 = Vec3(rot_mat[1].x, rot_mat[1].y, rot_mat[1].z);
    }
    if (const auto *ct = r.try_get<const Transform>(joint.ConnectedNode)) {
        settings.mPosition2 = RVec3{ct->P.x, ct->P.y, ct->P.z};
        const auto rot_mat = glm::mat3_cast(glm::normalize(ct->R));
        settings.mAxisX2 = Vec3(rot_mat[0].x, rot_mat[0].y, rot_mat[0].z);
        settings.mAxisY2 = Vec3(rot_mat[1].x, rot_mat[1].y, rot_mat[1].z);
    }

    ConfigureJointSettings(settings, def);

    const auto &lock_iface = P->System.GetBodyLockInterfaceNoLock();
    const BodyLockWrite lock1{lock_iface, BodyID(h1->BodyId)};
    std::optional<BodyLockWrite> lock2_opt;
    if (h2) lock2_opt.emplace(lock_iface, BodyID{h2->BodyId});
    if (!lock1.Succeeded() || (lock2_opt && !lock2_opt->Succeeded())) return;

    auto &b1 = lock1.GetBody();
    auto &b2 = h2 ? lock2_opt->GetBody() : Body::sFixedToWorld;
    auto *constraint = static_cast<SixDOFConstraint *>(settings.Create(b1, b2));
    ApplyDriveTargets(*constraint, def);
    P->System.AddConstraint(constraint);
    P->Constraints.emplace_back(constraint);

    // Keep all non-static bodies in constraints awake.
    if (!b1.IsStatic()) b1.SetAllowSleeping(false);
    if (!b2.IsStatic()) b2.SetAllowSleeping(false);

    // Disable collision between connected bodies unless explicitly enabled.
    if (!joint.EnableCollision && h2) {
        const auto it1 = P->BodySubGroups.find(body1_entity), it2 = P->BodySubGroups.find(body2_entity);
        if (it1 != P->BodySubGroups.end() && it2 != P->BodySubGroups.end())
            P->FilterRef->DisableCollision(it1->second, it2->second);
    }
}

void PhysicsWorld::FlushJoints(const entt::registry &r, bool joints_changed) {
    if (!joints_changed && !P->JointsDirty) return;
    P->JointsDirty = false;

    for (auto &c : P->Constraints) P->System.RemoveConstraint(c);
    P->Constraints.clear();
    P->FilterRef->ResetDisabledPairs();
    for (auto entity : r.view<const PhysicsJoint>()) BuildJoint(r, entity);
    P->FilterRef->FinalizeDisabledPairs();
}

namespace {
// Walk self+ancestors for nearest PhysicsMotion. null if none.
entt::entity FindMotionOwner(const entt::registry &r, entt::entity e) {
    for (; e != entt::null;) {
        if (r.all_of<PhysicsMotion>(e)) return e;
        const auto p = GetParentEntity(r, e);
        if (p == e) return entt::null;
        e = p;
    }
    return entt::null;
}

// For a child collider, return the body-owning motion ancestor whose compound contains this leaf.
// null if entity owns its own body or has no parent compound body.
entt::entity FindCompoundParentBody(const entt::registry &r, entt::entity e) {
    if (r.all_of<PhysicsBodyHandle>(e)) return entt::null;
    for (auto cur = GetParentEntity(r, e); cur != entt::null;) {
        if (r.all_of<PhysicsMotion, PhysicsBodyHandle>(cur)) return cur;
        const auto p = GetParentEntity(r, cur);
        if (p == cur) return entt::null;
        cur = p;
    }
    return entt::null;
}

// True if any PhysicsJoint's body ancestor resolves to this motion owner.
bool IsJointConstrained(const entt::registry &r, entt::entity motion_owner) {
    for (auto [je, _] : r.view<const PhysicsJoint>().each()) {
        if (FindMotionOwner(r, je) == motion_owner) return true;
    }
    return false;
}

// True iff some static TriangleMesh collider's filter mask permits contact with this filter.
bool CouldHitStaticMesh(const entt::registry &r, std::optional<uint32_t> filter_idx, const KHRCollisionFilter *filter) {
    for (auto [e, cs] : r.view<const ColliderShape>().each()) {
        if (cs.Shape.Type != PhysicsShapeType::TriangleMesh) continue;
        if (FindMotionOwner(r, e) != entt::null) continue; // not static
        const auto *cm = r.try_get<const ColliderMaterial>(e);
        const auto smf = cm ? cm->CollisionFilterIndex : std::nullopt;
        if (!filter_idx || !smf) return true;
        if (!filter || filter->MasksCollide(*filter_idx, *smf)) return true;
    }
    return false;
}

// Promote dynamic TriangleMesh→ConvexHull when all conditions are met:
// Owner is not kinematic. mass > 0. Filters allow contact with some static TriangleMesh. Not joint-constrained.
bool ShouldPromoteMesh(const entt::registry &r, entt::entity motion_owner, const PhysicsMotion &motion, std::optional<uint32_t> filter_idx, const KHRCollisionFilter *filter) {
    if (motion.IsKinematic) return false;
    if (motion.Mass.value_or(0.f) <= 0.f) return false; // mass=0 = locked translation, keep concave mesh
    if (!CouldHitStaticMesh(r, filter_idx, filter)) return false;
    if (IsJointConstrained(r, motion_owner)) return false;
    return true;
}

// Build a leaf Jolt shape for a single collider entity. Promotes mesh→convex per ShouldPromoteMesh,
// stores filter index UserData on the leaf, then wraps in ScaledShape if entity has non-unit scale.
Ref<Shape> BuildLeafShape(const entt::registry &r, entt::entity entity, const ColliderShape &cs, const ColliderMaterial *cm, const PhysicsMotion *owner_motion, entt::entity owner_entity, const KHRCollisionFilter *filter) {
    auto shape_proto = cs.Shape;
    const auto filter_idx = cm ? cm->CollisionFilterIndex : std::nullopt;
    if (shape_proto.Type == PhysicsShapeType::TriangleMesh && owner_motion && ShouldPromoteMesh(r, owner_entity, *owner_motion, filter_idx, filter)) {
        shape_proto.Type = PhysicsShapeType::ConvexHull;
    }
    const auto *mesh = shape_proto.MeshEntity ? r.try_get<const Mesh>(*shape_proto.MeshEntity) : nullptr;
    auto js = CreateJoltShape(shape_proto, mesh);
    if (!js) return {};
    if (filter_idx.has_value()) js->SetUserData(*filter_idx + 1);
    const auto *t = r.try_get<const Transform>(entity);
    if (t && (t->S.x != 1 || t->S.y != 1 || t->S.z != 1)) js = new ScaledShape(js, ToJolt(t->S));
    return js;
}

// Walk descendants of `owner`, stopping at any descendant PhysicsMotion. Collects ColliderShape entities.
void GatherCompoundChildren(const entt::registry &r, entt::entity owner, std::vector<entt::entity> &out) {
    std::function<void(entt::entity)> walk = [&](entt::entity e) {
        for (auto child : Children{&r, e}) {
            if (r.all_of<PhysicsMotion>(child)) continue;
            if (r.all_of<ColliderShape>(child)) out.emplace_back(child);
            walk(child);
        }
    };
    walk(owner);
}

struct BodyShape {
    Ref<Shape> Shape; // null = no body should be created here
    bool IsSensor = false;
    std::optional<PhysicsShapeType> SingleShapeType; // for ApplyPhysicsProperties mesh handling
    const ColliderMaterial *SingleMaterial{nullptr}; // material for body settings (single-collider bodies only)
    std::optional<uint32_t> FilterIdx; // resolved/validated filter index
};

// Build the body's complete shape (single collider, compound, or sensor). Does NOT apply CoM wrap.
// Skips child colliders whose parent motion-owner builds them as compound children.
BodyShape BuildBodyShape(const entt::registry &r, entt::entity entity, const KHRCollisionFilter *filter, size_t filter_count) {
    BodyShape out;
    const auto *motion = r.try_get<const PhysicsMotion>(entity);
    const auto *trigger = r.try_get<const PhysicsTrigger>(entity);
    const auto *collider = r.try_get<const ColliderShape>(entity);

    auto resolve = [&](std::optional<uint32_t> idx) -> std::optional<uint32_t> {
        return idx.has_value() && *idx < filter_count ? idx : std::nullopt;
    };

    if (motion) {
        std::vector<entt::entity> colliders;
        if (collider) colliders.emplace_back(entity);
        GatherCompoundChildren(r, entity, colliders);
        if (!colliders.empty()) {
            if (colliders.size() == 1 && colliders[0] == entity) {
                out.SingleMaterial = r.try_get<const ColliderMaterial>(entity);
                out.SingleShapeType = collider->Shape.Type;
                out.FilterIdx = resolve(out.SingleMaterial ? out.SingleMaterial->CollisionFilterIndex : std::nullopt);
                out.Shape = BuildLeafShape(r, entity, *collider, out.SingleMaterial, motion, entity, filter);
                return out;
            }
            const auto *bt = r.try_get<const Transform>(entity);
            const auto inv_parent = bt ? glm::inverse(glm::translate(mat4{1}, bt->P) * glm::mat4_cast(glm::normalize(bt->R))) : mat4{1};
            StaticCompoundShapeSettings compound;
            for (auto ce : colliders) {
                const auto &cs = r.get<const ColliderShape>(ce);
                const auto *cm = r.try_get<const ColliderMaterial>(ce);
                const auto sub = BuildLeafShape(r, ce, cs, cm, motion, entity, filter);
                if (!sub) continue;
                if (ce == entity) compound.AddShape(Vec3::sZero(), Quat::sIdentity(), sub);
                else {
                    const auto *wt = r.try_get<const WorldTransform>(ce);
                    const auto world = wt ? ToMatrix(*wt) : mat4{1};
                    const auto rel = inv_parent * world;
                    compound.AddShape(ToJolt(vec3{rel[3]}), ToJoltQuat(glm::normalize(glm::quat_cast(glm::mat3{rel}))), sub);
                }
            }
            if (!compound.mSubShapes.empty()) {
                if (const auto result = compound.Create(); result.IsValid()) out.Shape = result.Get();
            }
            return out;
        }
        // Motion entity with no colliders falls through to trigger-sensor path if applicable.
    }

    if (collider && !motion) {
        // Static body — skip if a motion ancestor exists (this leaf is a compound child).
        if (FindMotionOwner(r, GetParentEntity(r, entity)) != entt::null) return out;
        out.SingleMaterial = r.try_get<const ColliderMaterial>(entity);
        out.SingleShapeType = collider->Shape.Type;
        out.FilterIdx = resolve(out.SingleMaterial ? out.SingleMaterial->CollisionFilterIndex : std::nullopt);
        out.Shape = BuildLeafShape(r, entity, *collider, out.SingleMaterial, nullptr, entt::null, filter);
        return out;
    }

    if (trigger && trigger->Shape) {
        out.IsSensor = true;
        out.SingleShapeType = trigger->Shape->Type;
        out.FilterIdx = resolve(trigger->CollisionFilterIndex);
        const auto *mesh = trigger->Shape->MeshEntity ? r.try_get<const Mesh>(*trigger->Shape->MeshEntity) : nullptr;
        auto js = CreateJoltShape(*trigger->Shape, mesh);
        if (!js) return out;
        const auto *t = r.try_get<const Transform>(entity);
        if (t && (t->S.x != 1 || t->S.y != 1 || t->S.z != 1)) js = new ScaledShape(js, ToJolt(t->S));
        out.Shape = js;
    }
    return out;
}

// Wrap with OffsetCenterOfMassShape (KHR semantics: CoM is an absolute local-space point).
// Writes the inner mass props the caller needs to override the BCS with (avoids PAT inflation).
Ref<Shape> WrapCenterOfMass(Ref<Shape> inner, const PhysicsMotion *motion, MassProperties &out_inner_mass_props) {
    if (!motion || !motion->CenterOfMass) return inner;
    out_inner_mass_props = inner->GetMassProperties();
    return new OffsetCenterOfMassShape(inner, ToJolt(*motion->CenterOfMass) - inner->GetCenterOfMass());
}
} // namespace

void PhysicsWorld::AddBody(entt::registry &r, entt::entity entity) {
    if (r.all_of<PhysicsBodyHandle>(entity)) return;

    const auto built = BuildBodyShape(r, entity, P->FilterRef.GetPtr(), Filters.size());
    if (!built.Shape) return;

    const auto *motion = r.try_get<const PhysicsMotion>(entity);
    MassProperties inner_mass_props;
    const auto shape = WrapCenterOfMass(built.Shape, built.IsSensor ? nullptr : motion, inner_mass_props);

    const auto *t = r.try_get<const Transform>(entity);
    const auto pos = t ? RVec3{t->P.x, t->P.y, t->P.z} : RVec3::sZero();
    const auto rot = t ? ToJoltQuat(t->R) : Quat::sIdentity();
    const auto motion_type = motion ? (motion->IsKinematic ? EMotionType::Kinematic : EMotionType::Dynamic) : EMotionType::Static;
    const auto layer = motion ? Layers::Moving : Layers::NonMoving;

    BodyCreationSettings bcs{shape, pos, rot, motion_type, layer};
    bcs.mIsSensor = built.IsSensor;
    ApplyPhysicsProperties(bcs, motion, r.try_get<const PhysicsVelocity>(entity), built.SingleShapeType, built.SingleMaterial, Materials);

    if (motion && motion->CenterOfMass && bcs.mMotionType == EMotionType::Dynamic && !built.IsSensor) {
        inner_mass_props.ScaleToMass(bcs.mMassPropertiesOverride.mMass);
        bcs.mMassPropertiesOverride = inner_mass_props;
        bcs.mOverrideMassProperties = EOverrideMassProperties::MassAndInertiaProvided;
    }

    if (P->FilterRef) {
        const uint32_t sub = P->FilterRef->RegisterBody(built.FilterIdx);
        bcs.mCollisionGroup = CollisionGroup(P->FilterRef, 0, sub);
        P->BodySubGroups[entity] = sub;
    }

    auto &bi = P->System.GetBodyInterface();
    const auto *body = bi.CreateBody(bcs);
    if (!body) return;
    bi.AddBody(body->GetID(), motion_type == EMotionType::Static ? EActivation::DontActivate : EActivation::Activate);
    r.emplace_or_replace<PhysicsBodyHandle>(entity, PhysicsBodyHandle{body->GetID().GetIndexAndSequenceNumber()});

    if (motion && body->IsDynamic() && !built.IsSensor) {
        auto *mp = const_cast<Body *>(body)->GetMotionProperties();
        if (motion->Mass == 0) mp->SetInverseMass(0); // KHR §128: explicit zero = infinite mass
        if (motion->InertiaDiagonal) {
            const auto &d = *motion->InertiaDiagonal;
            const Vec3 inv_diag{d.x > 0 ? 1 / d.x : 0, d.y > 0 ? 1 / d.y : 0, d.z > 0 ? 1 / d.z : 0};
            const auto irot = motion->InertiaOrientation ? ToJoltQuat(*motion->InertiaOrientation) : Quat::sIdentity();
            mp->SetInverseInertia(inv_diag, irot);
        }
    }

    P->JointsDirty = true;
}

void PhysicsWorld::RemoveBody(entt::registry &r, entt::entity entity) {
    const auto *handle = r.try_get<const PhysicsBodyHandle>(entity);
    if (!handle || handle->BodyId == UINT32_MAX) return;
    auto &bi = P->System.GetBodyInterface();
    BodyID id{handle->BodyId};
    bi.RemoveBody(id);
    bi.DestroyBody(id);
    r.remove<PhysicsBodyHandle>(entity);
    P->BodySubGroups.erase(entity);
    P->JointsDirty = true;
}

void PhysicsWorld::ApplyShape(const entt::registry &r, entt::entity entity) {
    const auto *handle = r.try_get<const PhysicsBodyHandle>(entity);
    if (!handle) return;
    const auto built = BuildBodyShape(r, entity, P->FilterRef.GetPtr(), Filters.size());
    if (!built.Shape) return;
    MassProperties inner_mass_props;
    const auto shape = WrapCenterOfMass(built.Shape, built.IsSensor ? nullptr : r.try_get<const PhysicsMotion>(entity), inner_mass_props);
    // updateMassProperties=false: preserves explicit Mass/Inertia overrides and avoids
    // GetMassProperties() on shapes that return zero inertia (TriangleMesh, MeshShape).
    // ApplyMassPropertiesFromShape re-derives mass props with the right guards.
    P->System.GetBodyInterface().SetShape(BodyID{handle->BodyId}, shape, /*updateMassProperties=*/false, EActivation::Activate);
    ApplyMassPropertiesFromShape(r, entity);
}

void PhysicsWorld::ApplyMassPropertiesFromShape(const entt::registry &r, entt::entity entity) {
    const auto *handle = r.try_get<const PhysicsBodyHandle>(entity);
    const auto *motion = r.try_get<const PhysicsMotion>(entity);
    if (!handle || !motion) return;
    if (motion->InertiaDiagonal) return; // explicit override wins
    if (motion->Mass == 0.0f) return; // KHR §128 infinite mass — handled separately

    BodyLockWrite lock(P->System.GetBodyLockInterface(), BodyID{handle->BodyId});
    if (!lock.Succeeded()) return;
    auto &body = lock.GetBody();
    if (!body.IsDynamic()) return; // sensors, static, kinematic skip

    const float mass = motion->Mass.value_or(DefaultMass);
    auto props = body.GetShape()->GetMassProperties();
    props.ScaleToMass(mass > 0 ? mass : DefaultMass);
    body.GetMotionProperties()->SetMassProperties(EAllowedDOFs::All, props);
}

void PhysicsWorld::ApplyMotion(const entt::registry &r, entt::entity entity) {
    const auto *handle = r.try_get<const PhysicsBodyHandle>(entity);
    const auto *motion = r.try_get<const PhysicsMotion>(entity);
    if (!handle || !motion) return;

    auto &bi = P->System.GetBodyInterface();
    const BodyID id{handle->BodyId};

    // Kinematic toggle is a motion-type flip (Dynamic ↔ Kinematic, both stay in Layers::Moving).
    const auto desired_type = motion->IsKinematic ? EMotionType::Kinematic : EMotionType::Dynamic;
    if (bi.GetMotionType(id) != desired_type) bi.SetMotionType(id, desired_type, EActivation::Activate);

    bi.SetGravityFactor(id, motion->GravityFactor);

    {
        BodyLockWrite lock(P->System.GetBodyLockInterface(), id);
        if (!lock.Succeeded()) return;
        auto &body = lock.GetBody();
        if (!body.IsDynamic() && !body.IsKinematic()) return;

        auto *mp = body.GetMotionProperties();
        mp->SetLinearDamping(motion->LinearDamping);
        mp->SetAngularDamping(motion->AngularDamping);

        if (body.IsDynamic()) {
            if (motion->Mass == 0.0f) {
                mp->SetInverseMass(0); // KHR §128: explicit zero = infinite mass (locked translation)
            } else {
                const float mass = motion->Mass.value_or(DefaultMass);
                mp->SetInverseMass(mass > 0 ? 1.f / mass : 1.f / DefaultMass);
            }
            if (motion->InertiaDiagonal) {
                const auto &d = *motion->InertiaDiagonal;
                const Vec3 inv_diag{d.x > 0 ? 1 / d.x : 0, d.y > 0 ? 1 / d.y : 0, d.z > 0 ? 1 / d.z : 0};
                const auto irot = motion->InertiaOrientation ? ToJoltQuat(*motion->InertiaOrientation) : Quat::sIdentity();
                mp->SetInverseInertia(inv_diag, irot);
            }
        }
    }

    // CenterOfMass (set or cleared) requires SetShape with a refreshed OffsetCenterOfMassShape
    // wrapper; ApplyShape also re-derives mass props from the new shape for the no-override case.
    ApplyShape(r, entity);
}

void PhysicsWorld::ApplyMaterial(const entt::registry &r, entt::entity entity) {
    const auto *material = r.try_get<const ColliderMaterial>(entity);
    if (!material) return;

    if (const auto owner = FindCompoundParentBody(r, entity); owner != entt::null) {
        // Child collider: leaf UserData lives inside the parent compound shape — rebuild it.
        ApplyShape(r, owner);
        return;
    }
    const auto *handle = r.try_get<const PhysicsBodyHandle>(entity);
    if (!handle) return;

    auto &bi = P->System.GetBodyInterface();
    const BodyID id{handle->BodyId};

    if (material->PhysicsMaterialIndex.has_value() && *material->PhysicsMaterialIndex < Materials.size()) {
        const auto &mat = Materials[*material->PhysicsMaterialIndex];
        bi.SetFriction(id, mat.DynamicFriction);
        bi.SetRestitution(id, mat.Restitution);
    } else {
        const ::PhysicsMaterial defaults;
        bi.SetFriction(id, defaults.DynamicFriction);
        bi.SetRestitution(id, defaults.Restitution);
    }

    if (P->FilterRef) {
        if (const auto it = P->BodySubGroups.find(entity); it != P->BodySubGroups.end()) {
            const auto filter_idx = material->CollisionFilterIndex.has_value() && *material->CollisionFilterIndex < Filters.size() ? material->CollisionFilterIndex : std::nullopt;
            P->FilterRef->SetBodyFilter(it->second, filter_idx);
        }
    }

    BodyLockWrite lock(P->System.GetBodyLockInterface(), id);
    if (lock.Succeeded()) {
        auto *leaf = const_cast<Shape *>(lock.GetBody().GetShape());
        while (leaf->GetType() == EShapeType::Decorated) {
            leaf = const_cast<Shape *>(static_cast<const DecoratedShape *>(leaf)->GetInnerShape());
        }
        leaf->SetUserData(material->CollisionFilterIndex.value_or(UINT32_MAX) + 1);
    }
}

void PhysicsWorld::OnShapeChange(entt::registry &r, entt::entity e) {
    if (!r.valid(e)) return;
    const bool has_shape = r.all_of<ColliderShape>(e);
    const bool has_body = r.all_of<PhysicsBodyHandle>(e);
    const auto compound_owner = FindCompoundParentBody(r, e);
    if (!has_shape) {
        if (has_body) RemoveBody(r, e);
        else if (compound_owner != entt::null) ApplyShape(r, compound_owner);
        return;
    }
    if (compound_owner != entt::null) ApplyShape(r, compound_owner);
    else if (has_body) ApplyShape(r, e);
    else AddBody(r, e);
}

void PhysicsWorld::OnMotionChange(entt::registry &r, entt::entity e) {
    if (!r.valid(e)) return;
    const bool has_motion = r.all_of<PhysicsMotion>(e);
    const bool has_body = r.all_of<PhysicsBodyHandle>(e);
    if (!has_motion) {
        if (has_body) RemoveBody(r, e);
        if (r.all_of<ColliderShape>(e)) AddBody(r, e); // demote: rebuild as static
        return;
    }
    // Object layer is fixed at body creation. Static lives in NonMoving; Dynamic/Kinematic in Moving.
    // Crossing that boundary needs a full recreate; Dynamic↔Kinematic within Moving can cheap-apply.
    if (has_body) {
        const auto *handle = r.try_get<const PhysicsBodyHandle>(e);
        const bool is_static = P->System.GetBodyInterface().GetMotionType(BodyID{handle->BodyId}) == EMotionType::Static;
        if (is_static) {
            RemoveBody(r, e);
            AddBody(r, e);
        } else {
            ApplyMotion(r, e);
        }
    } else {
        AddBody(r, e); // motion appeared on an entity that didn't have a body yet
    }
}

void PhysicsWorld::OnMaterialChange(entt::registry &r, entt::entity e) {
    if (!r.valid(e) || !r.all_of<ColliderMaterial>(e)) return;
    ApplyMaterial(r, e);
}

void PhysicsWorld::OnTriggerChange(entt::registry &r, entt::entity e) {
    if (!r.valid(e)) return;
    const bool has_trigger = r.all_of<PhysicsTrigger>(e);
    const bool has_body = r.all_of<PhysicsBodyHandle>(e);
    if (!has_trigger) {
        // Trigger removed: only drop the body if it was sensor-only (no collider/motion).
        if (has_body && !r.any_of<ColliderShape, PhysicsMotion>(e)) RemoveBody(r, e);
        return;
    }
    if (has_body) ApplyShape(r, e);
    else AddBody(r, e);
}

void PhysicsWorld::Step(entt::registry &r, float dt) {
    P->System.SetGravity(ToJolt(Gravity));
    P->System.Update(dt, SubSteps, &P->TempAllocator, &P->JobSystem);

    // Sync Jolt → ECS for dynamic and kinematic bodies.
    // Kinematic bodies move according to their velocity in Jolt and must be read back.
    // PhysicsVelocity is mutated by direct assignment (no patch) so we don't trigger reactive updates each frame.
    const auto &bi = P->System.GetBodyInterface();
    for (auto [entity, velocity, handle] : r.view<PhysicsVelocity, const PhysicsBodyHandle>().each()) {
        const BodyID id{handle.BodyId};
        if (!bi.IsActive(id)) continue;

        r.patch<Transform>(entity, [&](Transform &t) {
            t.P = FromJoltRVec3(bi.GetPosition(id));
            t.R = FromJoltQuat(bi.GetRotation(id));
        });
        velocity.Linear = FromJoltVec3(bi.GetLinearVelocity(id));
        velocity.Angular = FromJoltVec3(bi.GetAngularVelocity(id));
    }
}

void PhysicsWorld::SaveSnapshot(entt::registry &r) {
    P->Snapshots.clear();
    const auto &bi = P->System.GetBodyInterface();
    for (auto [entity, _, handle] : r.view<const PhysicsVelocity, const PhysicsBodyHandle>().each()) {
        const BodyID id{handle.BodyId};
        const auto *t = r.try_get<const Transform>(entity);
        P->Snapshots.emplace_back(BodySnapshot{
            entity,
            t ? t->P : vec3{0},
            t ? t->R : quat{1, 0, 0, 0},
            t ? t->S : vec3{1},
            {FromJoltVec3(bi.GetLinearVelocity(id)), FromJoltVec3(bi.GetAngularVelocity(id))},
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
        if (auto *vel = r.try_get<PhysicsVelocity>(snap.Entity)) *vel = snap.Velocity;
    }
    // Rebuild all Jolt bodies and constraints from the restored ECS state.
    // This guarantees deterministic replay by eliminating stale solver warm-start cache.
    // Ideally we'd just clear the solver cache, but Jolt doesn't expose such an API.
    Rebuild(r);
}
