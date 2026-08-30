// All Jolt includes are isolated to this file.
#include "PhysicsSystem.h"
#include "PhysicsChanges.h"
#include "PhysicsContact.h"
#include "PhysicsTypes.h"
#include "Reactive.h"
#include "TransformMath.h"
#include "Variant.h"
#include "mesh/Mesh.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "scene/WorldTransform.h"

#include "Jolt/Jolt.h"

#include "Jolt/Core/Factory.h"
#include "Jolt/Core/JobSystemThreadPool.h"
#include "Jolt/Physics/Body/BodyCreationSettings.h"
#include "Jolt/Physics/Collision/CollideShape.h"
#include "Jolt/Physics/Collision/Shape/BoxShape.h"
#include "Jolt/Physics/Collision/Shape/CapsuleShape.h"
#include "Jolt/Physics/Collision/Shape/ConvexHullShape.h"
#include "Jolt/Physics/Collision/Shape/CylinderShape.h"
#include "Jolt/Physics/Collision/Shape/EmptyShape.h"
#include "Jolt/Physics/Collision/Shape/MeshShape.h"
#include "Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h"
#include "Jolt/Physics/Collision/Shape/PlaneShape.h"
#include "Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h"
#include "Jolt/Physics/Collision/Shape/ScaledShape.h"
#include "Jolt/Physics/Collision/Shape/SphereShape.h"
#include "Jolt/Physics/Collision/Shape/StaticCompoundShape.h"
#include "Jolt/Physics/Collision/Shape/TaperedCapsuleShape.h"
#include "Jolt/Physics/Collision/Shape/TaperedCylinderShape.h"
#include "Jolt/Physics/Collision/SimShapeFilter.h"
#include "Jolt/Physics/Constraints/HingeConstraint.h"
#include "Jolt/Physics/Constraints/SixDOFConstraint.h"
#include "Jolt/Physics/PhysicsSystem.h"
#include "Jolt/RegisterTypes.h"

#include <algorithm>
#include <iostream>
#include <mutex>
#include <ranges>

JPH_SUPPRESS_WARNINGS

using namespace JPH;
using namespace JPH::literals;

namespace {
inline Vec3 ToJolt(vec3 v) { return {v.x, v.y, v.z}; }
inline vec3 FromJolt(RVec3 v) { return {v.GetX(), v.GetY(), v.GetZ()}; }
inline Quat ToJolt(quat q) { return {q.x, q.y, q.z, q.w}; }
inline quat FromJolt(Quat q) { return {q.GetW(), q.GetX(), q.GetY(), q.GetZ()}; }

inline bool HasNonUnitScale(const Transform &t) { return t.S.x != 1 || t.S.y != 1 || t.S.z != 1; }

// A Shape's UserData packs the collider node it was built for in the high half and that collider's collision filter in the low half.
// Each is offset by 1 so 0 means none, and both are read back per sub-shape.
inline uint64_t ShapeUserData(entt::entity collider, entt::entity filter) {
    const auto pack = [](entt::entity e) { return e != null_entity ? uint64_t(uint32_t(e)) + 1 : 0; };
    return (pack(collider) << 32) | pack(filter);
}
inline entt::entity UnpackEntity(uint32_t v) { return v != 0 ? entt::entity(v - 1) : null_entity; }
inline entt::entity ShapeFilterEntity(uint64_t data) { return UnpackEntity(uint32_t(data)); }
inline entt::entity ShapeColliderEntity(uint64_t data) { return UnpackEntity(uint32_t(data >> 32)); }

inline entt::entity ResolveFilterEntity(const entt::registry &r, entt::entity e) {
    return e != null_entity && r.valid(e) && r.all_of<CollisionFilter>(e) ? e : null_entity;
}

inline float ResolvedMass(const PhysicsMotion &m) {
    const float v = m.Mass.value_or(DefaultMass);
    return v > 0 ? v : DefaultMass;
}

inline void ApplyInertiaDiagonal(MotionProperties &mp, const PhysicsMotion &motion) {
    if (!motion.InertiaDiagonal) return;
    const auto &d = *motion.InertiaDiagonal;
    const Vec3 inv_diag{d.x > 0 ? 1 / d.x : 0, d.y > 0 ? 1 / d.y : 0, d.z > 0 ? 1 / d.z : 0};
    const auto irot = motion.InertiaOrientation ? ToJolt(*motion.InertiaOrientation) : Quat::sIdentity();
    mp.SetInverseInertia(inv_diag, irot);
}

namespace Layers {
constexpr ObjectLayer NonMoving{0}, Moving{1}, NumLayers{2};
} // namespace Layers

namespace BPLayers {
constexpr BroadPhaseLayer NonMoving{0}, Moving{1};
constexpr uint32_t NumLayers{2};
} // namespace BPLayers

class BPLayerInterface final : public BroadPhaseLayerInterface {
public:
    BPLayerInterface() {
        Mapping[Layers::NonMoving] = BPLayers::NonMoving;
        Mapping[Layers::Moving] = BPLayers::Moving;
    }
    uint32_t GetNumBroadPhaseLayers() const override { return BPLayers::NumLayers; }
    BroadPhaseLayer GetBroadPhaseLayer(ObjectLayer layer) const override { return Mapping[layer]; }
    const char *GetBroadPhaseLayerName(BroadPhaseLayer layer) const override {
        if (layer == BPLayers::NonMoving) return "NON_MOVING";
        if (layer == BPLayers::Moving) return "MOVING";
        return "INVALID";
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

// Filter storage is keyed by the CollisionFilter entity's raw value (uint32_t),
// stored directly in Body/Shape UserData. UINT32_MAX (== null_entity) means "no filter".
class KHRCollisionFilter : public GroupFilter {
    std::unordered_map<uint32_t, CollisionMask> Masks; // filter entity value → mask
    // Each body gets a unique SubGroupID via RegisterBody(). Maps SubGroupID → filter entity value.
    std::vector<uint32_t> BodyFilterByGroup; // UINT32_MAX = no KHR filter
    // Pairs of SubGroupIDs that should not collide (from joints with EnableCollision=false).
    std::vector<std::pair<uint32_t, uint32_t>> DisabledPairs; // sorted after FinalizeDisabledPairs()

public:
    KHRCollisionFilter() = default;

    void Update(const entt::registry &r) {
        // Assign a bit index to each CollisionSystem entity by iteration order.
        std::unordered_map<entt::entity, uint32_t> bit_index;
        {
            uint32_t i = 0;
            for (auto e : r.view<const CollisionSystem>()) bit_index.emplace(e, i++);
        }
        auto systems_to_mask = [&](const std::vector<entt::entity> &systems) {
            uint32_t m = 0;
            for (auto e : systems) {
                if (auto it = bit_index.find(e); it != bit_index.end()) m |= 1u << it->second;
            }
            return m;
        };

        Masks.clear();
        for (auto [e, f] : r.view<const CollisionFilter>().each()) {
            CollisionMask mask;
            mask.Membership = systems_to_mask(f.Systems);
            switch (f.Mode) {
                case CollideMode::Allowlist: mask.Collide = systems_to_mask(f.CollideSystems); break;
                case CollideMode::Blocklist: mask.Collide = ~systems_to_mask(f.CollideSystems); break;
                case CollideMode::All: mask.Collide = ~0u; break;
            }
            Masks.emplace(uint32_t(e), mask);
        }
    }

    // Register a body and return its unique SubGroupID. filter is the KHR collision filter entity.
    uint32_t RegisterBody(entt::entity filter = null_entity) {
        const uint32_t id = BodyFilterByGroup.size();
        BodyFilterByGroup.emplace_back(uint32_t(filter));
        return id;
    }

    void SetBodyFilter(uint32_t sub_group_id, entt::entity filter) {
        if (sub_group_id < BodyFilterByGroup.size()) BodyFilterByGroup[sub_group_id] = uint32_t(filter);
    }

    void Reset() {
        BodyFilterByGroup.clear();
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
        if (!DisabledPairs.empty()) {
            if (const auto key = std::make_pair(std::min(s1, s2), std::max(s1, s2));
                std::binary_search(DisabledPairs.begin(), DisabledPairs.end(), key)) return false;
        }
        const uint32_t f1 = s1 < BodyFilterByGroup.size() ? BodyFilterByGroup[s1] : UINT32_MAX;
        const uint32_t f2 = s2 < BodyFilterByGroup.size() ? BodyFilterByGroup[s2] : UINT32_MAX;
        return MasksCollide(entt::entity(f1), entt::entity(f2));
    }

    bool MasksCollide(entt::entity a, entt::entity b) const {
        const auto ia = Masks.find(uint32_t(a));
        const auto ib = Masks.find(uint32_t(b));
        if (ia == Masks.end() || ib == Masks.end()) return true; // missing/null filter = permissive
        return (ia->second.Membership & ib->second.Collide) != 0 && (ib->second.Membership & ia->second.Collide) != 0;
    }

    bool DirectionalAllows(entt::entity source, entt::entity target) const {
        const auto is = Masks.find(uint32_t(source));
        const auto it = Masks.find(uint32_t(target));
        if (is == Masks.end() || it == Masks.end()) return true;
        return (is->second.Membership & it->second.Collide) != 0;
    }
};

// Body::UserData stores the PhysicsMaterial entity's raw value (uint32_t, widened to uint64_t).
// UINT32_MAX (== null_entity) means "no material assigned".
constexpr uint64_t NoMaterialSentinel{UINT32_MAX};

float ApplyCombineMode(PhysicsCombineMode mode, float a, float b) {
    switch (mode) {
        case PhysicsCombineMode::Average: return (a + b) * 0.5f;
        case PhysicsCombineMode::Minimum: return std::min(a, b);
        case PhysicsCombineMode::Maximum: return std::max(a, b);
        case PhysicsCombineMode::Multiply: return a * b;
    }
    return (a + b) * 0.5f;
}

// The tangential impulse the solver applied, resolved along the tangent basis its own cached normal implies.
// Friction lambdas act along that basis, so reading them needs the normal they were solved against, cached in body 2's frame.
// A cache entry holding no normal keeps `fallback_normal`.
struct SolverFriction {
    Vec3 Normal, Impulse;
};
SolverFriction ResolveSolverFriction(const ContactConstraintManager::AppliedContactImpulses &applied, QuatArg body2_rotation, Vec3Arg fallback_normal) {
    const Vec3 cached = body2_rotation * applied.mNormal;
    const Vec3 normal = cached.LengthSq() > 0.5f ? cached.Normalized() : Vec3{fallback_normal};
    const Vec3 tangent1 = normal.GetNormalizedPerpendicular();
    return {normal, tangent1 * applied.mFrictionImpulse1 + normal.Cross(tangent1) * applied.mFrictionImpulse2};
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
// 3. Detailed reporting of new solid contacts and persisting ones (see RawContact, RawSustained).
class KHRContactListener : public ContactListener {
public:
    const entt::registry *R{nullptr};
    const KHRCollisionFilter *Filter{nullptr};
    const JPH::PhysicsSystem *System{nullptr}; // For reading back the impulses the solver applied.
    // Whether each body reports its contacts in detail, by BodyID index.
    // Rebuilt from the ReportContacts tag before each step.
    std::vector<uint8_t> Reporting;

    // One qualifying new contact. Bodies are stored by index; the drain resolves them to entities and
    // splits the impulse into a per-body impact. Filled from multiple threads during PhysicsSystem::Update.
    struct RawContact {
        uint32_t Body1Index, Body2Index;
        entt::entity Collider1, Collider2; // The touching collider node of each body.
        Vec3 Point, Direction; // world space; Direction is the unit impulse on body 2 (body 1 gets its negation)
        Vec3 ResultantPoint; // world space, the manifold's load-weighted centre, shared by all its points
        float Impulse, Speed;
        float InvMass1, InvMass2; // inverse masses, kg⁻¹, 0 for static/kinematic bodies
        float NominalArea; // area of the manifold's contact polygon, m^2
    };
    // A new manifold awaiting the solver's receipt.
    // OnContactAdded records it, and the first step boundary whose applied normal impulse is nonzero turns it into RawContacts, or OnContactRemoved retires it.
    // OnContactRemoved runs with no body access, so everything geometric is captured here at add time.
    struct PendingImpact {
        uint32_t Body1Index, Body2Index;
        entt::entity Collider1, Collider2; // The touching collider node of each body.
        RMat44 COMTransform1; // Body 1's centre-of-mass transform, mapping cached contact points to world.
        Quat Rotation2; // Body 2's rotation, mapping the cached contact normal to world.
        Vec3 Normal; // world space, unit, into body 2.
        // The load this manifold carries once settled, N: the weight component each dynamic body presses along the normal.
        float SupportForce;
        float InvMass1, InvMass2; // inverse masses, kg⁻¹, 0 for static/kinematic bodies
        float Restitution; // Combined restitution of the pair.
        float NominalArea; // area of the manifold's contact polygon, m^2
    };
    struct SubShapeIDPairHash {
        size_t operator()(const SubShapeIDPair &pair) const { return size_t(pair.GetHash()); }
    };
    // Guarded by ContactMutex. Persists across steps: a manifold added in one frame's last substep resolves in the next.
    std::unordered_map<SubShapeIDPair, PendingImpact, SubShapeIDPairHash> PendingImpacts;
    // Mirrors PendingImpacts.size() so the per-substep persisted callbacks skip the lock while nothing is pending.
    // A manifold added in the current substep cannot also persist in it, so a stale zero never skips a strike.
    std::atomic<uint32_t> PendingImpactCount{0};
    float SubstepDt{1.f / 600}; // Simulated seconds per collision step, the span one applied impulse covers.

    // One persisting contact, per sub-shape pair, which the drain merges into one contact per manifold.
    struct RawSustained {
        uint32_t Body1Index, Body2Index;
        uint64_t SubShapeKey; // The touching sub-shapes, identifying the manifold within its body pair.
        entt::entity Collider1, Collider2; // The touching collider node of each body.
        Vec3 Point; // world space
        Vec3 Normal; // world space unit normal on body 2
        Vec3 Slip; // world-space relative tangential velocity of body 1's material point
        Vec3 Local1, Local2; // contact point in each body's own frame, for differencing into a sweep velocity
        Vec3 FrictionImpulse; // world space, the solver's tangential impulse on body 1
        float NormalImpulse, Restitution, Friction;
        float NominalArea; // area of the manifold's contact polygon, m^2
        float NominalExtent; // extent of the manifold's contact polygon along the slide, m
    };
    std::mutex ContactMutex;
    std::vector<RawContact> RawContacts;
    std::vector<RawSustained> RawSustainedContacts;

    ValidateResult OnContactValidate(const Body &b1, const Body &b2, RVec3Arg, const CollideShapeResult &result) override {
        if (!Filter) return ValidateResult::AcceptAllContactsForThisBodyPair;
        const auto e1 = ShapeFilterEntity(b1.GetShape()->GetSubShapeUserData(result.mSubShapeID1));
        const auto e2 = ShapeFilterEntity(b2.GetShape()->GetSubShapeUserData(result.mSubShapeID2));
        if (e1 == null_entity || e2 == null_entity) return ValidateResult::AcceptContact;
        return Filter->MasksCollide(e1, e2) ? ValidateResult::AcceptContact : ValidateResult::RejectContact;
    }

    void OnContactAdded(const Body &b1, const Body &b2, const ContactManifold &manifold, ContactSettings &s) override {
        CombineMaterials(b1, b2, manifold, s);
        RecordPendingImpact(b1, b2, manifold, s);
    }
    void OnContactPersisted(const Body &b1, const Body &b2, const ContactManifold &manifold, ContactSettings &s) override {
        CombineMaterials(b1, b2, manifold, s);
        // The previous collision step's applied impulses are now cached, so a manifold that appeared then strikes here.
        ConsumePendingImpact(SubShapeIDPair{b1.GetID(), manifold.mSubShapeID1, b2.GetID(), manifold.mSubShapeID2}, false);
        CollectSustained(b1, b2, manifold, s);
    }
    // Fires during cache finalization with no body access, after the removed manifold's last solved step.
    // A manifold that appeared and was removed between two boundaries reports its whole touch and bounce here.
    void OnContactRemoved(const SubShapeIDPair &pair) override {
        ConsumePendingImpact(pair, true);
    }

private:
    bool Reports(const Body &b) const {
        const auto index = b.GetID().GetIndex();
        return index < Reporting.size() && Reporting[index] != 0;
    }
    // A pair is worth solving for detail when at least one of its bodies reports contacts.
    bool PairReports(const Body &b1, const Body &b2) const { return Reports(b1) || Reports(b2); }

    // The collider node a sub-shape was built for, which a compound body has one of per sub-shape.
    static entt::entity ColliderOf(const Body &b, const SubShapeID &sub) { return ShapeColliderEntity(b.GetShape()->GetSubShapeUserData(sub)); }

    // Reports where a persisting contact sits on each body, how hard it presses, and how fast the surfaces slide.
    // Fires once per collision step, so the drain sums the impulses over the frame's simulated time.
    // Area of the manifold's contact polygon, m^2.
    // Jolt clips the touching faces into an ordered convex polygon, so fanning it from its first point gives the area two faces share.
    // A point or an edge spans none, which is what separates a load-set patch from a geometry-set one.
    static float ManifoldArea(const ContactManifold &manifold) {
        const auto &points = manifold.mRelativeContactPointsOn1;
        if (points.size() < 3) return 0;
        Vec3 twice_area = Vec3::sZero();
        for (uint i = 1; i + 1 < points.size(); ++i) {
            twice_area += (points[i] - points[0]).Cross(points[i + 1] - points[0]);
        }
        return 0.5f * std::abs(twice_area.Dot(manifold.mWorldSpaceNormal));
    }

    // Extent of the manifold's contact polygon along the slide, m.
    // The polygon's points carry the solver's load across their whole spread, so the spread measures the region confining the contact even where the polygon's area collapses to a line.
    // A contact that is not sliding has no track direction, so it takes the widest spread the points have.
    static float ManifoldExtent(const ContactManifold &manifold, Vec3Arg slip) {
        const auto &points = manifold.mRelativeContactPointsOn1;
        const float slip_len = slip.Length();
        if (slip_len > 1e-6f) {
            const Vec3 dir = slip / slip_len;
            float lo = std::numeric_limits<float>::max(), hi = std::numeric_limits<float>::lowest();
            for (const auto &p : points) {
                const float along = p.Dot(dir);
                lo = std::min(lo, along);
                hi = std::max(hi, along);
            }
            return hi - lo;
        }
        float span = 0;
        for (uint i = 0; i < points.size(); ++i) {
            for (uint j = i + 1; j < points.size(); ++j) span = std::max(span, (points[i] - points[j]).Length());
        }
        return span;
    }

    void CollectSustained(const Body &b1, const Body &b2, const ContactManifold &manifold, const ContactSettings &s) {
        if (!PairReports(b1, b2) || b1.IsSensor() || b2.IsSensor() || manifold.mRelativeContactPointsOn1.empty()) return;

        // The normal (Hertz) impulses the solver applied, which carry the load every other contact presses into this pair.
        // They are the previous collision step's, since this fires before the one that is about to be solved.
        ContactConstraintManager::AppliedContactImpulses applied;
        if (!System->GetAppliedContactImpulses(SubShapeIDPair{b1.GetID(), manifold.mSubShapeID1, b2.GetID(), manifold.mSubShapeID2}, applied)) return;

        // The load weighted centre of the manifold's points, where one resultant reproduces both the force and the moment.
        // Each point is cached in its own body's centre of mass space, the frame the sweep reference is differenced in.
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

        const RVec3 world_point = b1.GetCenterOfMassTransform() * local1;
        const Vec3 relative_velocity = b1.GetPointVelocity(world_point) - b2.GetPointVelocity(world_point);
        const Vec3 normal = manifold.mWorldSpaceNormal;
        const Vec3 slip = relative_velocity - normal * relative_velocity.Dot(normal);
        // The tangential force the solver applied, kinetic on a slide and whatever holds a roll.
        const Vec3 friction_impulse = ResolveSolverFriction(applied, b2.GetRotation(), normal).Impulse;

        const uint64_t sub_shape_key = (uint64_t(manifold.mSubShapeID1.GetValue()) << 32) | manifold.mSubShapeID2.GetValue();
        const std::scoped_lock lock{ContactMutex};
        RawSustainedContacts.emplace_back(
            b1.GetID().GetIndex(), b2.GetID().GetIndex(), sub_shape_key,
            ColliderOf(b1, manifold.mSubShapeID1), ColliderOf(b2, manifold.mSubShapeID2),
            Vec3{world_point}, normal, slip, local1, local2, friction_impulse, normal_impulse, s.mCombinedRestitution, s.mCombinedFriction,
            ManifoldArea(manifold), ManifoldExtent(manifold, slip)
        );
    }

    // Records a new manifold whose first solved boundary will report the strike.
    // Nothing is read mid-step: the impulse comes from the solver's cache one substep later.
    void RecordPendingImpact(const Body &b1, const Body &b2, const ContactManifold &manifold, const ContactSettings &s) {
        if (!PairReports(b1, b2) || b1.IsSensor() || b2.IsSensor() || manifold.mRelativeContactPointsOn1.empty()) return;

        const Vec3 normal = manifold.mWorldSpaceNormal;
        const Vec3 gravity = System->GetGravity();
        const float inv_mass1 = b1.IsDynamic() ? b1.GetMotionProperties()->GetInverseMass() : 0.f;
        const float inv_mass2 = b2.IsDynamic() ? b2.GetMotionProperties()->GetInverseMass() : 0.f;
        // The weight component each dynamic body presses along the normal, the load the manifold carries once settled.
        // A body settling onto several manifolds at once presses its weight into each estimate, which biases toward silence.
        float support = 0;
        if (inv_mass2 > 0) support += std::max(0.f, -gravity.Dot(normal)) / inv_mass2;
        if (inv_mass1 > 0) support += std::max(0.f, gravity.Dot(normal)) / inv_mass1;

        const std::scoped_lock lock{ContactMutex};
        PendingImpacts.insert_or_assign(
            SubShapeIDPair{b1.GetID(), manifold.mSubShapeID1, b2.GetID(), manifold.mSubShapeID2},
            PendingImpact{
                b1.GetID().GetIndex(), b2.GetID().GetIndex(),
                ColliderOf(b1, manifold.mSubShapeID1), ColliderOf(b2, manifold.mSubShapeID2),
                b1.GetCenterOfMassTransform(), b2.GetRotation(),
                normal, support, inv_mass1, inv_mass2, s.mCombinedRestitution, ManifoldArea(manifold)
            }
        );
        PendingImpactCount.store(uint32_t(PendingImpacts.size()), std::memory_order_relaxed);
    }

    // Removes a pending entry, keeping the lock-free count in step. ContactMutex must be held.
    void ErasePendingImpact(decltype(PendingImpacts)::iterator it) {
        PendingImpacts.erase(it);
        PendingImpactCount.store(uint32_t(PendingImpacts.size()), std::memory_order_relaxed);
    }

    // Reports a pending manifold's strike from the impulses the solver applied, one report per cached contact point.
    // The settled share of the boundary impulse is support, and the strike is the transient excess over it, so a resting contact's first impulse renders silence.
    // `final` marks the manifold's removal, until which a pending contact the solver has not loaded stays pending, a speculative manifold being reported one boundary before it presses.
    void ConsumePendingImpact(const SubShapeIDPair &pair, bool final) {
        if (PendingImpactCount.load(std::memory_order_relaxed) == 0) return;
        const std::scoped_lock lock{ContactMutex};
        const auto it = PendingImpacts.find(pair);
        if (it == PendingImpacts.end()) return;

        ContactConstraintManager::AppliedContactImpulses applied;
        float normal_impulse = 0;
        if (System->GetAppliedContactImpulses(pair, applied)) {
            for (const auto &p : applied.mPoints) normal_impulse += p.mNormalImpulse;
        }
        if (normal_impulse <= 0) {
            if (final) ErasePendingImpact(it);
            return;
        }
        const PendingImpact pi = it->second;
        ErasePendingImpact(it);

        const float excess = normal_impulse - pi.SupportForce * SubstepDt;
        if (excess <= 1e-6f) return; // pure support: the body settled rather than struck
        const float excess_scale = excess / normal_impulse;

        // The strike substep's normal and tangential impulse, against the add-time normal as fallback.
        // The strike keeps the excess fraction of the friction, matching its share of the normal impulse.
        const auto solved = ResolveSolverFriction(applied, pi.Rotation2, pi.Normal);
        const Vec3 normal = solved.Normal;
        const Vec3 friction = solved.Impulse * excess_scale;
        // The approach speed the strike arrested, v = J/(m_eff*(1+e)), exact for a point contact and a rolloff corner for a manifold.
        // This drives the Hertz contact time.
        const float speed = excess * (pi.InvMass1 + pi.InvMass2) / (1 + pi.Restitution);

        // The whole manifold lands as one collision, so every point of it carries the polygon the faces share and the load-weighted centre the collision's resultant acts through.
        Vec3 resultant = Vec3::sZero();
        for (const auto &p : applied.mPoints) resultant += Vec3{pi.COMTransform1 * p.mPosition1} * p.mNormalImpulse;
        resultant /= normal_impulse;

        for (const auto &p : applied.mPoints) {
            const float share = p.mNormalImpulse * excess_scale;
            const Vec3 j = share * normal + friction * (p.mNormalImpulse / normal_impulse);
            const float impulse = j.Length();
            if (impulse < 1e-6f) continue; // a point of the manifold carrying no load
            RawContacts.emplace_back(
                pi.Body1Index, pi.Body2Index, pi.Collider1, pi.Collider2,
                Vec3{pi.COMTransform1 * p.mPosition1}, j / impulse, resultant, impulse, speed, pi.InvMass1, pi.InvMass2, pi.NominalArea
            );
        }
    }

    const ::PhysicsMaterial *LookupMaterial(uint64_t userdata) const {
        if (!R || userdata == NoMaterialSentinel) return nullptr;
        const auto e = entt::entity(uint32_t(userdata));
        return R->valid(e) ? R->try_get<const ::PhysicsMaterial>(e) : nullptr;
    }

    // The material of the sub-shape actually touching, falling back to the body's when that sub-shape names none.
    const ::PhysicsMaterial *MaterialOf(const Body &b, const SubShapeID &sub) const {
        if (const auto collider = ColliderOf(b, sub); R && collider != null_entity && R->valid(collider)) {
            if (const auto *cm = R->try_get<const ColliderMaterial>(collider); cm && cm->PhysicsMaterialEntity != null_entity && R->valid(cm->PhysicsMaterialEntity)) {
                if (const auto *m = R->try_get<const ::PhysicsMaterial>(cm->PhysicsMaterialEntity)) return m;
            }
        }
        return LookupMaterial(b.GetUserData());
    }

    void CombineMaterials(const Body &b1, const Body &b2, const ContactManifold &manifold, ContactSettings &s) const {
        const auto *m1 = MaterialOf(b1, manifold.mSubShapeID1);
        const auto *m2 = MaterialOf(b2, manifold.mSubShapeID2);
        if (!m1 && !m2) return; // both use Jolt defaults
        // Default combine mode is Average per KHR spec.
        const auto fc = [](const ::PhysicsMaterial *m) { return m ? m->FrictionCombine : PhysicsCombineMode::Average; };
        const auto rc = [](const ::PhysicsMaterial *m) { return m ? m->RestitutionCombine : PhysicsCombineMode::Average; };
        // KHR combine mode priority: Maximum > Multiply > Average > Minimum.
        const auto priority = [](PhysicsCombineMode m) {
            switch (m) {
                case PhysicsCombineMode::Maximum: return 3;
                case PhysicsCombineMode::Multiply: return 2;
                case PhysicsCombineMode::Average: return 1;
                case PhysicsCombineMode::Minimum: return 0;
            }
            return 1;
        };
        const auto pick = [&](PhysicsCombineMode a, PhysicsCombineMode b) { return priority(a) >= priority(b) ? a : b; };
        // A sub-shape naming its own material overrides what the body settings resolved.
        const auto friction = [](const ::PhysicsMaterial *m, const Body &b) { return m ? m->DynamicFriction : b.GetFriction(); };
        const auto restitution = [](const ::PhysicsMaterial *m, const Body &b) { return m ? m->Restitution : b.GetRestitution(); };
        s.mCombinedFriction = ApplyCombineMode(pick(fc(m1), fc(m2)), friction(m1, b1), friction(m2, b2));
        s.mCombinedRestitution = ApplyCombineMode(pick(rc(m1), rc(m2)), restitution(m1, b1), restitution(m2, b2));
    }
};

// One contact manifold's entries for a frame, accumulated with impulse weights.
// A manifold is reported once per collision step, so a frame holds one entry per step for the same manifold.
struct ManifoldMerge {
    Vec3 Point{Vec3::sZero()}, Normal{Vec3::sZero()}, Slip{Vec3::sZero()};
    Vec3 Local1{Vec3::sZero()}, Local2{Vec3::sZero()}; // Contact position in each body's own frame, for differencing into a sweep velocity.
    Vec3 FrictionImpulse{Vec3::sZero()};
    float Impulse{0};
};

// All Jolt-side physics state. Lives in the registry context (see physics::Init); the physics:: free
// functions resolve it from the registry. Plain data holder — behavior lives in the functions below.
struct PhysicsState {
    TempAllocatorImpl TempAllocator{64 * 1024 * 1024};
    JobSystemThreadPool JobSystem{cMaxPhysicsJobs, cMaxPhysicsBarriers, int(std::max(1u, std::thread::hardware_concurrency() - 1))};
    BPLayerInterface BPLayerIface;
    ObjectLayerPairFilterImpl ObjectPairFilter;
    ObjectVsBPLayerFilterImpl ObjectVsBPFilter;
    std::optional<PhysicsSystem> System; // optional only so ResetSystem can reinit it in place (PhysicsSystem is non-movable); always engaged.

    std::unordered_map<entt::entity, Ref<Constraint>> ConstraintsByJoint; // PhysicsJoint entity → its constraint.
    uint32_t CacheStartFrame{1};
    uint32_t CacheEndFrame{0}; // For range validation only — actual storage is per-entity BodyPoseCache.
    std::optional<uint32_t> Baked; // highest frame baked (inclusive); nullopt if nothing baked since last clear.
    Ref<KHRCollisionFilter> FilterRef;
    KHRContactListener ContactListener;
    MeshVsMeshShapeFilter MeshFilter;

    std::unordered_map<entt::entity, uint32_t> BodySubGroups; // entity -> KHRCollisionFilter SubGroupID.
    std::unordered_map<uint32_t, entt::entity> EntityByBodyIndex; // BodyID index -> owning entity, for contact-impact routing.
    std::vector<KHRContactListener::RawContact> ContactDrainScratch; // reused each step so the swap doesn't free the accumulator
    std::vector<KHRContactListener::RawSustained> SustainedDrainScratch;
    // The (packed body-index key, manifold key, raw index) tuples this step's contacts are gathered by,
    // and the body pairs that gather covered. Both come out sorted, so the ended-pair sweep binary-searches.
    std::vector<std::tuple<uint64_t, uint64_t, uint32_t>> SustainedOrderScratch;
    std::vector<uint64_t> SustainedPairScratch;
    // Live manifolds, in lists keyed by the packed body-index pair.
    // Each carries its own id and the previous step's contact point in each body's own frame, so the sweep measures
    // surfaces travelling rather than load moving between manifolds.
    struct SustainedManifold {
        uint64_t Key{0}; // Sub-shape pair, identifying the manifold within its body pair.
        uint64_t Id{0};
        vec3 Local1{0}, Local2{0};
        uint64_t LastStep{0};
    };
    std::unordered_map<uint64_t, std::vector<SustainedManifold>> SustainedManifolds;
    uint64_t NextSustainedId{1};
    uint64_t SustainedStep{0}; // Counts drains, so a manifold that stopped touching is dropped.

    std::vector<BodyID> PendingBodyRemovals; // queued for batched removal

    bool JointsDirty{false};

    float DefaultPenetrationSlop{0}, DefaultSpeculativeContactDistance{0};

    PhysicsState() : FilterRef(new KHRCollisionFilter()) {
        ResetSystem();
        ContactListener.Filter = FilterRef.GetPtr();
    }

    // Recreate the Jolt system so its BodyManager restarts clean and a scene's body ids are deterministic.
    // emplace() reinits in place, keeping JobSystem / TempAllocator alive.
    void ResetSystem() {
        auto &system = System.emplace();
        system.Init(
            65536, // max bodies
            0, // num body mutexes (0 = default)
            65536, // max body pairs
            65536, // max contact constraints
            BPLayerIface,
            ObjectVsBPFilter,
            ObjectPairFilter
        );
        system.SetContactListener(&ContactListener);
        system.SetSimShapeFilter(&MeshFilter);
        // Jolt MeshShape is single-sided by default.
        // Enable back-face collision so mesh colliders block from both sides.
        system.SetSimCollideBodyVsBody([](const Body &b1, const Body &b2, Mat44Arg t1, Mat44Arg t2,
                                          CollideShapeSettings &settings, CollideShapeCollector &collector,
                                          const ShapeFilter &filter) {
            settings.mBackFaceMode = EBackFaceMode::CollideWithBackFaces;
            PhysicsSystem::sDefaultSimCollideBodyVsBody(b1, b2, t1, t2, settings, collector, filter);
        });
        const auto settings = system.GetPhysicsSettings();
        DefaultPenetrationSlop = settings.mPenetrationSlop;
        DefaultSpeculativeContactDistance = settings.mSpeculativeContactDistance;
    }
};

// The two bodies a sustained contact is between, packed so one key orders the contacts by pair and indexes
// PhysicsState::SustainedManifolds. Only these two functions know the packing.
uint64_t BodyPairKey(uint32_t body1_index, uint32_t body2_index) { return (uint64_t{body1_index} << 32) | body2_index; }
std::array<uint32_t, 2> BodyPairIndices(uint64_t key) { return {uint32_t(key >> 32), uint32_t(key)}; }

constexpr float PlaneThickness{1e-3f}; // Y-thickness of the BoxShape used as a finite-plane / non-static-infinite-plane fallback

Ref<Shape> CreateJoltShape(const PhysicsShape &shape, const Mesh *mesh, bool body_is_static = false) {
    const Ref<Shape> js = std::visit(
        overloaded{
            // KHR spec uses full size, Jolt uses half-extents.
            [](const physics::Box &s) -> Ref<Shape> { return new BoxShape(ToJolt(s.Size * 0.5f)); },
            [](const physics::Sphere &s) -> Ref<Shape> { return new SphereShape(s.Radius); },
            [](const physics::Capsule &s) -> Ref<Shape> {
                assert(s.Height >= physics::MinShapeHeight);
                if (std::abs(s.RadiusTop - s.RadiusBottom) < 1e-6f) return new CapsuleShape(s.Height * 0.5f, s.RadiusBottom);
                if (const auto r = TaperedCapsuleShapeSettings(s.Height * 0.5f, s.RadiusTop, s.RadiusBottom).Create(); r.IsValid()) return r.Get();
                return new CapsuleShape(s.Height * 0.5f, s.RadiusBottom);
            },
            [](const physics::Cylinder &s) -> Ref<Shape> {
                assert(s.Height >= physics::MinShapeHeight);
                if (std::abs(s.RadiusTop - s.RadiusBottom) < 1e-6f) return new CylinderShape(s.Height * 0.5f, s.RadiusBottom);
                if (const auto r = TaperedCylinderShapeSettings(s.Height * 0.5f, s.RadiusTop, s.RadiusBottom).Create(); r.IsValid()) return r.Get();
                return new CylinderShape(s.Height * 0.5f, std::max(s.RadiusTop, s.RadiusBottom));
            },
            // Jolt PlaneShape is single-sided and static-only. Everything else collapses to a thin BoxShape;
            // infinite non-static uses a large half-extent.
            [body_is_static](const physics::Plane &s) -> Ref<Shape> {
                const bool is_infinite = s.SizeX <= 0.f || s.SizeZ <= 0.f;
                if (body_is_static && is_infinite && !s.DoubleSided) return new PlaneShape(Plane(Vec3(0, 1, 0), 0));
                const float hx = is_infinite ? PlaneShapeSettings::cDefaultHalfExtent : s.SizeX * 0.5f;
                const float hz = is_infinite ? PlaneShapeSettings::cDefaultHalfExtent : s.SizeZ * 0.5f;
                return new BoxShape(Vec3(hx, PlaneThickness * 0.5f, hz));
            },
            [mesh](const physics::ConvexHull &) -> Ref<Shape> {
                if (!mesh || mesh->VertexCount() == 0) return {};
                auto verts = mesh->GetVerticesSpan();
                // Jolt Vec3 is a 16-byte SIMD type — must convert from interleaved Vertex positions
                Array<Vec3> points;
                points.reserve(verts.size());
                for (const auto &v : verts) points.emplace_back(v.Position.x, v.Position.y, v.Position.z);
                const ConvexHullShapeSettings settings(points.data(), points.size());
                if (const auto r = settings.Create(); r.IsValid()) return r.Get();
                return {};
            },
            [mesh](const physics::TriangleMesh &) -> Ref<Shape> {
                if (!mesh || mesh->FaceCount() == 0) return {};
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
                const MeshShapeSettings settings{std::move(vertices), std::move(triangles)};
                if (const auto r = settings.Create(); r.IsValid()) return r.Get();
                return {};
            },
        },
        shape
    );
    return js ? js : Ref<Shape>(new BoxShape(Vec3(0.5f, 0.5f, 0.5f)));
}

// Apply motion and material properties to body creation settings.
void ApplyPhysicsProperties(BodyCreationSettings &bcs, const PhysicsMotion *motion, const PhysicsVelocity *velocity, bool is_mesh_shape, const ColliderMaterial *material, const entt::registry &r) {
    bcs.mUserData = NoMaterialSentinel;
    if (motion) {
        if (bcs.mMotionType == EMotionType::Dynamic) {
            // KHR spec §128: an explicit mass of 0 means infinite (lock translation).
            if (motion->Mass == 0.0f) {
                bcs.mAllowedDOFs = EAllowedDOFs::All & ~(EAllowedDOFs::TranslationX | EAllowedDOFs::TranslationY | EAllowedDOFs::TranslationZ);
            }
            bcs.mMassPropertiesOverride.mMass = ResolvedMass(*motion);
            bcs.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
        }
        // MeshShape can't compute mass properties - provide placeholders for any non-static body.
        if (is_mesh_shape && bcs.mMotionType != EMotionType::Static) {
            bcs.mOverrideMassProperties = EOverrideMassProperties::MassAndInertiaProvided;
            bcs.mMassPropertiesOverride.mMass = ResolvedMass(*motion);
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
    if (material && material->PhysicsMaterialEntity != null_entity) {
        if (const auto *mat = r.try_get<const ::PhysicsMaterial>(material->PhysicsMaterialEntity)) {
            bcs.mFriction = mat->DynamicFriction;
            bcs.mRestitution = mat->Restitution;
            bcs.mUserData = uint32_t(material->PhysicsMaterialEntity);
        }
    }
}

// Find the nearest ancestor (or self) that has a PhysicsBodyHandle.
entt::entity FindBodyAncestor(const entt::registry &r, entt::entity e) {
    return FindAncestorIf(r, e, [&](auto x) { return r.all_of<PhysicsBodyHandle>(x); });
}

void ConfigureJointSettings(SixDOFConstraintSettings &settings, const PhysicsJointDef &def) {
    // Per KHR_physics_rigid_bodies, only axes mentioned in limits are constrained.
    for (uint32_t a = 0; a < SixDOFConstraintSettings::EAxis::Num; ++a) {
        settings.MakeFreeAxis(SixDOFConstraintSettings::EAxis(a));
    }

    // Apply limits — each limit entry may cover multiple axes
    for (const auto &limit : def.Limits) {
        const auto configure_axis = [&](SixDOFConstraintSettings::EAxis axis) {
            if (!limit.Min && !limit.Max) settings.MakeFreeAxis(axis);
            else settings.SetLimitedAxis(axis, limit.Min.value_or(-FLT_MAX), limit.Max.value_or(FLT_MAX));
            if (limit.Stiffness) {
                if (axis < SixDOFConstraintSettings::EAxis::NumTranslation) {
                    settings.mLimitsSpringSettings[axis] = SpringSettings(ESpringMode::StiffnessAndDamping, *limit.Stiffness, limit.Damping);
                } else {
                    // Jolt's SixDOFConstraint has no spring path for angular limits — they're treated as hard.
                    // (See lib/JoltPhysics for an in-progress local patch adding this; not yet upstream.)
                    static constexpr const char *AxisNames[]{"X", "Y", "Z"};
                    const auto *const label = def.Name.empty() ? "<unnamed>" : def.Name.c_str();
                    std::cerr << std::format("Warning: joint '{}': soft angular limit on {} ignored (Jolt SixDOFConstraint lacks angular spring limits).\n", label, AxisNames[int(axis) - int(SixDOFConstraintSettings::EAxis::NumTranslation)]);
                }
            }
        };
        for (const uint8_t a : limit.LinearAxes) {
            if (a < 3) configure_axis(SixDOFConstraintSettings::EAxis(a));
        }
        for (const uint8_t a : limit.AngularAxes) {
            if (a < 3) configure_axis(SixDOFConstraintSettings::EAxis(a + 3));
        }
    }

    // Apply drives as motor settings.
    for (const auto &drive : def.Drives) {
        const auto axis_index = (drive.Type == PhysicsDriveType::Linear ? 0 : 3) + drive.Axis;
        const auto axis = SixDOFConstraintSettings::EAxis(axis_index);
        auto &motor = settings.mMotorSettings[axis_index];
        const auto spring_mode = drive.Mode == PhysicsDriveMode::Acceleration ? ESpringMode::MassNormalizedStiffnessAndDamping : ESpringMode::StiffnessAndDamping;
        motor.mSpringSettings = SpringSettings(spring_mode, drive.Stiffness, drive.Damping);
        if (drive.Type == PhysicsDriveType::Linear) motor.SetForceLimit(drive.MaxForce);
        else motor.SetTorqueLimit(drive.MaxForce);
        // Ensure the axis isn't fixed so the motor can act
        if (settings.IsFixedAxis(axis)) settings.MakeFreeAxis(axis);
    }
}

// Returns the angular drive axis (0/1/2) for hinge-like joints, -1 otherwise. Hinge-like = one
// continuous-spin angular drive + all 3 linear axes locked at 0 + the other 2 angular axes locked
// within ±5°. These route to HingeConstraint, which is continuous through ±π, rather than SixDOF,
// whose swing-twist decomposition is ill-conditioned there and jumps the bodies as it passes through.
int DetectHingeAxis(const PhysicsJointDef &def) {
    int spin_axis = -1;
    for (const auto &drive : def.Drives) {
        if (drive.Type != PhysicsDriveType::Angular) return -1;
        if (drive.Stiffness <= 0.0f || drive.Damping <= 0.0f || drive.VelocityTarget == 0.0f) return -1;
        if (spin_axis != -1) return -1;
        spin_axis = drive.Axis;
    }
    if (spin_axis < 0) return -1;
    bool linear_locked[3] = {false, false, false};
    bool angular_locked[3] = {false, false, false};
    for (const auto &lim : def.Limits) {
        const float lo = lim.Min.value_or(-FLT_MAX), hi = lim.Max.value_or(FLT_MAX);
        if (std::abs(lo) >= 0.09f || std::abs(hi) >= 0.09f) return -1; // ~5° / 9cm tolerance
        for (const uint8_t a : lim.LinearAxes)
            if (a < 3) linear_locked[a] = true;
        for (const uint8_t a : lim.AngularAxes)
            if (a < 3) angular_locked[a] = true;
    }
    for (int i = 0; i < 3; ++i)
        if (!linear_locked[i]) return -1;
    for (int i = 0; i < 3; ++i)
        if (i != spin_axis && !angular_locked[i]) return -1;
    return spin_axis;
}

// After constraint creation, set motor states and targets from drives.
void ApplyDriveTargets(SixDOFConstraint &constraint, const PhysicsJointDef &def) {
    auto target_pos = Vec3::sZero(), target_vel = Vec3::sZero();
    auto target_ang_vel = Vec3::sZero();
    auto target_orient = Quat::sIdentity();
    bool has_orient_target = false;
    for (const auto &drive : def.Drives) {
        const int axis_index = (drive.Type == PhysicsDriveType::Linear ? 0 : 3) + drive.Axis;
        const auto axis = SixDOFConstraintSettings::EAxis(axis_index);
        // PositionAndVelocity uses the spring formula `k*(posT-posC) + c*(velT-velC)`, matching KHR spec.
        // Jolt's pure Velocity mode is a rigid constraint that ignores spring settings.
        const bool active = drive.Stiffness > 0.0f || drive.Damping > 0.0f;
        constraint.SetMotorState(axis, active ? EMotorState::PositionAndVelocity : EMotorState::Off);
        const bool has_position = drive.Stiffness > 0.0f;
        const bool has_velocity = drive.Damping > 0.0f;
        if (has_position) {
            if (drive.Type == PhysicsDriveType::Linear) {
                target_pos.SetComponent(drive.Axis, drive.PositionTarget);
            } else {
                // Compose per-axis angular position targets into a quaternion
                static const Vec3 axes[]{Vec3::sAxisX(), Vec3::sAxisY(), Vec3::sAxisZ()};
                target_orient = target_orient * Quat::sRotation(axes[drive.Axis], drive.PositionTarget);
                has_orient_target = true;
            }
        }
        if (has_velocity) {
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

void RecomputeSceneScale(PhysicsState &s, const entt::registry &r) {
    // Scale physics tolerances to scene size (default slop is too large for small scenes).
    float min_dim = std::numeric_limits<float>::max();
    for (auto [entity, collider] : r.view<const ColliderShape>().each()) {
        std::visit(
            overloaded{
                [&](const physics::Sphere &s) { min_dim = std::min(min_dim, s.Radius); },
                [&](const physics::Box &s) { min_dim = std::min({min_dim, s.Size.x, s.Size.y, s.Size.z}); },
                [&](const physics::Capsule &s) { min_dim = std::min({min_dim, s.RadiusTop, s.RadiusBottom, s.Height}); },
                [&](const physics::Cylinder &s) { min_dim = std::min({min_dim, s.RadiusTop, s.RadiusBottom, s.Height}); },
                [](const auto &) {}, // Plane / mesh shapes — no analytic size
            },
            collider.Shape
        );
    }
    // Shrink for small colliders, and grow back (capped at default) when they're removed or embiggened.
    auto settings = s.System->GetPhysicsSettings();
    settings.mPenetrationSlop = (min_dim < std::numeric_limits<float>::max()) ?
        std::min(s.DefaultPenetrationSlop, min_dim * 0.02f) :
        s.DefaultPenetrationSlop;
    settings.mSpeculativeContactDistance = (min_dim < std::numeric_limits<float>::max()) ?
        std::min(s.DefaultSpeculativeContactDistance, min_dim * 0.02f) :
        s.DefaultSpeculativeContactDistance;
    s.System->SetPhysicsSettings(settings);
}

void BuildJoint(PhysicsState &s, const entt::registry &r, entt::entity entity) {
    const auto *joint_p = r.try_get<const PhysicsJoint>(entity);
    if (!joint_p || joint_p->ConnectedNode == null_entity || joint_p->JointDefEntity == null_entity) return;
    const auto *def_p = r.try_get<const PhysicsJointDef>(joint_p->JointDefEntity);
    if (!def_p) return;
    const auto &joint = *joint_p;
    const auto &def = *def_p;

    // Body 1: nearest ancestor (or self) with a physics body. Body 2: ancestor body, or world if none.
    const auto body1_entity = FindBodyAncestor(r, entity);
    const auto *h1 = body1_entity != null_entity ? r.try_get<const PhysicsBodyHandle>(body1_entity) : nullptr;
    if (!h1) return;
    const auto body2_entity = FindBodyAncestor(r, joint.ConnectedNode);
    const auto *h2 = body2_entity != null_entity ? r.try_get<const PhysicsBodyHandle>(body2_entity) : nullptr;

    // Joint and connected-node world transforms define the joint frame on each body.
    Vec3 axis_x1 = Vec3::sAxisX(), axis_y1 = Vec3::sAxisY();
    Vec3 axis_x2 = Vec3::sAxisX(), axis_y2 = Vec3::sAxisY();
    RVec3 pos1 = RVec3::sZero(), pos2 = RVec3::sZero();
    if (const auto *jt = r.try_get<const WorldTransform>(entity)) {
        const auto rot_mat = numeric::ToMat3(numeric::Normalize(jt->R));
        pos1 = pos2 = ToJolt(jt->P);
        axis_x1 = axis_x2 = ToJolt(rot_mat[0]);
        axis_y1 = axis_y2 = ToJolt(rot_mat[1]);
    }
    if (const auto *ct = r.try_get<const WorldTransform>(joint.ConnectedNode)) {
        const auto rot_mat = numeric::ToMat3(numeric::Normalize(ct->R));
        pos2 = ToJolt(ct->P);
        axis_x2 = ToJolt(rot_mat[0]);
        axis_y2 = ToJolt(rot_mat[1]);
    }

    const auto &lock_iface = s.System->GetBodyLockInterfaceNoLock();
    const BodyLockWrite lock1{lock_iface, BodyID(h1->BodyId)};
    std::optional<BodyLockWrite> lock2_opt;
    if (h2) lock2_opt.emplace(lock_iface, BodyID{h2->BodyId});
    if (!lock1.Succeeded() || (lock2_opt && !lock2_opt->Succeeded())) return;

    auto &b1 = lock1.GetBody();
    auto &b2 = h2 ? lock2_opt->GetBody() : Body::sFixedToWorld;

    Constraint *constraint = nullptr;
    if (const int hinge_axis = DetectHingeAxis(def); hinge_axis >= 0) {
        HingeConstraintSettings hs;
        hs.mSpace = EConstraintSpace::WorldSpace;
        hs.mPoint1 = pos1;
        hs.mPoint2 = pos2;
        const Vec3 frame1[3]{axis_x1, axis_y1, axis_x1.Cross(axis_y1)};
        const Vec3 frame2[3]{axis_x2, axis_y2, axis_x2.Cross(axis_y2)};
        hs.mHingeAxis1 = frame1[hinge_axis];
        hs.mHingeAxis2 = frame2[hinge_axis];
        hs.mNormalAxis1 = frame1[(hinge_axis + 1) % 3];
        hs.mNormalAxis2 = frame2[(hinge_axis + 1) % 3];
        hs.mLimitsMin = -JPH_PI;
        hs.mLimitsMax = JPH_PI;
        const auto &drive = def.Drives.front();
        const auto spring_mode = drive.Mode == PhysicsDriveMode::Acceleration ? ESpringMode::MassNormalizedStiffnessAndDamping : ESpringMode::StiffnessAndDamping;
        hs.mMotorSettings.mSpringSettings = SpringSettings(spring_mode, drive.Stiffness, drive.Damping);
        hs.mMotorSettings.SetTorqueLimit(drive.MaxForce);
        auto *hinge = static_cast<HingeConstraint *>(hs.Create(b1, b2));
        hinge->SetMotorState(EMotorState::Velocity);
        hinge->SetTargetAngularVelocity(drive.VelocityTarget);
        constraint = hinge;
    } else {
        SixDOFConstraintSettings settings;
        settings.mSpace = EConstraintSpace::WorldSpace;
        // Default Cone swing couples Y/Z into an ellipse, which degenerates when one
        // axis is locked. Pyramid gives independent per-axis angular limits.
        settings.mSwingType = ESwingType::Pyramid;
        settings.mPosition1 = pos1;
        settings.mPosition2 = pos2;
        settings.mAxisX1 = axis_x1;
        settings.mAxisY1 = axis_y1;
        settings.mAxisX2 = axis_x2;
        settings.mAxisY2 = axis_y2;
        ConfigureJointSettings(settings, def);
        auto *six = static_cast<SixDOFConstraint *>(settings.Create(b1, b2));
        ApplyDriveTargets(*six, def);
        constraint = six;
    }
    s.System->AddConstraint(constraint);
    s.ConstraintsByJoint[entity] = constraint;

    // Keep all non-static bodies in constraints awake.
    if (!b1.IsStatic()) b1.SetAllowSleeping(false);
    if (!b2.IsStatic()) b2.SetAllowSleeping(false);

    // Disable collision between connected bodies unless explicitly enabled.
    if (!joint.EnableCollision && h2) {
        if (const auto it1 = s.BodySubGroups.find(body1_entity), it2 = s.BodySubGroups.find(body2_entity);
            it1 != s.BodySubGroups.end() && it2 != s.BodySubGroups.end())
            s.FilterRef->DisableCollision(it1->second, it2->second);
    }
}

void FlushJoints(PhysicsState &s, const entt::registry &r, bool joints_changed) {
    if (!joints_changed && !s.JointsDirty) return;
    s.JointsDirty = false;

    for (auto &[_, c] : s.ConstraintsByJoint) s.System->RemoveConstraint(c);
    s.ConstraintsByJoint.clear();
    s.FilterRef->ResetDisabledPairs();
    for (auto entity : r.view<const PhysicsJoint>()) BuildJoint(s, r, entity);
    s.FilterRef->FinalizeDisabledPairs();
}

entt::entity FindMotionOwner(const entt::registry &r, entt::entity e) {
    return FindAncestorIf(r, e, [&](auto x) { return r.all_of<PhysicsMotion>(x); });
}
entt::entity FindCompoundParentBody(const entt::registry &r, entt::entity e) {
    if (r.all_of<PhysicsBodyHandle>(e)) return entt::null;
    return FindAncestorIf(r, GetParentEntity(r, e), [&](auto x) { return r.all_of<PhysicsMotion, PhysicsBodyHandle>(x); });
}
bool IsJointConstrained(const entt::registry &r, entt::entity motion_owner) {
    for (auto [je, _] : r.view<const PhysicsJoint>().each()) {
        if (FindMotionOwner(r, je) == motion_owner) return true;
    }
    return false;
}

// True iff some static TriangleMesh collider's filter mask permits contact with this filter.
bool CouldHitStaticMesh(const entt::registry &r, entt::entity filter_entity, const KHRCollisionFilter *filter) {
    for (auto [e, cs] : r.view<const ColliderShape>().each()) {
        if (!std::holds_alternative<physics::TriangleMesh>(cs.Shape)) continue;
        if (FindMotionOwner(r, e) != null_entity) continue; // not static
        const auto *cm = r.try_get<const ColliderMaterial>(e);
        const auto other = cm ? cm->CollisionFilterEntity : null_entity;
        if (filter_entity == null_entity || other == null_entity) return true;
        if (!filter || filter->MasksCollide(filter_entity, other)) return true;
    }
    return false;
}

// Promote dynamic TriangleMesh→ConvexHull when all conditions are met:
// Owner is not kinematic. mass > 0. Filters allow contact with some static TriangleMesh. Not joint-constrained.
bool ShouldPromoteMesh(const entt::registry &r, entt::entity motion_owner, const PhysicsMotion &motion, entt::entity filter_entity, const KHRCollisionFilter *filter) {
    if (motion.IsKinematic) return false;
    if (motion.Mass.value_or(0.f) <= 0.f) return false; // mass=0 = locked translation, keep concave mesh
    if (!CouldHitStaticMesh(r, filter_entity, filter)) return false;
    if (IsJointConstrained(r, motion_owner)) return false;
    return true;
}

Ref<Shape> BuildLeafShape(const entt::registry &r, entt::entity entity, const ColliderShape &cs, const ColliderMaterial *cm, const PhysicsMotion *owner_motion, entt::entity owner_entity, const KHRCollisionFilter *filter) {
    auto shape_proto = cs.Shape;
    const auto filter_entity = cm ? cm->CollisionFilterEntity : null_entity;
    if (std::holds_alternative<physics::TriangleMesh>(shape_proto) &&
        owner_motion && ShouldPromoteMesh(r, owner_entity, *owner_motion, filter_entity, filter)) {
        shape_proto = physics::ConvexHull{};
    }
    const auto mesh_entity = IsMeshBackedShape(shape_proto) ? cs.MeshEntity : null_entity;
    const auto mesh = mesh_entity != null_entity ? TryGetMesh(r, mesh_entity) : std::nullopt;
    auto js = CreateJoltShape(shape_proto, mesh ? &*mesh : nullptr, owner_motion == nullptr);
    if (!js) return {};
    js->SetUserData(ShapeUserData(entity, filter_entity));
    // Offset is in entity-local pre-scale coords (matches CenterOfMass convention) — translate before scaling.
    if (cs.LocalOffset != vec3{0}) js = new RotatedTranslatedShape(ToJolt(cs.LocalOffset), Quat::sIdentity(), js);
    const auto *t = r.try_get<const WorldTransform>(entity);
    if (t && HasNonUnitScale(*t)) js = new ScaledShape(js, ToJolt(t->S));
    return js;
}

// Walk descendants of `owner`, stopping at any descendant PhysicsMotion, collecting ColliderShape entities.
void GatherCompoundChildren(const entt::registry &r, entt::entity owner, std::vector<entt::entity> &out) {
    const auto walk = [&](this auto &self, entt::entity e) -> void {
        for (auto child : Children{&r, e}) {
            if (!r.all_of<PhysicsMotion>(child)) {
                if (r.all_of<ColliderShape>(child)) out.emplace_back(child);
                self(child);
            }
        }
    };
    walk(owner);
}

struct BodyShape {
    Ref<Shape> Shape; // null = no body should be created here
    bool IsSensor{false};
    bool IsMeshShape{false}; // TriangleMesh leaf — needs mass-properties placeholders
    const ColliderMaterial *SingleMaterial{nullptr}; // material for body settings (single-collider bodies only)
    entt::entity FilterEntity{null_entity}; // resolved collision-filter entity (null = no KHR filter)
};

// Build the body's complete shape (single collider, compound, or sensor). Does NOT apply CoM wrap.
// Skips child colliders whose parent motion-owner builds them as compound children.
BodyShape BuildBodyShape(const entt::registry &r, entt::entity entity, const KHRCollisionFilter *filter) {
    BodyShape out;
    const auto *motion = r.try_get<const PhysicsMotion>(entity);
    const auto *collider = r.try_get<const ColliderShape>(entity);
    const bool is_trigger = collider && r.all_of<const TriggerTag>(entity);

    if (is_trigger) {
        out.IsSensor = true;
        out.IsMeshShape = std::holds_alternative<physics::TriangleMesh>(collider->Shape);
        const auto *cm = r.try_get<const ColliderMaterial>(entity);
        out.FilterEntity = ResolveFilterEntity(r, cm ? cm->CollisionFilterEntity : null_entity);
        const auto mesh_entity = IsMeshBackedShape(collider->Shape) ? collider->MeshEntity : null_entity;
        const auto mesh = mesh_entity != null_entity ? TryGetMesh(r, mesh_entity) : std::nullopt;
        auto js = CreateJoltShape(collider->Shape, mesh ? &*mesh : nullptr);
        if (!js) return out;
        if (collider->LocalOffset != vec3{0}) js = new RotatedTranslatedShape(ToJolt(collider->LocalOffset), Quat::sIdentity(), js);
        const auto *t = r.try_get<const WorldTransform>(entity);
        if (t && HasNonUnitScale(*t)) js = new ScaledShape(js, ToJolt(t->S));
        out.Shape = js;
        return out;
    }

    auto build_single = [&](const PhysicsMotion *m, entt::entity owner) {
        out.SingleMaterial = r.try_get<const ColliderMaterial>(entity);
        out.IsMeshShape = std::holds_alternative<physics::TriangleMesh>(collider->Shape);
        out.FilterEntity = ResolveFilterEntity(r, out.SingleMaterial ? out.SingleMaterial->CollisionFilterEntity : null_entity);
        out.Shape = BuildLeafShape(r, entity, *collider, out.SingleMaterial, m, owner, filter);
    };

    if (motion) {
        std::vector<entt::entity> colliders;
        if (collider) colliders.emplace_back(entity);
        GatherCompoundChildren(r, entity, colliders);
        if (colliders.empty()) {
            // KHR_physics_rigid_bodies: `motion` alone defines a rigid body even without a collider.
            // Use EmptyShape so Jolt can still create the body — it collides with nothing.
            out.Shape = new EmptyShape();
            return out;
        }
        if (colliders.size() == 1 && colliders[0] == entity) {
            build_single(motion, entity);
            return out;
        }
        const auto *bt = r.try_get<const WorldTransform>(entity);
        const auto inv_parent = bt ? numeric::Inverse(numeric::Translate(mat4{1}, bt->P) * numeric::ToMat4(numeric::Normalize(bt->R))) : mat4{1};
        StaticCompoundShapeSettings compound;
        // Pull friction/restitution from the first child collider's material.
        // Otherwise compound bodies fall back to Jolt's BCS default instead of material value.
        for (auto ce : colliders) {
            const auto &cs = r.get<const ColliderShape>(ce);
            const auto *cm = r.try_get<const ColliderMaterial>(ce);
            if (!out.SingleMaterial && cm) out.SingleMaterial = cm;
            const auto sub = BuildLeafShape(r, ce, cs, cm, motion, entity, filter);
            if (!sub) continue;
            if (ce == entity) compound.AddShape(Vec3::sZero(), Quat::sIdentity(), sub);
            else {
                const auto *wt = r.try_get<const WorldTransform>(ce);
                const auto rel = inv_parent * (wt ? ToMatrix(*wt) : mat4{1});
                compound.AddShape(ToJolt(vec3{rel[3]}), ToJolt(numeric::Normalize(numeric::ToQuat(mat3{rel}))), sub);
            }
        }
        if (!compound.mSubShapes.empty())
            if (const auto result = compound.Create(); result.IsValid()) out.Shape = result.Get();
        return out;
    }

    // Static leaf — skip if a motion ancestor exists (this leaf is a compound child).
    if (collider && FindMotionOwner(r, GetParentEntity(r, entity)) == entt::null) build_single(nullptr, entt::null);
    return out;
}

// Wrap with OffsetCenterOfMassShape (KHR semantics: CoM is an absolute local-space point).
// Writes the inner mass props the caller needs to override the BCS with (avoids PAT inflation).
// glTF CoM is in node-local pre-scale coords. Jolt body frames are world-meters, so scale by world_scale to convert.
Ref<Shape> WrapCenterOfMass(Ref<Shape> inner, const PhysicsMotion *motion, vec3 world_scale, MassProperties &out_inner_mass_props) {
    if (!motion || !motion->CenterOfMass) return inner;
    out_inner_mass_props = inner->GetMassProperties();
    return new OffsetCenterOfMassShape(inner, ToJolt(*motion->CenterOfMass * world_scale) - inner->GetCenterOfMass());
}

void AddBody(PhysicsState &s, entt::registry &r, entt::entity entity) {
    if (r.all_of<PhysicsBodyHandle>(entity)) return;

    const auto built = BuildBodyShape(r, entity, s.FilterRef.GetPtr());
    if (!built.Shape) return;

    const auto *motion = r.try_get<const PhysicsMotion>(entity);
    const auto *t = r.try_get<const WorldTransform>(entity);
    const auto world_scale = t ? t->S : vec3{1};
    MassProperties inner_mass_props;
    const auto shape = WrapCenterOfMass(built.Shape, built.IsSensor ? nullptr : motion, world_scale, inner_mass_props);

    const auto pos = t ? ToJolt(t->P) : RVec3::sZero();
    const auto rot = t ? ToJolt(t->R) : Quat::sIdentity();
    const auto motion_type = motion ? (motion->IsKinematic ? EMotionType::Kinematic : EMotionType::Dynamic) : EMotionType::Static;
    const auto layer = motion ? Layers::Moving : Layers::NonMoving;

    BodyCreationSettings bcs{shape, pos, rot, motion_type, layer};
    bcs.mIsSensor = built.IsSensor;
    ApplyPhysicsProperties(bcs, motion, r.try_get<const PhysicsVelocity>(entity), built.IsMeshShape, built.SingleMaterial, r);

    if (motion && motion->CenterOfMass && bcs.mMotionType == EMotionType::Dynamic && !built.IsSensor) {
        inner_mass_props.ScaleToMass(bcs.mMassPropertiesOverride.mMass);
        bcs.mMassPropertiesOverride = inner_mass_props;
        bcs.mOverrideMassProperties = EOverrideMassProperties::MassAndInertiaProvided;
    }

    if (s.FilterRef) {
        const uint32_t sub = s.FilterRef->RegisterBody(built.FilterEntity);
        bcs.mCollisionGroup = CollisionGroup(s.FilterRef, 0, sub);
        s.BodySubGroups[entity] = sub;
    }

    auto &bi = s.System->GetBodyInterface();
    const auto *body = bi.CreateBody(bcs);
    if (!body) return;
    bi.AddBody(body->GetID(), motion_type == EMotionType::Static ? EActivation::DontActivate : EActivation::Activate);
    s.EntityByBodyIndex[body->GetID().GetIndex()] = entity;
    r.emplace_or_replace<PhysicsBodyHandle>(entity, PhysicsBodyHandle{body->GetID().GetIndexAndSequenceNumber()});
    r.emplace_or_replace<BodyPoseCache>(entity);

    if (motion && body->IsDynamic() && !built.IsSensor) {
        auto *mp = const_cast<Body *>(body)->GetMotionProperties();
        if (motion->Mass == 0) mp->SetInverseMass(0); // KHR §128: explicit zero = infinite mass
        ApplyInertiaDiagonal(*mp, *motion);
    }

    s.JointsDirty = true;
}

// on_destroy<PhysicsBodyHandle>: captures the BodyId before the component is erased (on component removal or
// full entity destroy), so the body can be dropped in the next batched flush.
void OnDestroyPhysicsBody(entt::registry &r, entt::entity entity) {
    auto *s = r.ctx().find<PhysicsState>();
    if (!s) return; // registry teardown after physics::Deinit erased the state
    // Reported contacts are refilled only while stepping, so drop this body's records rather than let them
    // outlive it while the playhead is parked.
    if (auto *sustained = r.ctx().find<PhysicsSustainedContacts>()) {
        std::erase_if(sustained->Active, [entity](const SustainedContact &c) {
            return c.Sides.front().Entity == entity || c.Sides.back().Entity == entity;
        });
    }
    if (auto *impacts = r.ctx().find<PhysicsContactImpacts>()) {
        std::erase_if(impacts->Events, [entity](const ContactImpact &i) { return i.Entity == entity || i.Other == entity; });
    }
    if (const auto &handle = r.get<const PhysicsBodyHandle>(entity); handle.BodyId != UINT32_MAX) {
        const auto index = BodyID{handle.BodyId}.GetIndex();
        s->PendingBodyRemovals.emplace_back(handle.BodyId);
        s->EntityByBodyIndex.erase(index);
        // Jolt reuses body indices, so per-manifold state left behind would attach to the next body taking this index.
        std::erase_if(s->SustainedManifolds, [index](const auto &pair) { return std::ranges::contains(BodyPairIndices(pair.first), index); });
    }
    s->BodySubGroups.erase(entity);
    s->JointsDirty = true;
}

// Batched removal of bodies queued by OnDestroyPhysicsBody. See PendingBodyRemovals.
void FlushPendingBodyRemovals(PhysicsState &s) {
    auto &ids = s.PendingBodyRemovals;
    if (ids.empty()) return;
    auto &bi = s.System->GetBodyInterface();
    bi.RemoveBodies(ids.data(), int(ids.size()));
    bi.DestroyBodies(ids.data(), int(ids.size()));
    ids.clear();
}

// Drops the entity's body; the on_destroy hook queues the Jolt removal.
void RemoveBody(entt::registry &r, entt::entity entity) {
    r.remove<PhysicsBodyHandle>(entity);
    r.remove<BodyPoseCache>(entity);
}

void ApplyMassPropertiesFromShape(PhysicsState &s, const entt::registry &r, entt::entity entity) {
    const auto *handle = r.try_get<const PhysicsBodyHandle>(entity);
    const auto *motion = r.try_get<const PhysicsMotion>(entity);
    if (!handle || !motion) return;
    if (motion->InertiaDiagonal) return; // explicit override wins
    if (motion->Mass == 0.0f) return; // KHR §128 infinite mass — handled separately

    const BodyLockWrite lock(s.System->GetBodyLockInterface(), BodyID{handle->BodyId});
    if (!lock.Succeeded()) return;
    auto &body = lock.GetBody();
    if (!body.IsDynamic()) return; // sensors, static, kinematic skip

    auto props = body.GetShape()->GetMassProperties();
    props.ScaleToMass(ResolvedMass(*motion));
    body.GetMotionProperties()->SetMassProperties(EAllowedDOFs::All, props);
}

void ApplyShape(PhysicsState &s, const entt::registry &r, entt::entity entity) {
    const auto *handle = r.try_get<const PhysicsBodyHandle>(entity);
    if (!handle) return;
    const auto built = BuildBodyShape(r, entity, s.FilterRef.GetPtr());
    if (!built.Shape) return;
    MassProperties inner_mass_props;
    const auto *t = r.try_get<const WorldTransform>(entity);
    const auto world_scale = t ? t->S : vec3{1};
    const auto shape = WrapCenterOfMass(built.Shape, built.IsSensor ? nullptr : r.try_get<const PhysicsMotion>(entity), world_scale, inner_mass_props);
    // updateMassProperties=false: preserves explicit Mass/Inertia overrides and avoids
    // GetMassProperties() on shapes that return zero inertia (TriangleMesh, MeshShape).
    // ApplyMassPropertiesFromShape re-derives mass props with the right guards.
    s.System->GetBodyInterface().SetShape(BodyID{handle->BodyId}, shape, /*updateMassProperties=*/false, EActivation::Activate);
    ApplyMassPropertiesFromShape(s, r, entity);
}

void ApplyMotion(PhysicsState &s, const entt::registry &r, entt::entity entity) {
    const auto *handle = r.try_get<const PhysicsBodyHandle>(entity);
    const auto *motion = r.try_get<const PhysicsMotion>(entity);
    if (!handle || !motion) return;

    auto &bi = s.System->GetBodyInterface();
    const BodyID id{handle->BodyId};

    // Kinematic toggle is a motion-type flip (Dynamic ↔ Kinematic, both stay in Layers::Moving).
    if (const auto desired_type = motion->IsKinematic ? EMotionType::Kinematic : EMotionType::Dynamic;
        bi.GetMotionType(id) != desired_type) bi.SetMotionType(id, desired_type, EActivation::Activate);

    bi.SetGravityFactor(id, motion->GravityFactor);

    {
        const BodyLockWrite lock(s.System->GetBodyLockInterface(), id);
        if (!lock.Succeeded()) return;
        auto &body = lock.GetBody();
        if (!body.IsDynamic() && !body.IsKinematic()) return;

        auto *mp = body.GetMotionProperties();
        mp->SetLinearDamping(motion->LinearDamping);
        mp->SetAngularDamping(motion->AngularDamping);

        if (body.IsDynamic()) {
            // KHR §128: explicit zero Mass = infinite mass (locked translation).
            mp->SetInverseMass(motion->Mass == 0.0f ? 0.f : 1.f / ResolvedMass(*motion));
            ApplyInertiaDiagonal(*mp, *motion);
        }
    }

    // CenterOfMass (set or cleared) requires SetShape with a refreshed OffsetCenterOfMassShape wrapper.
    // ApplyShape also re-derives mass props from the new shape for the no-override case.
    ApplyShape(s, r, entity);
}

void ApplyMaterial(PhysicsState &s, const entt::registry &r, entt::entity entity) {
    const auto *material = r.try_get<const ColliderMaterial>(entity);
    if (!material) return;

    if (const auto owner = FindCompoundParentBody(r, entity); owner != entt::null) {
        // Child collider: leaf UserData lives inside the parent compound shape — rebuild it.
        ApplyShape(s, r, owner);
        return;
    }
    const auto *handle = r.try_get<const PhysicsBodyHandle>(entity);
    if (!handle) return;

    auto &bi = s.System->GetBodyInterface();
    const BodyID id{handle->BodyId};

    const auto mat_entity = material->PhysicsMaterialEntity;
    const auto *mat = mat_entity != null_entity && r.valid(mat_entity) ? r.try_get<const ::PhysicsMaterial>(mat_entity) : nullptr;
    const ::PhysicsMaterial defaults;
    const auto &m = mat ? *mat : defaults;
    bi.SetFriction(id, m.DynamicFriction);
    bi.SetRestitution(id, m.Restitution);

    const auto filter_entity = ResolveFilterEntity(r, material->CollisionFilterEntity);
    if (s.FilterRef) {
        if (const auto it = s.BodySubGroups.find(entity); it != s.BodySubGroups.end()) {
            s.FilterRef->SetBodyFilter(it->second, filter_entity);
        }
    }

    const BodyLockWrite lock(s.System->GetBodyLockInterface(), id);
    if (lock.Succeeded()) {
        auto &body = lock.GetBody();
        auto *leaf = const_cast<Shape *>(body.GetShape());
        while (leaf->GetType() == EShapeType::Decorated) {
            leaf = const_cast<Shape *>(static_cast<const DecoratedShape *>(leaf)->GetInnerShape());
        }
        leaf->SetUserData(ShapeUserData(entity, filter_entity));
        // Body UserData stores the material entity value for contact-listener lookup.
        body.SetUserData(mat ? uint32_t(mat_entity) : NoMaterialSentinel);
    }
}

void OnShapeChange(PhysicsState &s, entt::registry &r, entt::entity e) {
    if (!r.valid(e)) return;
    const bool has_shape = r.all_of<ColliderShape>(e);
    const bool has_body = r.all_of<PhysicsBodyHandle>(e);
    const auto compound_owner = FindCompoundParentBody(r, e);
    if (!has_shape) {
        if (has_body) RemoveBody(r, e);
        else if (compound_owner != entt::null) ApplyShape(s, r, compound_owner);
        return;
    }
    if (compound_owner != entt::null) ApplyShape(s, r, compound_owner);
    else if (has_body) ApplyShape(s, r, e);
    else AddBody(s, r, e);
}

void OnMotionChange(PhysicsState &s, entt::registry &r, entt::entity e) {
    if (!r.valid(e)) return;
    const bool has_motion = r.all_of<PhysicsMotion>(e);
    const bool has_body = r.all_of<PhysicsBodyHandle>(e);
    if (!has_motion) {
        if (has_body) RemoveBody(r, e);
        if (r.all_of<ColliderShape>(e)) AddBody(s, r, e); // demote: rebuild as static
        return;
    }
    // Object layer is fixed at body creation. Static lives in NonMoving; Dynamic/Kinematic in Moving.
    // Crossing that boundary needs a full recreate; Dynamic↔Kinematic within Moving can cheap-apply.
    if (has_body) {
        const auto *handle = r.try_get<const PhysicsBodyHandle>(e);
        const bool is_static = s.System->GetBodyInterface().GetMotionType(BodyID{handle->BodyId}) == EMotionType::Static;
        if (is_static) {
            RemoveBody(r, e);
            AddBody(s, r, e);
        } else {
            ApplyMotion(s, r, e);
        }
    } else {
        AddBody(s, r, e); // motion appeared on an entity that didn't have a body yet
    }
}

void OnMaterialChange(PhysicsState &s, entt::registry &r, entt::entity e) {
    if (!r.valid(e) || !r.all_of<ColliderMaterial>(e)) return;
    ApplyMaterial(s, r, e);
}

void OnTriggerChange(PhysicsState &s, entt::registry &r, entt::entity e) {
    if (!r.valid(e)) return;
    // Body's sensor flag (part of BodyCreationSettings) is baked at create, so any transition requires a full rebuild.
    // AddBody is a no-op when no shape is available.
    if (r.all_of<PhysicsBodyHandle>(e)) RemoveBody(r, e);
    AddBody(s, r, e);
}

// Authored Transform edit: mirror the new world pose onto e's body and any descendant bodies.
// Scale is baked into the shape, so a scale change rebuilds every descendant's shape.
// The simulator never writes WorldTransform::S, so it still holds the scale the shapes were built at.
void OnPoseChange(PhysicsState &s, entt::registry &r, entt::entity e) {
    if (!r.valid(e)) return;
    std::vector<entt::entity> bodies;
    const auto gather = [&](this auto &&self, entt::entity x) -> void {
        if (r.all_of<PhysicsBodyHandle>(x)) bodies.emplace_back(x);
        for (auto child : Children{&r, x}) self(child);
    };
    gather(e);
    if (bodies.empty()) return;

    const auto *ewt = r.try_get<const WorldTransform>(e);
    const vec3 old_scale = ewt ? ewt->S : vec3{1};
    UpdateWorldTransformRecursive(r, e);
    ewt = r.try_get<const WorldTransform>(e);
    const bool scale_changed = ewt && ewt->S != old_scale;

    auto &bi = s.System->GetBodyInterface();
    for (auto b : bodies) {
        const auto *t = r.try_get<const WorldTransform>(b);
        if (!t) continue;
        if (scale_changed) ApplyShape(s, r, b);
        const BodyID id{r.get<const PhysicsBodyHandle>(b).BodyId};
        const auto activation = bi.GetMotionType(id) == EMotionType::Static ? EActivation::DontActivate : EActivation::Activate;
        bi.SetPositionAndRotation(id, ToJolt(t->P), ToJolt(numeric::Normalize(t->R)), activation);
    }
}

// Deduce the owner class from a data-member pointer
template<typename M> struct ptr_class;
template<typename C, typename V> struct ptr_class<V C::*> {
    using type = C;
};

// Patches every component instance whose `.*Field` equals `deleted`, setting it to null_entity.
template<auto Field>
void ClearDanglingRefs(entt::registry &r, entt::entity deleted) {
    using C = typename ptr_class<decltype(Field)>::type;
    for (auto [e, c] : r.view<C>().each()) {
        if (c.*Field == deleted) r.patch<C>(e, [](C &x) { x.*Field = null_entity; });
    }
}

template<auto Field>
void ClearDanglingRefsFromVector(entt::registry &r, entt::entity deleted) {
    using C = typename ptr_class<decltype(Field)>::type;
    for (auto [e, c] : r.view<C>().each()) {
        const auto &vec = c.*Field;
        if (std::find(vec.begin(), vec.end(), deleted) != vec.end()) {
            r.patch<C>(e, [deleted](C &x) { std::erase(x.*Field, deleted); });
        }
    }
}

void OnPhysicsMaterialDefChange(PhysicsState &s, entt::registry &r, entt::entity e) {
    if (!r.valid(e) || !r.all_of<::PhysicsMaterial>(e)) {
        ClearDanglingRefs<&ColliderMaterial::PhysicsMaterialEntity>(r, e);
        return;
    }
    // Jolt friction is per-body: direct colliders update in place. Compound leaves require a parent rebuild.
    const auto &mat = r.get<const ::PhysicsMaterial>(e);
    auto &bi = s.System->GetBodyInterface();
    for (auto [ce, cm] : r.view<const ColliderMaterial>().each()) {
        if (cm.PhysicsMaterialEntity != e) continue;
        if (const auto *bh = r.try_get<const PhysicsBodyHandle>(ce)) {
            const BodyID id{bh->BodyId};
            bi.SetFriction(id, mat.DynamicFriction);
            bi.SetRestitution(id, mat.Restitution);
            bi.ActivateBody(id);
        } else if (const auto owner = FindCompoundParentBody(r, ce); owner != entt::null) {
            ApplyShape(s, r, owner);
        }
    }
}

void OnCollisionSystemDefChange(PhysicsState &s, entt::registry &r, entt::entity e) {
    if (!r.valid(e) || !r.all_of<CollisionSystem>(e)) {
        ClearDanglingRefsFromVector<&CollisionFilter::Systems>(r, e);
        ClearDanglingRefsFromVector<&CollisionFilter::CollideSystems>(r, e);
    }
    s.FilterRef->Update(r);
}

void OnCollisionFilterDefChange(PhysicsState &s, entt::registry &r, entt::entity e) {
    if (!r.valid(e) || !r.all_of<CollisionFilter>(e)) {
        ClearDanglingRefs<&ColliderMaterial::CollisionFilterEntity>(r, e);
        ClearDanglingRefs<&TriggerNodes::CollisionFilterEntity>(r, e);
    }
    s.FilterRef->Update(r);
}

void OnPhysicsJointDefChange(PhysicsState &s, entt::registry &r, entt::entity e) {
    // SixDOFConstraintSettings is baked at Create().
    // No cheap-apply path, so each referencing joint must remove+rebuild its constraint.
    const bool destroyed = !r.valid(e) || !r.all_of<PhysicsJointDef>(e);
    for (auto [je, j] : r.view<const PhysicsJoint>().each()) {
        if (j.JointDefEntity != e) continue;
        if (auto it = s.ConstraintsByJoint.find(je); it != s.ConstraintsByJoint.end()) {
            s.System->RemoveConstraint(it->second);
            s.ConstraintsByJoint.erase(it);
        }
        if (!destroyed) BuildJoint(s, r, je);
    }
    if (destroyed) ClearDanglingRefs<&PhysicsJoint::JointDefEntity>(r, e);
}

void ClearSimulation(PhysicsState &s, entt::registry &r) {
    for (auto &[_, c] : s.ConstraintsByJoint) s.System->RemoveConstraint(c);
    s.ConstraintsByJoint.clear();
    // Every live contact goes with its bodies.
    // Advancing the step alongside the clear publishes the empty set at once.
    r.ctx().get<PhysicsContactImpacts>().Events.clear();
    auto &sustained = r.ctx().get<PhysicsSustainedContacts>();
    sustained.Active.clear();
    sustained.Step = ++s.SustainedStep;
    s.SustainedManifolds.clear();
    r.clear<PhysicsBodyHandle>(); // OnDestroyPhysicsBody queues every body for the batched removal below
    r.clear<BodyPoseCache>();
    FlushPendingBodyRemovals(s);
    s.Baked = {};
    s.FilterRef->Update(r);
    s.FilterRef->Reset();
    s.BodySubGroups.clear();
    s.EntityByBodyIndex.clear();
    s.ContactListener.RawContacts.clear();
    s.ContactListener.RawSustainedContacts.clear();
    s.ContactListener.PendingImpacts.clear();
    s.ContactListener.PendingImpactCount.store(0, std::memory_order_relaxed);
}

void Rebuild(entt::registry &r) {
    auto &s = r.ctx().get<PhysicsState>();
    ClearSimulation(s, r);

    for (auto [e, _] : r.view<const Transform>().each()) {
        const auto *node = r.try_get<const SceneNode>(e);
        if (!node || node->Parent == entt::null) UpdateWorldTransformRecursive(r, e);
    }

    for (auto entity : r.view<PhysicsMotion>()) AddBody(s, r, entity);
    for (auto entity : r.view<ColliderShape>()) AddBody(s, r, entity);

    s.JointsDirty = false;
    for (auto entity : r.view<const PhysicsJoint>()) BuildJoint(s, r, entity);

    s.FilterRef->FinalizeDisabledPairs();
    s.System->OptimizeBroadPhase();
}

// Drops the bake frontier. The cached slots keep their contents and are overwritten on the next bake.
void ClearCache(entt::registry &r) { r.ctx().get<PhysicsState>().Baked = {}; }
bool HasBodies(const entt::registry &r) { return r.ctx().get<PhysicsState>().System->GetNumBodies() > 0; }

// The entity whose body carries this index, or null once the body is gone.
entt::entity EntityForBody(const PhysicsState &s, uint32_t body_index) {
    const auto it = s.EntityByBodyIndex.find(body_index);
    return it != s.EntityByBodyIndex.end() ? it->second : null_entity;
}

// A body's world rotation, which maps a displacement in that body's frame into the world frame contacts are reported in.
// Read from the simulation rather than from WorldTransform, which the step has yet to sync.
quat BodyRotation(const PhysicsState &s, const entt::registry &r, entt::entity entity) {
    const auto *handle = r.valid(entity) ? r.try_get<const PhysicsBodyHandle>(entity) : nullptr;
    if (!handle) return quat{1, 0, 0, 0};
    return FromJolt(s.System->GetBodyInterface().GetRotation(BodyID{handle->BodyId}));
}

// Resolve this step's raw contacts into per-body impacts.
// Both bodies of a pair are struck: body 2 takes the impulse direction, body 1 its negation.
void CollectContactImpacts(PhysicsState &s, entt::registry &r) {
    auto &raw = s.ContactDrainScratch;
    {
        const std::scoped_lock lock{s.ContactListener.ContactMutex};
        raw.swap(s.ContactListener.RawContacts);
    }
    auto &out = r.ctx().get<PhysicsContactImpacts>().Events;
    for (const auto &c : raw) {
        const vec3 dir = FromJolt(c.Direction), point = FromJolt(c.Point), resultant = FromJolt(c.ResultantPoint);
        const auto e1 = EntityForBody(s, c.Body1Index), e2 = EntityForBody(s, c.Body2Index);
        const auto emit = [&](entt::entity self, entt::entity self_collider, entt::entity other, entt::entity other_collider, vec3 d, float other_inv_mass) {
            out.emplace_back(ContactImpact{.Entity = self, .ColliderEntity = self_collider, .Other = other, .OtherColliderEntity = other_collider, .Point = point, .ResultantPoint = resultant, .Direction = d, .Impulse = c.Impulse, .Speed = c.Speed, .OtherInvMass = other_inv_mass, .NominalArea = c.NominalArea});
        };
        emit(e1, c.Collider1, e2, c.Collider2, -dir, c.InvMass2);
        emit(e2, c.Collider2, e1, c.Collider1, dir, c.InvMass1);
    }
    raw.clear();
}

// Resolve this step's persisting contacts into per-body contact states, one contact per touching manifold.
void CollectSustainedContacts(PhysicsState &s, entt::registry &r, float sim_dt) {
    auto &raw = s.SustainedDrainScratch;
    {
        const std::scoped_lock lock{s.ContactListener.ContactMutex};
        raw.swap(s.ContactListener.RawSustainedContacts);
    }
    auto &sustained = r.ctx().get<PhysicsSustainedContacts>();
    sustained.Active.clear();

    // Gather this step's entries by body pair and then by manifold, which is the level a contact is reported at.
    // The collision jobs run in parallel, so a manifold's entries arrive interleaved with other manifolds'.
    auto &order = s.SustainedOrderScratch;
    order.clear();
    order.reserve(raw.size());
    for (uint32_t i = 0; i < raw.size(); ++i) {
        order.emplace_back(BodyPairKey(raw[i].Body1Index, raw[i].Body2Index), raw[i].SubShapeKey, i);
    }
    std::sort(order.begin(), order.end());

    const float inv_dt = sim_dt > 0 ? 1.f / sim_dt : 0.f;

    sustained.Step = ++s.SustainedStep;
    auto &pairs_seen = s.SustainedPairScratch;
    pairs_seen.clear();
    // The gather is sorted, so a pair's entries are one run of it and a manifold's are one run of that.
    static constexpr auto same_pair = [](const auto &a, const auto &b) { return std::get<0>(a) == std::get<0>(b); };
    static constexpr auto same_manifold = [](const auto &a, const auto &b) { return std::get<1>(a) == std::get<1>(b); };
    for (const auto pair_group : order | std::views::chunk_by(same_pair)) {
        const uint64_t pair_key = std::get<0>(pair_group.front());
        pairs_seen.push_back(pair_key);
        auto &tracked = s.SustainedManifolds[pair_key];
        // Body 2 is the second side, the one the normal and the slip are oriented toward.
        const auto [body1_index, body2_index] = BodyPairIndices(pair_key);
        const auto e1 = EntityForBody(s, body1_index), e2 = EntityForBody(s, body2_index);
        const auto rot1 = BodyRotation(s, r, e1), rot2 = BodyRotation(s, r, e2);

        // One contact per manifold, since a pair's manifolds are grouped by contact normal and act on different faces.
        for (const auto manifold_group : pair_group | std::views::chunk_by(same_manifold)) {
            const uint64_t manifold_key = std::get<1>(manifold_group.front());
            ManifoldMerge m;
            for (const auto &entry : manifold_group) {
                const auto &c = raw[std::get<2>(entry)];
                m.Point += c.Point * c.NormalImpulse;
                m.Normal += c.Normal * c.NormalImpulse;
                m.Slip += c.Slip * c.NormalImpulse;
                m.Local1 += c.Local1 * c.NormalImpulse;
                m.Local2 += c.Local2 * c.NormalImpulse;
                m.FrictionImpulse += c.FrictionImpulse;
                m.Impulse += c.NormalImpulse;
            }
            if (m.Impulse <= 0) continue;
            // The colliders and the combined constants belong to the manifold rather than to one of its reports,
            // so they come from the last step to report it.
            const auto &last = raw[std::get<2>(manifold_group.back())];

            // Each surface traverses the contact at its own rate, differenced in that body's frame and rotated into world.
            const vec3 local1 = FromJolt(m.Local1 / m.Impulse), local2 = FromJolt(m.Local2 / m.Impulse);
            auto prev = std::ranges::find(tracked, manifold_key, &PhysicsState::SustainedManifold::Key);
            const bool is_new = prev == tracked.end();
            // A manifold seen for the first time has no previous position to difference, so it reports no travel this step.
            const vec3 sweep1 = is_new ? vec3{0} : rot1 * (local1 - prev->Local1) * inv_dt;
            const vec3 sweep2 = is_new ? vec3{0} : rot2 * (local2 - prev->Local2) * inv_dt;
            const uint64_t id = is_new ? s.NextSustainedId++ : prev->Id;
            if (is_new) prev = tracked.emplace(tracked.end());
            *prev = {manifold_key, id, local1, local2, s.SustainedStep};

            // A manifold's normal is one direction, so this only degenerates if it reversed within the frame.
            const vec3 normal_sum = FromJolt(m.Normal / m.Impulse);
            if (numeric::Length(normal_sum) < 1e-6f) continue;

            sustained.Active.emplace_back(SustainedContact{
                .Id = id,
                .Sides = {
                    SustainedContactSide{.Entity = e1, .ColliderEntity = last.Collider1, .SweepVelocity = sweep1},
                    SustainedContactSide{.Entity = e2, .ColliderEntity = last.Collider2, .SweepVelocity = sweep2},
                },
                .Point = FromJolt(m.Point / m.Impulse),
                .Normal = numeric::Normalize(normal_sum),
                .Slip = FromJolt(m.Slip / m.Impulse),
                .NormalForce = m.Impulse * inv_dt,
                .FrictionForce = FromJolt(m.FrictionImpulse) * inv_dt,
                .NominalArea = last.NominalArea,
                .NominalExtent = last.NominalExtent,
                .Restitution = last.Restitution,
                .Friction = last.Friction,
            });
        }
        // A manifold the step didn't report has stopped touching.
        std::erase_if(tracked, [&](const auto &manifold) { return manifold.LastStep != s.SustainedStep; });
    }
    raw.clear();

    // A pair the step didn't report has stopped touching entirely.
    std::erase_if(s.SustainedManifolds, [&](const auto &pair) { return !std::ranges::binary_search(pairs_seen, pair.first); });
}

// Write a body's world pose to ECS and propagate it to the body's children.
void SyncBodyWorldTransform(entt::registry &r, entt::entity entity, const vec3 &pos, const quat &rot) {
    r.patch<WorldTransform>(entity, [&](WorldTransform &t) {
        t.P = pos;
        t.R = rot;
    });
    for (const auto child : Children{&r, entity}) UpdateWorldTransformRecursive(r, child);
}

// Advance the simulation, marking the bodies whose contacts it reports in detail as it goes.
void StepSimulation(PhysicsState &s, const entt::registry &r, float sim_dt, uint32_t collision_steps) {
    s.ContactListener.SubstepDt = collision_steps > 0 ? sim_dt / float(collision_steps) : sim_dt;
    auto &reporting = s.ContactListener.Reporting;
    reporting.clear();
    for (const auto [e, handle] : r.view<const ReportContacts, const PhysicsBodyHandle>().each()) {
        const auto index = BodyID{handle.BodyId}.GetIndex();
        if (index >= reporting.size()) reporting.resize(index + 1, 0);
        reporting[index] = 1;
    }
    s.System->Update(sim_dt, collision_steps, &s.TempAllocator, &s.JobSystem);
}

// Step the sim one frame, collect contacts, record `frame`'s poses into the cache, and sync active
// bodies to WorldTransform. Advances the bake frontier.
void BakeFrame(entt::registry &r, entt::entity viewport, PhysicsState &s, uint32_t frame, float fps) {
    // Each collision step integrates (dt * TimeScale) / SubstepsPerFrame seconds of sim time.
    const auto &settings = r.get<const PhysicsSimulationSettings>(viewport);
    const float dt = fps > 0 ? 1.f / fps : 1.f / 60.f;
    const float sim_dt = dt * settings.TimeScale;
    StepSimulation(s, r, sim_dt, settings.SubstepsPerFrame);
    CollectContactImpacts(s, r);
    CollectSustainedContacts(s, r, sim_dt);
    // Sync Jolt -> ECS WorldTransform only. PhysicsVelocity is authored-only; live velocity stays in Jolt.
    // WT is owned by the simulator during sim; local Transform is the authored pose, restored on Rebuild.
    const auto &bi = s.System->GetBodyInterface();
    const bool record = frame >= s.CacheStartFrame && frame <= s.CacheEndFrame;
    const auto idx = frame - s.CacheStartFrame;
    for (auto [entity, handle, cache] : r.view<const PhysicsBodyHandle, BodyPoseCache>().each()) {
        const BodyID id{handle.BodyId};
        if (bi.GetMotionType(id) == EMotionType::Static) continue; // No pose change to cache or sync.
        const auto pos = FromJolt(bi.GetPosition(id));
        const auto rot = FromJolt(bi.GetRotation(id));
        if (record) {
            if (cache.Frames.size() <= idx) cache.Frames.resize(idx + 1);
            cache.Frames[idx] = CachedPose{pos, rot};
        }
        if (!bi.IsActive(id)) continue;
        SyncBodyWorldTransform(r, entity, pos, rot);
    }
    if (record) s.Baked = frame;
}

} // namespace

namespace physics {
void ApplySimulationSettings(entt::registry &r, const PhysicsSimulationSettings &settings) {
    auto &s = r.ctx().get<PhysicsState>();
    s.System->SetGravity(ToJolt(settings.Gravity));
    auto system_settings = s.System->GetPhysicsSettings();
    system_settings.mNumVelocitySteps = settings.SolverIterations;
    s.System->SetPhysicsSettings(system_settings);
}

std::optional<uint32_t> BakedThrough(const entt::registry &r) { return r.ctx().get<PhysicsState>().Baked; }
uint32_t BodyCount(const entt::registry &r) { return r.ctx().get<PhysicsState>().System->GetNumBodies(); }

bool DoesFilterAllow(const entt::registry &r, entt::entity source, entt::entity target) {
    const auto &filter = r.ctx().get<PhysicsState>().FilterRef;
    return !filter || filter->DirectionalAllows(source, target);
}

bool AdvancePlayback(entt::registry &r, entt::entity viewport, int from_frame, int to_frame, int range_start_frame, int range_end_frame, float fps, bool range_changed, bool cache_invalid) {
    if (!reactive<changes::PhysicsSimulationSettings>(r).empty()) {
        ApplySimulationSettings(r, r.get<const PhysicsSimulationSettings>(viewport));
    }

    // Range edits invalidate the cache: bake frontier becomes meaningless when bounds shift.
    if (cache_invalid) {
        if (HasBodies(r)) ClearCache(r);
    } else if (range_changed && HasBodies(r)) {
        ClearCache(r);
    }
    if (!HasBodies(r)) return false;

    auto &s = r.ctx().get<PhysicsState>();
    // Keep the cache frame range in sync with the scrub bounds; shifting bounds resets the bake frontier.
    if (uint32_t(range_start_frame) != s.CacheStartFrame || uint32_t(range_end_frame) != s.CacheEndFrame) {
        s.CacheStartFrame = range_start_frame;
        s.CacheEndFrame = range_end_frame;
        s.Baked = {};
    }
    // Any physics-mutating change invalidates cached future frames so the next play-forward re-bakes.
    if (!reactive<changes::PhysicsShape>(r).empty() ||
        !reactive<changes::PhysicsMotion>(r).empty() ||
        !reactive<changes::PhysicsPose>(r).empty() ||
        !reactive<changes::PhysicsMaterial>(r).empty() ||
        !reactive<changes::PhysicsTrigger>(r).empty() ||
        !reactive<changes::PhysicsJoint>(r).empty() ||
        !reactive<changes::PhysicsMaterialDef>(r).empty() ||
        !reactive<changes::CollisionSystemDef>(r).empty() ||
        !reactive<changes::CollisionFilterDef>(r).empty() ||
        !reactive<changes::PhysicsJointDef>(r).empty() ||
        !reactive<changes::PhysicsSimulationSettings>(r).empty()) {
        // Stale slots past the frontier get overwritten on the next bake.
        const uint32_t frame = to_frame;
        s.Baked = (frame == 0 || !s.Baked) ? std::nullopt : std::optional{std::min(*s.Baked, frame - 1)};
    }

    // The cache is the timeline: bake one step past the frontier, restore within it, or reset at the range start.
    if (from_frame == to_frame) return false; // playhead didn't move
    if (to_frame == from_frame + 1 && (!s.Baked || uint32_t(to_frame) > *s.Baked)) {
        BakeFrame(r, viewport, s, uint32_t(to_frame), fps);
    } else if (s.Baked && uint32_t(to_frame) > s.CacheStartFrame && uint32_t(to_frame) <= *s.Baked) {
        // Restore a cached frame without re-simulating.
        const auto idx = uint32_t(to_frame) - s.CacheStartFrame;
        auto &bi = s.System->GetBodyInterface();
        for (auto [entity, handle, cache] : r.view<const PhysicsBodyHandle, const BodyPoseCache>().each()) {
            if (idx >= cache.Frames.size() || !cache.Frames[idx]) continue; // Body not simulated at this frame.
            const auto &p = *cache.Frames[idx];
            // DontActivate: scrub/replay shouldn't wake sleeping bodies. The cache is the timeline, not a sim resume point.
            bi.SetPositionAndRotationWhenChanged(BodyID{handle.BodyId}, ToJolt(p.P), ToJolt(p.R), EActivation::DontActivate);
            SyncBodyWorldTransform(r, entity, p.P, p.R);
        }
    } else if (to_frame == range_start_frame) {
        Rebuild(r); // Reset sim: covers wrap-after-play and user-initiated jump-to-start.
    } else {
        return false; // scrub past the bake frontier — hold current pose until play resumes baking
    }
    return true;
}

void BakeThrough(entt::registry &r, entt::entity viewport, int through_frame, float fps) {
    if (!HasBodies(r)) return;
    auto &s = r.ctx().get<PhysicsState>();
    if (!s.Baked) return; // Bake ahead only from a contiguous frontier, where the live sim matches Baked.
    const uint32_t target = std::min(uint32_t(std::max(through_frame, 1)), s.CacheEndFrame);
    while (*s.Baked < target) BakeFrame(r, viewport, s, *s.Baked + 1, fps);
}

void SamplePosesAtFrame(entt::registry &r, float frame) {
    if (!HasBodies(r)) return;
    auto &s = r.ctx().get<PhysicsState>();
    if (!s.Baked) return;
    // Interpolate cached poses at the (possibly fractional) frame, clamped to the baked range.
    const float clamped = std::clamp(frame, float(s.CacheStartFrame), float(*s.Baked));
    const uint32_t lo = uint32_t(std::floor(clamped));
    const uint32_t hi = std::min(lo + 1, *s.Baked);
    const float t = clamped - float(lo);
    const auto lo_idx = lo - s.CacheStartFrame, hi_idx = hi - s.CacheStartFrame;
    for (auto [entity, handle, cache] : r.view<const PhysicsBodyHandle, const BodyPoseCache>().each()) {
        if (lo_idx >= cache.Frames.size() || !cache.Frames[lo_idx]) continue; // Body not simulated at this frame.
        const auto &a = *cache.Frames[lo_idx];
        const auto &b = hi_idx < cache.Frames.size() && cache.Frames[hi_idx] ? *cache.Frames[hi_idx] : a;
        SyncBodyWorldTransform(r, entity, numeric::Mix(a.P, b.P, t), numeric::Slerp(a.R, b.R, t));
    }
}

void Init(entt::registry &r) {
    auto &state = r.ctx().emplace<PhysicsState>();
    state.ContactListener.R = &r;
    // ResetSystem reinits in place, so the optional's storage address outlives every reset.
    state.ContactListener.System = &*state.System;
    r.ctx().emplace<PhysicsContactImpacts>();
    r.ctx().emplace<PhysicsSustainedContacts>();
    r.on_destroy<PhysicsBodyHandle>().connect<&OnDestroyPhysicsBody>();

    track<changes::PhysicsMotion>(r).on<PhysicsMotion>(On::Create | On::Update | On::Destroy);
    track<changes::PhysicsShape>(r).on<ColliderShape>(On::Create | On::Update | On::Destroy);
    track<changes::PhysicsPose>(r).on<Transform>(On::Update);
    track<changes::ColliderPolicy>(r).on<::ColliderPolicy>(On::Create | On::Update);
    track<changes::PhysicsMaterial>(r).on<ColliderMaterial>(On::Update);
    track<changes::PhysicsTrigger>(r).on<TriggerTag>(On::Create | On::Destroy);
    track<changes::PhysicsJoint>(r).on<PhysicsJoint>(On::Create | On::Update | On::Destroy);
    track<changes::PhysicsMaterialDef>(r).on<::PhysicsMaterial>(On::Create | On::Update | On::Destroy);
    track<changes::CollisionSystemDef>(r).on<CollisionSystem>(On::Create | On::Update | On::Destroy);
    track<changes::CollisionFilterDef>(r).on<CollisionFilter>(On::Create | On::Update | On::Destroy);
    track<changes::PhysicsJointDef>(r).on<::PhysicsJointDef>(On::Create | On::Update | On::Destroy);
    track<changes::PhysicsSimulationSettings>(r).on<::PhysicsSimulationSettings>(On::Update);

    RegisterComponentEventHandler(r, [](entt::registry &r) {
        auto &s = r.ctx().get<PhysicsState>();
        const bool joint_events = !reactive<changes::PhysicsJoint>(r).empty();
        // Resource def handlers run first so dangling-ref patches fire before per-entity handlers this tick.
        for (auto e : reactive<changes::PhysicsMaterialDef>(r)) OnPhysicsMaterialDefChange(s, r, e);
        for (auto e : reactive<changes::CollisionSystemDef>(r)) OnCollisionSystemDefChange(s, r, e);
        for (auto e : reactive<changes::CollisionFilterDef>(r)) OnCollisionFilterDefChange(s, r, e);
        for (auto e : reactive<changes::PhysicsJointDef>(r)) OnPhysicsJointDefChange(s, r, e);
        if (!reactive<changes::PhysicsShape>(r).empty()) RecomputeSceneScale(s, r);
        for (auto e : reactive<changes::PhysicsShape>(r)) OnShapeChange(s, r, e);
        for (auto e : reactive<changes::PhysicsMotion>(r)) OnMotionChange(s, r, e);
        for (auto e : reactive<changes::PhysicsMaterial>(r)) OnMaterialChange(s, r, e);
        for (auto e : reactive<changes::PhysicsTrigger>(r)) OnTriggerChange(s, r, e);
        for (auto e : reactive<changes::PhysicsPose>(r)) OnPoseChange(s, r, e);
        FlushJoints(s, r, joint_events);
        // Destroy bodies queued by OnDestroyPhysicsBody together, after FlushJoints has cleared the
        // constraints that referenced them (Jolt asserts on destroying a body still in a constraint).
        FlushPendingBodyRemovals(s);
    });
}

void Deinit(entt::registry &r) { r.ctx().erase<PhysicsState>(); }

void Clear(entt::registry &r) {
    if (auto *s = r.ctx().find<PhysicsState>()) {
        ClearSimulation(*s, r);
        s->ResetSystem();
    }
}
} // namespace physics
