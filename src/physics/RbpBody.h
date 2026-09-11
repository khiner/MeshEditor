#pragma once

#include "PhysicsTypes.h"
#include "gpu/Shared.h"

#include <span>

namespace rbp {
struct World;
}

struct Transform;

namespace physics {

inline rbp::float3 ToRbp(vec3 v) { return {v.x, v.y, v.z}; }
inline rbp::float4 ToRbp(quat q) { return {q.x, q.y, q.z, q.w}; }
inline vec3 FromRbp(rbp::float3 v) { return {v.x, v.y, v.z}; }
inline quat FromRbp(rbp::float4 q) { return {q.w, q.x, q.y, q.z}; }
inline rbp::Material ToRbp(const PhysicsMaterial &m) {
    return {.StaticFriction = m.StaticFriction, .DynamicFriction = m.DynamicFriction, .Restitution = m.Restitution, .FrictionCombine = uint32_t(m.FrictionCombine), .RestitutionCombine = uint32_t(m.RestitutionCombine)};
}

inline rbp::Pose Inverse(rbp::Pose pose) {
    const auto q = rbp::QuatConjugate(pose.Orientation);
    return {rbp::Rotate(q, -pose.Position), q};
}

struct RbpBody {
    rbp::Index Body = rbp::NoIndex, Shape = rbp::NoIndex;
    // Centre-of-mass and principal-axis frame in the node's rigid frame, in world metres.
    rbp::Pose Frame = rbp::IdentityPose;
    rbp::Pose CookedFrame = rbp::IdentityPose, InitialPose = rbp::IdentityPose;
    rbp::BodyMass NaturalMass{};
    rbp::Velocity InitialVelocity{};
};

// Copy scaled colliders from the node's rigid frame into an owned compound, retaining caller ownership of input shapes.
// Replace previous's owned geometry when provided and preserve its body identity.
// Restore initial states and reset dynamics before advancing a replacement.
// Throw on invalid mass properties or exhausted pools and release partial allocations.
RbpBody BuildRbpBody(rbp::World &, std::span<const rbp::Index> colliders, const Transform &node, const PhysicsMotion *motion, const PhysicsVelocity *velocity = nullptr, bool sensor = false, const RbpBody *previous = nullptr);

// Update authored properties without recooking geometry; restore initial states and reset dynamics before advancing.
void UpdateRbpBody(rbp::World &, RbpBody &, const Transform &, const PhysicsMotion *, const PhysicsVelocity *);

CachedPose RbpNodePose(rbp::Pose body_pose, rbp::Pose frame);

} // namespace physics
