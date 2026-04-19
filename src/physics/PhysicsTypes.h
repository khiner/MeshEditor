#pragma once

#include "entt_fwd.h"
#include "numeric/quat.h"
#include "numeric/vec3.h"

#include <optional>
#include <string>
#include <variant>
#include <vector>

// KHR_physics_rigid_bodies-aligned component structs.

// --- Document-level resources (stored in PhysicsWorld) ---

enum class PhysicsCombineMode : uint8_t {
    Average,
    Minimum,
    Maximum,
    Multiply,
};

struct PhysicsMaterial {
    float StaticFriction{0.6f}, DynamicFriction{0.6f}, Restitution{0.0f};
    PhysicsCombineMode FrictionCombine{PhysicsCombineMode::Average}, RestitutionCombine{PhysicsCombineMode::Average};
    std::string Name{};
};

// Named collision system. Document-level resource; each entity gets a bit position
// (assigned by iteration order over view<CollisionSystem>) in the Jolt collision mask.
struct CollisionSystem {
    std::string Name{};
};

enum class CollideMode : uint8_t {
    All,
    Allowlist,
    Blocklist
};

// KHR_physics_rigid_bodies-aligned collision filter.
// Mode == All: no filtering beyond membership (CollideSystems ignored at rebuild, but preserved across mode toggles).
// Mode == Allowlist: collides only with bodies whose membership intersects CollideSystems.
// Mode == Blocklist: blocks collisions with bodies whose membership intersects CollideSystems.
// Type enforces the KHR schema's mutual-exclusion of collideWithSystems / notCollideWithSystems.
struct CollisionFilter {
    std::vector<entt::entity> Systems{};
    CollideMode Mode{CollideMode::All};
    std::vector<entt::entity> CollideSystems{};
    std::string Name{};
};

struct PhysicsJointLimit {
    std::vector<uint8_t> LinearAxes{}, AngularAxes{};
    std::optional<float> Min{}, Max{};
    std::optional<float> Stiffness{}; // nullopt = hard (infinite stiffness) limit
    float Damping{0.0f};
};

enum class PhysicsDriveType : uint8_t {
    Linear,
    Angular
};
enum class PhysicsDriveMode : uint8_t {
    Force,
    Acceleration
};

struct PhysicsJointDrive {
    PhysicsDriveType Type{PhysicsDriveType::Linear};
    PhysicsDriveMode Mode{PhysicsDriveMode::Force};
    uint8_t Axis{0}; // 0=X, 1=Y, 2=Z
    float MaxForce{std::numeric_limits<float>::max()};
    float PositionTarget{0}, VelocityTarget{0};
    float Stiffness{0}, Damping{0};
};

struct PhysicsJointDef {
    std::vector<PhysicsJointLimit> Limits{};
    std::vector<PhysicsJointDrive> Drives{};
    std::string Name{};
};

// --- Per-shape ---

namespace physics {
// KHR_physics_rigid_bodies / KHR_implicit_shapes-aligned shape primitives.
struct Box {
    vec3 Size{1.0f, 1.0f, 1.0f}; // full size (not half-extents) per KHR spec
};
struct Sphere {
    float Radius{0.5f};
};
struct Capsule {
    float Height{0.5f};
    float RadiusTop{0.25f};
    float RadiusBottom{0.25f};
};
struct Cylinder {
    float Height{0.5f};
    float RadiusTop{0.25f};
    float RadiusBottom{0.25f};
};
// Plane lies in the XZ plane with +Y normal. Size{X,Z} == 0 means infinite along that axis.
struct Plane {
    float SizeX{0.f}, SizeZ{0.f};
    bool DoubleSided{false};
};
// Mesh-backed shape kinds. Mesh reference lives on the ColliderShape wrapper.
struct ConvexHull {};
struct TriangleMesh {};
} // namespace physics

using PhysicsShape = std::variant<
    physics::Box,
    physics::Sphere,
    physics::Capsule,
    physics::Cylinder,
    physics::Plane,
    physics::ConvexHull,
    physics::TriangleMesh>;

inline bool IsMeshBackedShape(const PhysicsShape &shape) {
    return std::holds_alternative<physics::ConvexHull>(shape) || std::holds_alternative<physics::TriangleMesh>(shape);
}

// --- Per-node ---

// Engine default mass (kg) when PhysicsMotion::Mass is unset. Matches Blender.
inline constexpr float DefaultMass = 1.0f;

// Motion states (KHR_physics_rigid_bodies terms):
// Static: Only ColliderShape (~Blender Passive, unanimated).
// Dynamic: PhysicsMotion present. IsKinematic toggles infinite mass:
//   false: force-driven (~Blender Active)
//   true:  velocity-constant, animation-driven (~Blender Passive, animated)
struct PhysicsMotion {
    bool IsKinematic{false};
    std::optional<float> Mass{}; // nullopt = DefaultMass
    std::optional<vec3> CenterOfMass{}; // nullopt = auto-compute from shape
    std::optional<vec3> InertiaDiagonal{};
    std::optional<quat> InertiaOrientation{};
    float GravityFactor{1.};

    // Engine-specific (not in KHR_physics_rigid_bodies). Defaults match Blender.
    float LinearDamping{0.04f}, AngularDamping{0.1f};
};

// Linear and angular velocity of a dynamic body. Present iff PhysicsMotion is present.
struct PhysicsVelocity {
    vec3 Linear{0}, Angular{0};
};

// Geometry of the collision volume.
// MeshEntity references the mesh supplying geometry for ConvexHull / TriangleMesh shapes, or null_entity for primitives.
struct ColliderShape {
    PhysicsShape Shape{};
    entt::entity MeshEntity{null_entity};
};

// Material and collision filter assignment for a collider.
struct ColliderMaterial {
    entt::entity PhysicsMaterialEntity{null_entity};
    entt::entity CollisionFilterEntity{null_entity};
};

// Marker: this entity's ColliderShape participates as a sensor (KHR GeometryTrigger), not a solid body.
// CollisionFilterEntity for shape-flavor triggers lives on the shared ColliderMaterial (material slot unused).
struct TriggerTag {};

// Engine-only collider derivation policy, not serialized to glTF.
// AutoFitDims: refit ColliderShape dimensions to the mesh BBox on mesh changes.
// LockedKind: user picked the variant from the dropdown — engine never auto-changes the kind.
struct ColliderPolicy {
    bool AutoFitDims{true};
    bool LockedKind{false};
};

// Compound trigger (KHR NodesTrigger): zone defined by child nodes — no own shape.
// Listed nodes supply the geometry; engine reports entry/exit for any of them.
// Does not produce a Jolt body; exists for document structure and filter assignment.
struct TriggerNodes {
    std::vector<entt::entity> Nodes{};
    entt::entity CollisionFilterEntity{null_entity};
};

struct PhysicsJoint {
    entt::entity ConnectedNode{null_entity};
    entt::entity JointDefEntity{null_entity};
    bool EnableCollision{false};
};

// --- Internal (not serialized to glTF) ---

struct PhysicsBodyHandle {
    uint32_t BodyId{UINT32_MAX};
};

struct PhysicsConstraintHandle {
    uint32_t ConstraintIndex{UINT32_MAX};
};

// Links a collider entity to its wireframe overlay instance(s).
// Cylinder/Capsule use 6 instances (2 ring/cap + 4 side lines); Box/Sphere use 1.
struct ColliderWireframe {
    entt::entity Instances[6]{null_entity, null_entity, null_entity, null_entity, null_entity, null_entity};
    uint8_t Count{0};
};
