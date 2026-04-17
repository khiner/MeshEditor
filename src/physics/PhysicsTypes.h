#pragma once

#include "entt_fwd.h"
#include "numeric/quat.h"
#include "numeric/vec3.h"

#include <optional>
#include <string>
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

struct CollisionFilter {
    std::vector<std::string> CollisionSystems{}, CollideWithSystems{}, NotCollideWithSystems{};
    std::string Name{};
};

struct PhysicsJointLimit {
    std::vector<uint8_t> LinearAxes{}, AngularAxes{};
    std::optional<float> Min{}, Max{};
    std::optional<float> Stiffness{}; // nullopt = hard (infinite stiffness) limit
    float Damping{0.0f};
};

enum class PhysicsDriveType : uint8_t { Linear,
                                        Angular };
enum class PhysicsDriveMode : uint8_t { Force,
                                        Acceleration };

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

enum class PhysicsShapeType : uint8_t {
    Box,
    Sphere,
    Capsule,
    Cylinder,
    ConvexHull,
    TriangleMesh,
};

struct PhysicsShape {
    PhysicsShapeType Type{PhysicsShapeType::Box};
    vec3 Size{1.0f, 1.0f, 1.0f}; // Box full size (not half-extents) per KHR spec
    float Radius{0.5f}; // Sphere, Capsule, Cylinder
    float RadiusTop{0.25f}; // Capsule, Cylinder (tapered)
    float RadiusBottom{0.25f}; // Capsule, Cylinder (tapered)
    float Height{0.5f}; // Capsule, Cylinder
    // For mesh-based shapes
    std::optional<entt::entity> MeshEntity{};
    bool ConvexHull{false};
};

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
struct ColliderShape {
    PhysicsShape Shape{};
};

// Material and collision filter assignment for a collider.
struct ColliderMaterial {
    entt::entity PhysicsMaterialEntity{null_entity};
    entt::entity CollisionFilterEntity{null_entity};
};

struct PhysicsTrigger {
    std::optional<PhysicsShape> Shape{};
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
// Capsules use 3 instances (body + 2 caps); others use 1.
struct ColliderWireframe {
    entt::entity Instances[3]{null_entity, null_entity, null_entity};
    uint8_t Count{0};
};
