#pragma once

#include "entt_fwd.h"
#include "numeric/quat.h"
#include "numeric/vec3.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

// KHR_physics_rigid_bodies-aligned ECS component structs.
// No Jolt types — these are pure data, serializable to/from glTF.

// --- Document-level resources (stored in PhysicsWorld) ---

enum class PhysicsCombineMode : uint8_t {
    Average,
    Minimum,
    Maximum,
    Multiply,
};

struct PhysicsMaterial {
    float StaticFriction{0.6f};
    float DynamicFriction{0.6f};
    float Restitution{0.0f};
    PhysicsCombineMode FrictionCombine{PhysicsCombineMode::Average};
    PhysicsCombineMode RestitutionCombine{PhysicsCombineMode::Average};
    std::string Name{};
};

struct CollisionFilter {
    std::vector<std::string> CollisionSystems{};
    std::vector<std::string> CollideWithSystems{};
    std::vector<std::string> NotCollideWithSystems{};
    std::string Name{};
};

struct PhysicsJointLimit {
    std::vector<uint8_t> LinearAxes{};
    std::vector<uint8_t> AngularAxes{};
    std::optional<float> Min{};
    std::optional<float> Max{};
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
    float PositionTarget{0.0f};
    float VelocityTarget{0.0f};
    float Stiffness{0.0f};
    float Damping{0.0f};
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

// --- Per-node ECS components ---

struct PhysicsMotion {
    bool IsKinematic{false};
    std::optional<float> Mass{}; // nullopt = auto-compute from shape
    vec3 CenterOfMass{0.0f};
    std::optional<vec3> InertiaDiagonal{};
    std::optional<quat> InertiaOrientation{};
    vec3 LinearVelocity{0.0f};
    vec3 AngularVelocity{0.0f};
    float GravityFactor{1.0f};
};

struct PhysicsCollider {
    PhysicsShape Shape{};
    std::optional<uint32_t> PhysicsMaterialIndex{};
    std::optional<uint32_t> CollisionFilterIndex{};
};

struct PhysicsTrigger {
    std::optional<PhysicsShape> Shape{};
    std::vector<entt::entity> Nodes{};
    std::optional<uint32_t> CollisionFilterIndex{};
};

struct PhysicsJoint {
    entt::entity ConnectedNode{null_entity};
    uint32_t JointDefIndex{0};
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
