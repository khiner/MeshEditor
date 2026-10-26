#pragma once
#include <variant>

#include "Field.h"
#include "numeric/quat.h"
#include "numeric/vec3.h"
#include "physics/PhysicsSchema.h"
#include "state/Entity.h"

#include <vector>

using numeric::quat, numeric::vec3;

struct PhysicsSimulationSettings {
    vec3 Gravity{0.0f, -9.81f, 0.0f};
    uint32_t SubstepsPerFrame{10}, SolverIterations{10};
    float TimeScale{1.0f}; // Multiplier on simulated dt (1 = real-time)
    bool operator==(const PhysicsSimulationSettings &) const = default;
};
template<> inline constexpr field::SpecTable<PhysicsSimulationSettings> Specs<PhysicsSimulationSettings>{
    {&PhysicsSimulationSettings::SubstepsPerFrame, {.Min = 1, .Max = 100}},
    {&PhysicsSimulationSettings::SolverIterations, {.Min = 2, .Max = 50}},
    {&PhysicsSimulationSettings::TimeScale, {.Min = 0, .Max = 10, .Unit = FieldUnit::Factor}},
    {&PhysicsSimulationSettings::Gravity, {.Speed = 0.1f}},
};

// KHR_physics_rigid_bodies-aligned component structs.

// --- Document-level resources ---

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
    bool operator==(const PhysicsMaterial &) const = default;
};
template<> inline constexpr field::SpecTable<PhysicsMaterial> Specs<PhysicsMaterial>{
    {&PhysicsMaterial::StaticFriction, {.Min = 0, .Max = 2}},
    {&PhysicsMaterial::DynamicFriction, {.Min = 0, .Max = 2}},
    {&PhysicsMaterial::Restitution, {.Min = 0, .Max = 1}},
};

// Defines a document-level collision system assigned a collision mask bit at world creation.
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
    std::vector<state::Entity> Systems{};
    CollideMode Mode{CollideMode::All};
    std::vector<state::Entity> CollideSystems{};
    std::string Name{};
};

struct PhysicsJointLimit {
    std::vector<uint8_t> LinearAxes{}, AngularAxes{};
    std::optional<float> Min{}, Max{};
    std::optional<float> Stiffness{}; // nullopt = hard (infinite stiffness) limit
    float Damping{0.0f};
    bool operator==(const PhysicsJointLimit &) const = default;
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
    bool operator==(const PhysicsJointDrive &) const = default;
};

struct PhysicsJointDef {
    std::vector<PhysicsJointLimit> Limits{};
    std::vector<PhysicsJointDrive> Drives{};
    std::string Name{};
    bool operator==(const PhysicsJointDef &) const = default;
};

// --- Per-shape ---

namespace physics {
// KHR_implicit_shapes requires capsule/cylinder height strictly > 0.
inline constexpr float MinShapeHeight{1e-6f};

// KHR_physics_rigid_bodies / KHR_implicit_shapes-aligned shape primitives.
struct Box {
    vec3 Size{1.f}; // full size (not half-extents) per KHR spec
    bool operator==(const Box &) const = default;
};
struct Sphere {
    float Radius{0.5f};
    bool operator==(const Sphere &) const = default;
};
struct Capsule {
    float Height{0.5f}, RadiusTop{0.25f}, RadiusBottom{0.25f};
    bool operator==(const Capsule &) const = default;
};
struct Cylinder {
    float Height{0.5f}, RadiusTop{0.25f}, RadiusBottom{0.25f};
    bool operator==(const Cylinder &) const = default;
};
// Plane lies in the XZ plane with +Y normal.
// An absent size means infinite along that axis.
struct Plane {
    std::optional<float> SizeX{}, SizeZ{};
    bool DoubleSided{false};
    bool operator==(const Plane &) const = default;
};
// Mesh-backed shape kinds.
// Mesh reference lives on the ColliderShape wrapper.
struct ConvexHull {
    bool operator==(const ConvexHull &) const = default;
};
struct TriangleMesh {
    bool operator==(const TriangleMesh &) const = default;
};
} // namespace physics

namespace physics {
constexpr FieldSpec SizeSpec(double min, const char *help) { return {.Min = min, .Max = 100, .Speed = 0.01f, .Unit = FieldUnit::Meters, .Widget = FieldWidget::Drag, .Help = help}; }
} // namespace physics
template<> inline constexpr field::SpecTable<physics::Box> Specs<physics::Box>{{&physics::Box::Size, physics::SizeSpec(0.01, "Full extent along each local axis.")}};
template<> inline constexpr field::SpecTable<physics::Sphere> Specs<physics::Sphere>{{&physics::Sphere::Radius, physics::SizeSpec(0.001, "Sphere radius.")}};
template<> inline constexpr field::SpecTable<physics::Capsule> Specs<physics::Capsule>{
    {&physics::Capsule::Height, physics::SizeSpec(0.001, "Distance between the centers of the two end spheres along local Y.")},
    {&physics::Capsule::RadiusTop, physics::SizeSpec(0.001, "Radius of the end sphere at +Y.")},
    {&physics::Capsule::RadiusBottom, physics::SizeSpec(0.001, "Radius of the end sphere at -Y.")},
};
template<> inline constexpr field::SpecTable<physics::Cylinder> Specs<physics::Cylinder>{
    {&physics::Cylinder::Height, physics::SizeSpec(0.001, "Length along local Y.")},
    {&physics::Cylinder::RadiusTop, physics::SizeSpec(0.001, "Radius of the cap at +Y.")},
    {&physics::Cylinder::RadiusBottom, physics::SizeSpec(0.001, "Radius of the cap at -Y.")},
};
template<> inline constexpr field::SpecTable<physics::Plane> Specs<physics::Plane>{
    {&physics::Plane::SizeX, {.Min = 0.01, .Max = 1000, .Initial = 2, .Speed = 0.01f, .Unit = FieldUnit::Meters, .Widget = FieldWidget::Drag, .Help = "Full extent along local X. Absent means the plane is infinite along X."}},
    {&physics::Plane::SizeZ, {.Min = 0.01, .Max = 1000, .Initial = 2, .Speed = 0.01f, .Unit = FieldUnit::Meters, .Widget = FieldWidget::Drag, .Help = "Full extent along local Z. Absent means the plane is infinite along Z."}},
    {&physics::Plane::DoubleSided, {.Help = "Collide from both sides of the plane instead of only from the +Y side"}},
};

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

// Engine default mass (kg) when PhysicsMotion::Mass is unset.
inline constexpr float DefaultMass{1};

// Motion states (KHR_physics_rigid_bodies terms):
// Static: Only ColliderShape (~Blender Passive, unanimated).
// Dynamic: PhysicsMotion present.
// IsKinematic toggles infinite mass:
//   false: force-driven (~Blender Active)
//   true:  velocity-constant, animation-driven (~Blender Passive, animated)
struct PhysicsMotion {
    bool IsKinematic{false};
    std::optional<float> Mass{}; // nullopt = DefaultMass
    std::optional<vec3> CenterOfMass{}; // nullopt = auto-compute from shape
    std::optional<vec3> InertiaDiagonal{};
    quat InertiaOrientation{1, 0, 0, 0};
    float GravityFactor{1.};
    // Engine-specific (not in KHR_physics_rigid_bodies).
    // Defaults match Blender.
    float LinearDamping{0.04f}, AngularDamping{0.1f};
    bool operator==(const PhysicsMotion &) const = default;
};
template<> inline constexpr field::SpecTable<PhysicsMotion> Specs<PhysicsMotion>{
    {&PhysicsMotion::IsKinematic, {.Help = "Move the body by its velocity and animation with infinite mass instead of by forces and contacts"}},
    {&PhysicsMotion::Mass, {.Min = 0.001, .Max = 1e6, .Initial = DefaultMass, .Speed = 0.1f, .Unit = FieldUnit::Kilograms, .Widget = FieldWidget::Drag, .Help = "Body mass. Absent uses the default mass of 1 kg."}},
    {&PhysicsMotion::CenterOfMass, {.Speed = 0.01f, .Unit = FieldUnit::Meters, .Help = "Center of mass in local coordinates. Absent computes it from the collider shape."}},
    {&PhysicsMotion::InertiaDiagonal, {.Min = 0.001, .Max = 1e6, .Initial = 1, .Speed = 0.01f, .Widget = FieldWidget::Drag, .Help = "Principal moments of inertia. Absent computes them from the collider shape and mass."}},
    {&PhysicsMotion::InertiaOrientation, {.Help = "Rotation of the principal inertia axes from the body axes"}},
    {&PhysicsMotion::GravityFactor, {.Min = -10, .Max = 10, .Speed = 0.01f, .Widget = FieldWidget::Drag, .Help = "Multiplier on gravity for this body"}},
    {&PhysicsMotion::LinearDamping, {.Min = 0, .Max = 1, .Speed = 0.01f, .Widget = FieldWidget::Drag, .Help = "Fraction of linear velocity removed per second"}},
    {&PhysicsMotion::AngularDamping, {.Min = 0, .Max = 1, .Speed = 0.01f, .Widget = FieldWidget::Drag, .Help = "Fraction of angular velocity removed per second"}},
};

// Defines a force-driven positive-mass body whose mass properties control contact dynamics.
inline bool IsAuthoritativeDynamicBody(const PhysicsMotion &m) { return !m.IsKinematic && m.Mass.value_or(DefaultMass) > 0; }

// Present iff PhysicsMotion is present.
struct PhysicsVelocity {
    vec3 Linear{0}, Angular{0};
    bool operator==(const PhysicsVelocity &) const = default;
};
template<> inline constexpr field::SpecTable<PhysicsVelocity> Specs<PhysicsVelocity>{
    {&PhysicsVelocity::Linear, {.Speed = 0.1f, .Unit = FieldUnit::MetersPerSecond, .Label = "Linear velocity", .Help = "Initial linear velocity in the node's frame"}},
    {&PhysicsVelocity::Angular, {.Speed = 0.1f, .Unit = FieldUnit::RadiansPerSecond, .Label = "Angular velocity", .Help = "Initial angular velocity in the node's frame"}},
};

// MeshEntity present for ConvexHull/TriangleMesh, null for primitives.
// LocalOffset is the shape's center in entity-local pre-scale coords (same convention as PhysicsMotion::CenterOfMass).
struct ColliderShape {
    PhysicsShape Shape{};
    state::Entity MeshEntity{state::Null};
    vec3 LocalOffset{0};
    bool operator==(const ColliderShape &) const = default;
};

struct ColliderMaterial {
    state::Entity PhysicsMaterialEntity{state::Null}, CollisionFilterEntity{state::Null};
};

// Marker: this entity's ColliderShape participates as a sensor (KHR GeometryTrigger), not a solid body.
// CollisionFilterEntity for shape-flavor triggers lives on the shared ColliderMaterial (material slot unused).
struct TriggerTag {};

// Engine-only collider derivation policy, not serialized to glTF.
// AutoFitDims: refit ColliderShape dimensions to the mesh AABB on mesh changes.
// LockedKind: user picked the variant from the dropdown — engine never auto-changes the kind.
struct ColliderPolicy {
    bool AutoFitDims{true}, LockedKind{false};
};

// Compound trigger (KHR NodesTrigger): zone defined by child nodes — no own shape.
// Listed nodes supply the geometry; engine reports entry/exit for any of them.
// Does not produce a rigid body; exists for document structure and filter assignment.
struct TriggerNodes {
    std::vector<state::Entity> Nodes{};
    state::Entity CollisionFilterEntity{state::Null};
};

struct PhysicsJoint {
    state::Entity ConnectedNode{state::Null};
    state::Entity JointDefEntity{state::Null};
    bool EnableCollision{false};
    bool operator==(const PhysicsJoint &) const = default;
};

// --- Internal (not serialized to glTF) ---

struct PhysicsBodyHandle {
    uint32_t BodyId{UINT32_MAX};
};
struct PhysicsConstraintHandle {
    uint32_t ConstraintIndex{UINT32_MAX};
};

struct CachedPose {
    vec3 P;
    quat R;
};

// Contiguous poses beginning at the simulation cache start frame.
struct BodyPoseCache {
    std::vector<CachedPose> Frames;
};
