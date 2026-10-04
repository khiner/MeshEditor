#include "PhysicsSystem.h"
#include "PhysicsContact.h"
#include "ColliderUpdate.h"
#include "Profile.h"
#include "RbpBody.h"
#include "RbpShape.h"
#include "Solver.h"
#include "SortUnique.h"
#include "TransformMath.h"
#include "mesh/Mesh.h"
#include "metal/MetalContext.h"
#include "numeric/VectorMath.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "scene/WorldTransform.h"
#include "state/Scene.h"
#include "viewport/ViewportEvents.h"

#include <algorithm>
#include <array>
#include <bit>
#include <map>
#include <ranges>
#include <set>
#include <stdexcept>

using state::Change;
using state::On;

using physics::FromRbp;
using physics::Inverse;
using physics::ToRbp;

namespace {
using ContactKey = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint64_t, uint32_t, uint32_t>;
ContactKey Key(const rbp::ContactChange &c) { return {c.A.Slot, c.A.Spawn, c.B.Slot, c.B.Spawn, c.Children, c.SubShapeA, c.SubShape}; }

struct TrackedContact {
    uint64_t Id{}, Seen{};
    bool PendingImpact = true, HasPreviousFrame = false;
    rbp::float3 PreviousA{}, PreviousB{};
};
struct ContactSum {
    rbp::ContactChange Last{};
    rbp::float3 Point{}, Normal{}, Slip{}, LocalA{}, LocalB{}, FrictionImpulse{};
    float NormalImpulse = 0;
};
// A body's authored inputs: its node's world transform, its parent, its motion and the colliders its compound holds.
struct BodyInput {
    Transform Node;
    state::Entity Parent = state::Null;
    std::optional<PhysicsMotion> Motion;
    PhysicsVelocity Velocity;
    bool Sensor = false;
    std::vector<state::Entity> Colliders;
    bool operator==(const BodyInput &) const = default;
};
// A collider leaf in its owner's rigid frame, with the surface and filter it collides with.
struct ColliderInput {
    ColliderShape Shape;
    Transform Local{};
    PhysicsMaterial Material{};
    uint32_t Layer = ~0u, Collides = ~0u;
    bool HasFilter = false;
    bool operator==(const ColliderInput &) const = default;
};
// A body in the world with the inputs it was cooked from, one leaf per collider.
struct BodyRecord {
    physics::RbpBody Cooked;
    BodyInput Input;
    std::vector<ColliderInput> Leaves;
};
struct JointInput {
    PhysicsJoint Joint;
    std::optional<PhysicsJointDef> Definition{};
    state::Entity Owner = state::Null, ConnectedOwner = state::Null;
    Transform Node;
    std::optional<Transform> Connected;
    bool operator==(const JointInput &) const = default;
};
struct PhysicsState {
    // The solver runs on Metal 4 devices only, so this stays empty on other devices and no world is built.
    std::optional<rbp::mtl::Context> Context;
    std::optional<rbp::Solver> Solver;
    std::optional<rbp::World> World;
    rbp::StepSettings Settings;
    std::map<state::Entity, BodyRecord> Bodies;
    // The body whose compound holds each collider.
    std::unordered_map<state::Entity, state::Entity> LeafOwners;
    std::map<state::Entity, JointInput> Joints;
    std::set<state::Entity> JointUpdates;
    PhysicsSimulationSettings AppliedSettings;
    float CacheFps = 0;
    bool CacheInvalid = false;
    std::vector<state::Entity> Entities;
    // Entities whose motion, colliders or joints were destroyed with them, which the next ProcessChanges resolves to their bodies.
    std::vector<state::Entity> Destroyed;
    // Posed bodies with parents before children, rebuilt after input or body set changes.
    std::optional<std::vector<state::Entity>> SampleOrder;
    std::map<state::Entity, rbp::CollisionMask> Masks;
    std::vector<rbp::SensorFollower> SensorFollowers;
    rbp::Index WorldAnchor = rbp::NoIndex;
    uint32_t CacheStartFrame{1}, CacheEndFrame{0};
    std::optional<uint32_t> Baked;
    struct CachedContacts {
        PhysicsContactImpacts Impacts;
        PhysicsSustainedContacts Sustained;
    };
    std::vector<CachedContacts> ContactFrames;
    std::map<ContactKey, TrackedContact> Contacts;
    uint64_t NextContactId{1}, ContactStep{0}, Substep{0};
    bool Clearing = false;

    // The next playback advance restarts the simulation from its authored state.
    void Invalidate() { Baked.reset(); }
};

rbp::Pose PoseOf(const Transform &t) { return rbp::At(ToRbp(t.P), ToRbp(Normalize(t.R))); }
state::Entity MotionOwner(const state::Scene &r, state::Entity e) {
    return FindAncestorIf(r, e, [&](auto node) { return r.all_of<PhysicsMotion>(node); });
}

void UpdateMasks(PhysicsState &s, const state::Scene &r) {
    std::map<state::Entity, uint32_t> bits;
    for (const auto e : SortedEntities(r.view<const CollisionSystem>())) {
        if (bits.size() == 32) throw std::runtime_error("RBP supports at most 32 collision systems.");
        bits.emplace(e, uint32_t(1) << bits.size());
    }
    const auto mask = [&](const auto &systems) {
        uint32_t result = 0;
        for (auto e : systems)
            if (auto it = bits.find(e); it != bits.end()) result |= it->second;
        return result;
    };
    s.Masks.clear();
    for (auto [e, filter] : r.view<const CollisionFilter>().each()) {
        const auto collide = filter.Mode == CollideMode::All ? ~0u : filter.Mode == CollideMode::Allowlist ? mask(filter.CollideSystems) :
                                                                                                             ~mask(filter.CollideSystems);
        s.Masks.emplace(e, rbp::CollisionMask{mask(filter.Systems), collide});
    }
}

Transform ComposeAuthored(const Transform &parent, Transform result) {
    // Preserve exact scale under rigid edits instead of remeasuring quaternion matrix columns.
    if (parent.S.x == parent.S.y && parent.S.y == parent.S.z) {
        result.P *= parent.S;
        result.S *= parent.S;
    } else result = ToTransform(ToMatrix(Transform{.S = parent.S}) * ToMatrix(result));
    result.P = parent.P + Rotate(parent.R, result.P);
    result.R = Normalize(parent.R * result.R);
    return result;
}

// The node's authored world transform, composed from the local transforms of its ancestors.
Transform AuthoredWorld(const state::Scene &r, state::Entity e) {
    const auto *local = r.try_get<const Transform>(e);
    if (!local) return {};
    const auto parent = ParentOrNull(r, e);
    return parent != state::Null ? ComposeAuthored(AuthoredWorld(r, parent), *local) : *local;
}

// The body an entity forms, with a leaf per collider, or none for an entity that forms no body.
// A motion holder forms one, as does a collider that is a sensor or has no motion owner.
std::optional<BodyInput> ReadBody(const PhysicsState &s, const state::Scene &r, state::Entity entity, std::vector<ColliderInput> &leaves) {
    leaves.clear();
    if (!r.valid(entity)) return {};
    const auto *motion = r.try_get<const PhysicsMotion>(entity);
    const bool sensor = r.all_of<TriggerTag, ColliderShape>(entity);
    if (!motion && !(r.all_of<ColliderShape>(entity) && (sensor || MotionOwner(r, entity) == state::Null))) return {};
    BodyInput body{.Node = AuthoredWorld(r, entity), .Parent = ParentOrNull(r, entity), .Sensor = sensor};
    if (motion) {
        body.Motion = *motion;
        if (const auto *velocity = r.try_get<const PhysicsVelocity>(entity)) body.Velocity = *velocity;
    }
    if (r.all_of<ColliderShape>(entity)) body.Colliders.push_back(entity);
    // A solid motion body's compound holds the solid colliders below it, down to the next motion body.
    if (motion && !sensor) {
        const auto gather = [&](this const auto &self, state::Entity node) -> void {
            for (const auto child : Children{&r, node}) {
                if (r.all_of<PhysicsMotion>(child)) continue;
                if (r.all_of<ColliderShape>(child) && !r.all_of<TriggerTag>(child)) body.Colliders.push_back(child);
                self(child);
            }
        };
        gather(entity);
    }
    const auto local_transform = [&](this const auto &self, state::Entity node) -> Transform {
        if (node == entity) return {.S = body.Node.S};
        return ComposeAuthored(self(ParentOrNull(r, node)), r.get<const Transform>(node));
    };
    leaves.reserve(body.Colliders.size());
    for (const auto collider : body.Colliders) {
        auto &leaf = leaves.emplace_back(ColliderInput{.Shape = r.get<const ColliderShape>(collider), .Local = local_transform(collider)});
        if (const auto *material = r.try_get<const ColliderMaterial>(collider)) {
            if (const auto *definition = r.try_get<const PhysicsMaterial>(material->PhysicsMaterialEntity)) leaf.Material = *definition;
            if (const auto it = s.Masks.find(material->CollisionFilterEntity); it != s.Masks.end()) {
                leaf.Layer = it->second.Layer;
                leaf.Collides = it->second.Collides;
                leaf.HasFilter = true;
            }
        }
        leaf.Material.Name.clear();
    }
    return body;
}

// A joint's authored inputs, with the bodies holding its node and its connected node.
// A dangling definition leaves the joint inactive.
JointInput ReadJoint(const PhysicsState &s, const state::Scene &r, state::Entity entity, const PhysicsJoint &joint) {
    const auto owner = [&](state::Entity node) { return FindAncestorIf(r, node, [&](auto ancestor) { return s.Bodies.contains(ancestor); }); };
    JointInput input{
        .Joint = joint,
        .Owner = owner(entity),
        .ConnectedOwner = owner(joint.ConnectedNode),
        .Node = AuthoredWorld(r, entity),
        .Connected = r.valid(joint.ConnectedNode) ? std::optional{AuthoredWorld(r, joint.ConnectedNode)} : std::nullopt,
    };
    if (const auto *definition = r.try_get<const PhysicsJointDef>(joint.JointDefEntity)) {
        input.Definition = *definition;
        input.Definition->Name.clear();
    }
    return input;
}

bool IsActiveJoint(const JointInput &input) {
    return input.Definition && input.Connected && input.Owner != state::Null && input.Owner != input.ConnectedOwner;
}

void ApplyCollider(rbp::Shape &shape, state::Entity entity, const ColliderInput &input) {
    shape.UserData = uint64_t(uint32_t(entity)) + 1;
    shape.HasMaterial = true;
    shape.Surface = ToRbp(input.Material);
    shape.HasFilter = input.HasFilter;
    shape.Mask = {input.Layer, input.Collides};
}

void ClearContacts(PhysicsState &s, state::Scene &r) {
    s.Contacts.clear();
    s.ContactFrames.clear();
    r.Context.get<PhysicsContactImpacts>().Events.clear();
    auto &sustained = r.Context.get<PhysicsSustainedContacts>();
    sustained.Active.clear();
    sustained.Step = ++s.ContactStep;
}

void ClearSimulation(PhysicsState &s, state::Scene &r) {
    s.Clearing = true;
    r.clear<PhysicsConstraintHandle>();
    r.clear<PhysicsBodyHandle>();
    r.clear<BodyPoseCache>();
    s.Clearing = false;
    s.Bodies.clear();
    s.LeafOwners.clear();
    s.Joints.clear();
    s.JointUpdates.clear();
    s.Entities.clear();
    s.Destroyed.clear();
    s.SampleOrder.reset();
    s.SensorFollowers.clear();
    s.World.reset();
    s.WorldAnchor = rbp::NoIndex;
    s.Baked.reset();
    ClearContacts(s, r);
}

void OnDestroyPhysicsConstraint(state::Scene &r, state::Entity e) {
    auto *s = r.Context.find<PhysicsState>();
    if (!s || s->Clearing || !s->World) return;
    s->World->RemoveJoint(r.get<const PhysicsConstraintHandle>(e).ConstraintIndex);
    s->Invalidate();
}

// A destroyed entity's reactive entries are discarded, so its motion, collider and joint inputs record their entity here.
void OnDestroyPhysicsInput(state::Scene &r, state::Entity e) {
    if (auto *s = r.Context.find<PhysicsState>()) s->Destroyed.push_back(e);
}

// Every pool reserves twice its current need, with room for at least HeadroomColliders hull colliders, so added bodies build in place.
constexpr uint64_t HeadroomColliders = 16;

rbp::WorldLimits Limits(const state::Scene &r) {
    const uint32_t colliders = uint32_t(r.view<const ColliderShape>().size());
    const uint32_t motions = uint32_t(r.view<const PhysicsMotion>().size());
    const auto joints = uint32_t(r.view<const PhysicsJoint>().size());
    // Hulls take vertex and face runs, and triangle meshes take vertex, triangle and BVH runs.
    // Spheres, capsules and cylinders cook as hulls under a taper or a nonuniform scale.
    uint64_t hulls = 0, vertices = 1, triangles = 1;
    for (const auto [e, collider] : r.view<const ColliderShape>().each()) {
        if (std::holds_alternative<physics::TriangleMesh>(collider.Shape)) {
            if (const auto mesh = TryGetMesh(r, collider.MeshEntity)) {
                vertices += mesh->VertexCount();
                triangles += uint64_t(mesh->TriangleIndexCount() / 3);
            }
        } else if (!std::holds_alternative<physics::Box>(collider.Shape) && !std::holds_alternative<physics::Plane>(collider.Shape)) {
            ++hulls;
            vertices += rbp::MaxHullVertices;
        }
    }
    if (vertices * 3 > UINT32_MAX || triangles * 6 > UINT32_MAX) throw std::runtime_error("Physics geometry exceeds RBP pool indexing.");
    const auto room = [](uint64_t need, uint64_t floor) { return uint32_t(std::min<uint64_t>(std::max(2 * need, floor), UINT32_MAX)); };
    return {
        .Bodies = room(colliders + motions + 1, HeadroomColliders + 1),
        .Shapes = room(4 * colliders + motions + 4, 4 * HeadroomColliders + 4),
        .Joints = std::max(8u, joints + joints / 2),
        .ShapeVertices = room(vertices * 3, HeadroomColliders * rbp::MaxHullVertices * 3),
        .HullFaces = room(hulls * 384, HeadroomColliders * 384),
        .Triangles = room(triangles * 3, HeadroomColliders * 4 * 3),
        .BvhNodes = room(triangles * 6, HeadroomColliders * 4 * 6),
        .CompoundChildren = room(2 * colliders, 2 * HeadroomColliders),
    };
}

auto PoolOverflows(const rbp::World &world) {
    const auto &o = world.Overflow;
    return std::array{o.Bodies, o.Shapes, o.ShapeVertices, o.HullFaces, o.Triangles, o.BvhNodes, o.CompoundChildren};
}

physics::RbpBody CookBody(PhysicsState &s, const state::Scene &r, const BodyInput &input, std::span<const ColliderInput> leaves, const physics::RbpBody *previous = nullptr) {
    auto &world = *s.World;
    std::vector<rbp::Index> shapes;
    shapes.reserve(leaves.size());
    try {
        for (uint32_t i = 0; i < leaves.size(); ++i) {
            const auto &leaf = leaves[i];
            const auto &desc = leaf.Shape;
            auto local = PoseOf(leaf.Local);
            local.Position += rbp::Rotate(local.Orientation, ToRbp(desc.LocalOffset * leaf.Local.S));
            const auto mesh = IsMeshBackedShape(desc.Shape) ? TryGetMesh(r, desc.MeshEntity) : std::nullopt;
            const auto shape = physics::BuildRbpShape(world, desc.Shape, mesh ? &*mesh : nullptr, leaf.Local.S, local);
            shapes.push_back(shape);
            ApplyCollider(world.Shapes[shape], input.Colliders[i], leaf);
        }
        const auto body = physics::BuildRbpBody(world, shapes, input.Node, input.Motion ? &*input.Motion : nullptr, &input.Velocity, input.Sensor, previous);
        world.RemoveShapes(shapes);
        return body;
    } catch (...) {
        world.RemoveShapes(shapes);
        throw;
    }
}

// Drops the leaf owner entries that still name `body` for its colliders.
void ReleaseLeaves(PhysicsState &s, state::Entity body, std::span<const state::Entity> colliders) {
    for (const auto collider : colliders) {
        if (const auto leaf = s.LeafOwners.find(collider); leaf != s.LeafOwners.end() && leaf->second == body) s.LeafOwners.erase(leaf);
    }
}

void BuildBody(PhysicsState &s, state::Scene &r, state::Entity entity, BodyInput input, std::vector<ColliderInput> leaves) {
    const auto body = CookBody(s, r, input, leaves);
    for (const auto collider : input.Colliders) s.LeafOwners[collider] = entity;
    if (s.Entities.size() <= body.Body) s.Entities.resize(body.Body + 1, state::Null);
    s.Entities[body.Body] = entity;
    r.emplace_or_replace<PhysicsBodyHandle>(entity, PhysicsBodyHandle{body.Body});
    if (input.Motion) r.emplace_or_replace<BodyPoseCache>(entity, BodyPoseCache{{physics::RbpNodePose(body.InitialPose, body.Frame)}});
    s.Bodies.emplace(entity, BodyRecord{body, std::move(input), std::move(leaves)});
}

// Builds every body of the scene, motion bodies first, each group in entity order.
void BuildBodies(PhysicsState &s, state::Scene &r) {
    std::vector<ColliderInput> leaves;
    const auto build = [&](state::Entity entity) {
        if (s.Bodies.contains(entity)) return;
        if (auto input = ReadBody(s, r, entity, leaves)) BuildBody(s, r, entity, std::move(*input), std::move(leaves));
    };
    for (const auto entity : SortedEntities(r.view<const PhysicsMotion>())) build(entity);
    for (const auto entity : SortedEntities(r.view<const ColliderShape>())) build(entity);
}

void BuildJoint(PhysicsState &s, state::Scene &r, state::Entity entity) {
    const auto it = s.Joints.find(entity);
    if (it == s.Joints.end() || !IsActiveJoint(it->second)) return;
    const auto &input = it->second;
    const auto &joint = input.Joint;
    const auto &def = *input.Definition;
    const auto owner = input.Owner, connected = input.ConnectedOwner;
    auto &world = *s.World;
    if (connected == state::Null && s.WorldAnchor == rbp::NoIndex) s.WorldAnchor = world.AddBody({});
    // KHR measures the connected frame in the joint node's frame. RBP measures A in B.
    const auto a = PoseOf(*input.Connected), b = PoseOf(input.Node);
    rbp::JointDesc desc{
        .BodyA = connected == state::Null ? s.WorldAnchor : s.Bodies.at(connected).Cooked.Body,
        .BodyB = s.Bodies.at(owner).Cooked.Body,
        .AtA = a.Position,
        .AtB = b.Position,
        .FrameA = a.Orientation,
        .FrameB = b.Orientation,
        .Linear = {rbp::AxisFree, rbp::AxisFree, rbp::AxisFree},
        .Collide = joint.EnableCollision,
    };
    for (const auto &limit : def.Limits) {
        const bool linear = !limit.LinearAxes.empty();
        const auto &axes = linear ? limit.LinearAxes : limit.AngularAxes;
        uint32_t mask = 0;
        for (auto axis : axes) {
            if (axis > 2) throw std::invalid_argument("A physics joint axis must be X, Y or Z.");
            mask |= 1u << axis;
        }
        if (!mask) continue;
        const float low = limit.Min.value_or(-INFINITY), high = limit.Max.value_or(INFINITY);
        auto *modes = linear ? desc.Linear : desc.Angular;
        const auto configure = [&](uint32_t axis, bool grouped) {
            if (!grouped && low == high && low == 0) modes[axis] = rbp::AxisLocked;
            else if (!grouped && low == high) {
                modes[axis] = rbp::AxisPositioned;
                (linear ? desc.LinearMotorTarget : desc.MotorTarget)[axis] = low;
                (linear ? desc.LinearMotorMaxForce : desc.MotorMaxTorque)[axis] = INFINITY;
            } else modes[axis] = rbp::AxisLimited;
            (linear ? desc.LinearLimitLow : desc.LimitLow)[axis] = low;
            (linear ? desc.LinearLimitHigh : desc.LimitHigh)[axis] = high;
            (linear ? desc.LinearStiffness : desc.AngularStiffness)[axis] = limit.Stiffness.value_or(INFINITY);
            (linear ? desc.LinearDamping : desc.AngularDamping)[axis] = limit.Damping;
        };
        if (std::popcount(mask) > 1 && !(low == 0 && high == 0)) {
            const auto axis = uint32_t(std::countr_zero(mask));
            configure(axis, true);
            (linear ? desc.LinearLimitAxes : desc.AngularLimitAxes)[axis] = mask;
        } else
            for (uint32_t axis = 0; axis < 3; ++axis)
                if (mask & (1u << axis)) configure(axis, false);
    }
    for (const auto &drive : def.Drives) {
        if (drive.Axis > 2) throw std::invalid_argument("A physics joint drive axis must be X, Y or Z.");
        const bool linear = drive.Type == PhysicsDriveType::Linear;
        float mass = 1;
        if (drive.Mode == PhysicsDriveMode::Acceleration) {
            const auto axis = rbp::Rotate(b.Orientation, rbp::float3{drive.Axis == 0 ? 1.f : 0, drive.Axis == 1 ? 1.f : 0, drive.Axis == 2 ? 1.f : 0});
            float inverse = 0;
            for (auto body : {desc.BodyA, desc.BodyB}) {
                if (linear) inverse += world.Masses[body].InvMass;
                else {
                    const auto local = rbp::Rotate(rbp::QuatConjugate(world.Poses[body].Orientation), axis);
                    inverse += simd::dot(local * local, world.Masses[body].InvInertiaLocal);
                }
            }
            mass = inverse > 0 ? 1 / inverse : 0;
        }
        desc.Drives[(linear ? 0 : 3) + drive.Axis] = {.Enabled = uint32_t(drive.Stiffness > 0 || drive.Damping > 0), .Speed = drive.VelocityTarget, .Target = drive.PositionTarget, .MaxForce = drive.MaxForce, .Stiffness = drive.Stiffness * mass, .Damping = drive.Damping * mass};
    }
    if (const auto *existing = r.try_get<const PhysicsConstraintHandle>(entity)) {
        if (!world.SetJoint(existing->ConstraintIndex, desc)) throw std::runtime_error("RBP could not update a physics joint.");
        return;
    }
    const auto handle = world.AddJoint(desc);
    if (handle == rbp::NoIndex) throw std::runtime_error("RBP could not create a physics joint.");
    r.emplace_or_replace<PhysicsConstraintHandle>(entity, PhysicsConstraintHandle{handle});
}

void Rebuild(state::Scene &r) {
    const profile::CpuScope scope{"PhysicsRebuild"};
    auto &s = r.Context.get<PhysicsState>();
    ClearSimulation(s, r);
    if (!s.Context) return;
    if (!s.Solver) s.Solver.emplace(*s.Context);
    UpdateMasks(s, r);
    s.World.emplace(*s.Context, Limits(r));
    BuildBodies(s, r);
    for (const auto entity : SortedEntities(r.view<const PhysicsJoint>())) {
        s.Joints.emplace(entity, ReadJoint(s, r, entity, r.get<const PhysicsJoint>(entity)));
        BuildJoint(s, r, entity);
    }
}

const std::vector<state::Entity> &SampleOrder(PhysicsState &s, const state::Scene &r) {
    if (!s.SampleOrder) {
        std::vector<std::pair<uint32_t, state::Entity>> depths;
        for (auto entity : r.view<const PhysicsBodyHandle, const BodyPoseCache>()) {
            uint32_t depth = 0;
            for (auto parent = ParentOrNull(r, entity); parent != state::Null; parent = ParentOrNull(r, parent)) ++depth;
            depths.emplace_back(depth, entity);
        }
        std::ranges::sort(depths);
        s.SampleOrder = depths | std::views::values | std::ranges::to<std::vector>();
    }
    return *s.SampleOrder;
}

void Restart(PhysicsState &s, state::Scene &r) {
    const profile::CpuScope scope{"PhysicsReset"};
    ClearContacts(s, r);
    // Sampling writes the world transforms of posed bodies and their descendants, so each posed subtree returns to its authored pose.
    RecomputeWorldTransforms(r, SampleOrder(s, r), {}, {});
    for (const auto &[entity, record] : s.Bodies) {
        const auto &body = record.Cooked;
        s.World->Poses[body.Body] = body.InitialPose;
        s.World->Velocities[body.Body] = body.InitialVelocity;
        if (auto *cache = r.try_edit<BodyPoseCache>(entity)) cache->Frames = {physics::RbpNodePose(body.InitialPose, body.Frame)};
    }
    s.World->ResetDynamics();
    for (auto entity : s.JointUpdates) BuildJoint(s, r, entity);
    s.JointUpdates.clear();
    s.SensorFollowers.clear();
    for (const auto &[entity, record] : s.Bodies) {
        if (!r.all_of<TriggerTag>(entity) || r.all_of<PhysicsMotion>(entity)) continue;
        const auto owner = MotionOwner(r, entity);
        if (owner == state::Null) continue;
        const auto body = record.Cooked.Body, owner_body = s.Bodies.at(owner).Cooked.Body;
        s.SensorFollowers.push_back({body, owner_body, rbp::ComposePose(Inverse(s.World->Poses[owner_body]), s.World->Poses[body])});
    }
    s.World->WeldStatic();
    s.Baked = s.CacheStartFrame;
    s.ContactFrames = {{r.Context.get<PhysicsContactImpacts>(), r.Context.get<PhysicsSustainedContacts>()}};
}

state::Entity EntityForBody(const PhysicsState &s, rbp::Index body) { return body < s.Entities.size() ? s.Entities[body] : state::Null; }
state::Entity Collider(uint64_t data) { return data ? state::Entity(uint32_t(data - 1)) : state::Null; }
rbp::float3 WorldPoint(const rbp::ContactSide &side) { return rbp::WorldPoint(side.InitialPose, side.Point); }
rbp::float3 PointVelocity(const rbp::ContactSide &side) { return side.Velocity.Linear + simd::cross(side.Velocity.Angular, rbp::Rotate(side.Pose.Orientation, side.Point)); }

void CollectSubstep(PhysicsState &s, state::Scene &r, std::map<ContactKey, ContactSum> &frame) {
    const profile::CpuScope scope{"PhysicsContacts"};
    auto events = s.World->TakeContactChanges();
    std::ranges::stable_sort(events, {}, Key);
    ++s.Substep;
    auto &impacts = r.Context.get<PhysicsContactImpacts>().Events;
    const auto gravity = s.Settings.Gravity;
    for (const auto manifold : events | std::views::chunk_by([](const auto &a, const auto &b) { return Key(a) == Key(b); })) {
        const auto &first = manifold.front();
        const auto a = EntityForBody(s, first.A.Slot), b = EntityForBody(s, first.B.Slot);
        if (!r.valid(a) || !r.valid(b) || (!r.all_of<ReportContacts>(a) && !r.all_of<ReportContacts>(b))) continue;
        const auto key = Key(first);
        float normal_impulse = 0, support = 0, approach = 0;
        rbp::float3 resultant{};
        bool present = false;
        for (const auto &c : manifold) {
            if (c.Kind == rbp::ContactRemoved) continue;
            present = true;
            const float impulse = std::max(0.f, -c.Lambda.x * c.DeltaTime + c.BounceImpulse);
            normal_impulse += impulse;
            resultant += impulse * WorldPoint(c.SideA);
            approach = std::max(approach, c.Approach);
            const float force_a = c.SideA.InvMass > 0 ? std::max(0.f, -simd::dot(gravity * s.World->Masses[c.A.Slot].GravityScale, c.Normal)) / c.SideA.InvMass : 0;
            const float force_b = c.SideB.InvMass > 0 ? std::max(0.f, simd::dot(gravity * s.World->Masses[c.B.Slot].GravityScale, c.Normal)) / c.SideB.InvMass : 0;
            support = std::max(support, (force_a + force_b) * c.DeltaTime);
        }
        if (!present) continue;
        auto [it, fresh] = s.Contacts.try_emplace(key);
        auto &tracked = it->second;
        if (fresh) tracked.Id = s.NextContactId++;
        tracked.Seen = s.Substep;
        if (normal_impulse <= 0) continue;
        resultant /= normal_impulse;
        bool strike = tracked.PendingImpact;
        tracked.PendingImpact = false;
        const float excess = std::max(0.f, normal_impulse - support);
        strike &= approach > 2 * s.Settings.DeltaTime * simd::length(gravity) && excess > 1e-6f;
        auto &sum = frame[key];
        for (const auto &c : manifold) {
            if (c.Kind == rbp::ContactRemoved) continue;
            const float impulse = std::max(0.f, -c.Lambda.x * c.DeltaTime);
            sum.Last = c;
            sum.NormalImpulse += impulse;
            sum.Point += WorldPoint(c.SideA) * impulse;
            sum.Normal -= c.Normal * impulse;
            const auto velocity = PointVelocity(c.SideA) - PointVelocity(c.SideB);
            sum.Slip += (velocity - c.Normal * simd::dot(velocity, c.Normal)) * impulse;
            sum.LocalA += c.SideA.Point * impulse;
            sum.LocalB += c.SideB.Point * impulse;
            sum.FrictionImpulse += (-c.ForceOnA() - c.Normal * c.Lambda.x) * c.DeltaTime;
            if (!strike) continue;
            const auto vector = c.ImpulseOnA() * (excess / normal_impulse);
            const float magnitude = simd::length(vector);
            if (magnitude <= 1e-6f) continue;
            const float speed = excess * (c.SideA.InvMass + c.SideB.InvMass) / (1 + c.Restitution);
            for (bool side_a : {true, false}) impacts.push_back({
                .Entity = side_a ? a : b,
                .ColliderEntity = Collider(side_a ? c.SideA.UserData : c.SideB.UserData),
                .Other = side_a ? b : a,
                .OtherColliderEntity = Collider(side_a ? c.SideB.UserData : c.SideA.UserData),
                .Point = FromRbp(WorldPoint(c.SideA)),
                .ResultantPoint = FromRbp(resultant),
                .Direction = FromRbp(vector * ((side_a ? 1.f : -1.f) / magnitude)),
                .Impulse = magnitude,
                .Speed = speed,
                .OtherInvMass = side_a ? c.SideB.InvMass : c.SideA.InvMass,
                .NominalArea = c.NominalArea,
            });
        }
    }
    std::erase_if(s.Contacts, [&](const auto &entry) { return entry.second.Seen != s.Substep; });
}

void StepSimulation(PhysicsState &s, state::Scene &r, float sim_dt, uint32_t substeps) {
    const profile::CpuScope scope{"PhysicsFrame"};
    auto &out = r.Context.get<PhysicsSustainedContacts>();
    out.Active.clear();
    out.Step = ++s.ContactStep;
    r.Context.get<PhysicsContactImpacts>().Events.clear();
    if (sim_dt <= 0) return;
    substeps = std::max(1u, substeps);
    s.Settings.DeltaTime = sim_dt / float(substeps);
    auto &world = *s.World;
    world.TrackContacts = !r.view<const ReportContacts>().empty();
    std::map<ContactKey, ContactSum> contacts;
    rbp::AdvanceResult completed;
    {
        const profile::CpuScope advance_scope{"RbpAdvance"};
        completed = s.Solver->Advance(world, s.Settings, substeps, s.SensorFollowers, [&](const rbp::StepResult &) {
            CollectSubstep(s, r, contacts);
        });
    }
    profile::RecordCounter("RbpContactRefusals", double(completed.ContactRefusals));
    for (const auto &[key, sum] : contacts) {
        auto it = s.Contacts.find(key);
        if (it == s.Contacts.end() || sum.NormalImpulse <= 0) continue;
        auto &tracked = it->second;
        const auto &c = sum.Last;
        const auto local_a = sum.LocalA / sum.NormalImpulse, local_b = sum.LocalB / sum.NormalImpulse;
        const auto sweep_a = tracked.HasPreviousFrame ? rbp::Rotate(c.SideA.Pose.Orientation, local_a - tracked.PreviousA) / sim_dt : rbp::float3{};
        const auto sweep_b = tracked.HasPreviousFrame ? rbp::Rotate(c.SideB.Pose.Orientation, local_b - tracked.PreviousB) / sim_dt : rbp::float3{};
        tracked.PreviousA = local_a;
        tracked.PreviousB = local_b;
        tracked.HasPreviousFrame = true;
        out.Active.push_back({
            .Id = tracked.Id,
            .Sides = {SustainedContactSide{EntityForBody(s, c.A.Slot), Collider(c.SideA.UserData), FromRbp(sweep_a)}, SustainedContactSide{EntityForBody(s, c.B.Slot), Collider(c.SideB.UserData), FromRbp(sweep_b)}},
            .Point = FromRbp(sum.Point / sum.NormalImpulse),
            .Normal = FromRbp(simd::normalize(sum.Normal)),
            .Slip = FromRbp(sum.Slip / sum.NormalImpulse),
            .NormalForce = sum.NormalImpulse / sim_dt,
            .FrictionForce = FromRbp(sum.FrictionImpulse / sim_dt),
            .NominalArea = c.NominalArea,
            .NominalExtent = c.NominalExtent,
            .Restitution = c.Restitution,
            .Friction = c.Friction,
        });
    }
}

void BakeFrame(state::Scene &r, state::Entity viewport, PhysicsState &s, uint32_t frame, float fps) {
    const auto &settings = r.get<const PhysicsSimulationSettings>(viewport);
    const float sim_dt = (fps > 0 ? 1.f / fps : 1.f / 60) * settings.TimeScale;
    StepSimulation(s, r, sim_dt, settings.SubstepsPerFrame);
    for (auto [entity, handle, cache] : r.view<const PhysicsBodyHandle, BodyPoseCache>().each()) {
        const auto &body = s.Bodies.at(entity).Cooked;
        cache.Frames.push_back(physics::RbpNodePose(s.World->Poses[body.Body], body.Frame));
    }
    s.ContactFrames.push_back({std::move(r.Context.get<PhysicsContactImpacts>()), std::move(r.Context.get<PhysicsSustainedContacts>())});
    s.Baked = frame;
}

void UpdateSettings(state::Scene &r, state::Entity viewport, float fps) {
    auto &s = r.Context.get<PhysicsState>();
    const auto &settings = r.get<const PhysicsSimulationSettings>(viewport);
    if (s.AppliedSettings == settings && s.CacheFps == fps) return;
    physics::ApplySimulationSettings(r, settings);
    s.AppliedSettings = settings;
    s.CacheFps = fps;
    s.Invalidate();
}

// A filter uses each system once, whether as a member or as a collision target.
void CountDefinitionUses(state::Scene &r) {
    auto &counts = r.Context.get<PhysicsDefinitionUses>().Counts;
    counts.clear();
    const auto use = [&](state::Entity definition) {
        if (definition != state::Null) ++counts[definition];
    };
    for (const auto [_, material] : r.view<const ColliderMaterial>().each()) {
        use(material.PhysicsMaterialEntity);
        use(material.CollisionFilterEntity);
    }
    for (const auto [_, trigger] : r.view<const TriggerNodes>().each()) use(trigger.CollisionFilterEntity);
    for (const auto [_, filter] : r.view<const CollisionFilter>().each()) {
        for (const auto system : filter.Systems) use(system);
        for (const auto system : filter.CollideSystems)
            if (std::ranges::find(filter.Systems, system) == filter.Systems.end()) use(system);
    }
    for (const auto [_, joint] : r.view<const PhysicsJoint>().each()) use(joint.JointDefEntity);
}
} // namespace

namespace physics {
void ProcessChanges(state::Scene &r) {
    if (!reactive(r, Change::PhysicsDefinitionUses).empty()) CountDefinitionUses(r);
    auto &s = r.Context.get<PhysicsState>();
    const auto &inputs = reactive(r, Change::PhysicsInput), &moved = reactive(r, Change::PhysicsTransform), &geometry = reactive(r, Change::PhysicsGeometry);
    const bool definitions = AnyChanged(r, Change::PhysicsMaterialDef, Change::CollisionSystemDef, Change::CollisionFilterDef);
    if (inputs.empty() && moved.empty() && geometry.empty() && !definitions && s.Destroyed.empty()) return;
    const auto destroyed = std::exchange(s.Destroyed, {});
    // Without motion or colliders, the scene forms no body.
    if (r.view<const PhysicsMotion>().empty() && r.view<const ColliderShape>().empty()) {
        if (s.World) ClearSimulation(s, r);
        return;
    }
    if (!s.World) {
        Rebuild(r);
        return;
    }
    if (definitions) UpdateMasks(s, r);

    // The bodies a change reaches: each body and collider at or under a changed node, with each collider's former and current owner.
    std::vector<state::Entity> candidates;
    const auto touch = [&](state::Entity e) {
        if (const auto owner = s.LeafOwners.find(e); owner != s.LeafOwners.end()) candidates.push_back(owner->second);
        if (s.Bodies.contains(e)) candidates.push_back(e);
        if (!r.valid(e) || !r.any_of<PhysicsMotion, ColliderShape>(e)) return;
        candidates.push_back(e);
        if (const auto owner = MotionOwner(r, e); owner != state::Null) candidates.push_back(owner);
    };
    const auto touch_subtree = [&](this const auto &self, state::Entity e) -> void {
        touch(e);
        if (!r.valid(e)) return;
        for (const auto child : Children{&r, e}) self(child);
    };
    for (const auto e : inputs) touch_subtree(e);
    for (const auto e : moved) touch_subtree(e);
    for (const auto e : destroyed) touch(e);
    const auto &mesh_colliders = r.Context.get<const MeshColliders>();
    for (const auto mesh_entity : geometry)
        for (const auto collider : mesh_colliders.Of(mesh_entity)) touch(collider);
    // A definition reaches every leaf that names one.
    if (definitions)
        for (const auto e : r.view<const ColliderMaterial>()) touch(e);
    SortUnique(candidates);

    // Each candidate leaves the world, joins it, or compares its inputs with the ones it was cooked from.
    struct Pending {
        state::Entity Entity;
        BodyInput Input;
        std::vector<ColliderInput> Leaves;
        bool Recook = false;
    };
    std::vector<state::Entity> removed;
    std::vector<Pending> changed, added;
    std::vector<ColliderInput> leaves;
    const auto mesh_edited = [&](const ColliderInput &leaf) { return IsMeshBackedShape(leaf.Shape.Shape) && geometry.contains(leaf.Shape.MeshEntity); };
    for (const auto e : candidates) {
        auto input = ReadBody(s, r, e, leaves);
        const auto it = s.Bodies.find(e);
        if (it == s.Bodies.end()) {
            if (input) added.push_back({e, std::move(*input), std::move(leaves)});
            continue;
        }
        // A body that stops forming one leaves the world, and one turning sensor or solid joins it again.
        if (!input || input->Sensor != it->second.Input.Sensor) {
            removed.push_back(e);
            if (input) added.push_back({e, std::move(*input), std::move(leaves)});
            continue;
        }
        const auto &record = it->second;
        const bool recook = input->Colliders != record.Input.Colliders || std::ranges::any_of(leaves, mesh_edited) ||
            !std::ranges::equal(leaves, record.Leaves, [](const auto &a, const auto &b) { return a.Shape == b.Shape && a.Local == b.Local; });
        if (recook || *input != record.Input || leaves != record.Leaves) changed.push_back({e, std::move(*input), std::move(leaves), recook});
    }
    // Motion bodies join first, each group in entity order, as a full build adds them.
    std::ranges::stable_partition(added, [](const Pending &p) { return p.Input.Motion.has_value(); });

    // Entities whose world transforms physics stops writing.
    std::vector<state::Entity> released;
    if (!removed.empty()) {
        std::vector<rbp::Index> bodies, shapes;
        for (const auto e : removed) {
            const auto it = s.Bodies.find(e);
            const auto &body = it->second.Cooked;
            bodies.push_back(body.Body);
            if (body.Shape != rbp::NoIndex) shapes.push_back(body.Shape);
            ReleaseLeaves(s, e, it->second.Input.Colliders);
            s.Entities[body.Body] = state::Null;
            if (r.remove<BodyPoseCache>(e)) released.push_back(e);
            r.remove<PhysicsBodyHandle>(e);
            s.Bodies.erase(it);
        }
        s.World->RemoveBodies(bodies);
        s.World->RemoveShapes(shapes);
        ClearContacts(s, r);
    }

    std::set<rbp::Index> reframed;
    const auto overflows = PoolOverflows(*s.World);
    try {
        for (auto &[e, input, next_leaves, recook] : changed) {
            auto &record = s.Bodies.at(e);
            auto &body = record.Cooked;
            const auto old_pose = body.InitialPose;
            const auto old_mass = s.World->Masses[body.Body];
            if (recook) {
                body = CookBody(s, r, input, next_leaves, &body);
            } else {
                if (input != record.Input) physics::UpdateRbpBody(*s.World, body, input.Node, input.Motion ? &*input.Motion : nullptr, &input.Velocity);
                // Compound children follow the collider order, so each changed leaf updates its surface in place.
                for (uint32_t i = 0; i < next_leaves.size(); ++i) {
                    if (next_leaves[i] != record.Leaves[i]) ApplyCollider(s.World->Shapes[s.World->Child(body.Shape, i)], input.Colliders[i], next_leaves[i]);
                }
            }
            const auto mass = s.World->Masses[body.Body];
            if (recook || simd::any(old_pose.Position != body.InitialPose.Position) || simd::any(old_pose.Orientation != body.InitialPose.Orientation) || old_mass.InvMass != mass.InvMass || simd::any(old_mass.InvInertiaLocal != mass.InvInertiaLocal)) reframed.insert(body.Body);
            // A body changing between static and moving takes or gives up its pose cache.
            if (input.Motion && !record.Input.Motion) r.emplace<BodyPoseCache>(e, BodyPoseCache{{physics::RbpNodePose(body.InitialPose, body.Frame)}});
            else if (!input.Motion && record.Input.Motion && r.remove<BodyPoseCache>(e)) released.push_back(e);
            ReleaseLeaves(s, e, record.Input.Colliders);
            for (const auto collider : input.Colliders) s.LeafOwners[collider] = e;
            record.Input = std::move(input);
            record.Leaves = std::move(next_leaves);
        }
        for (auto &[e, input, next_leaves, _] : added) BuildBody(s, r, e, std::move(input), std::move(next_leaves));
    } catch (...) {
        if (PoolOverflows(*s.World) == overflows) throw;
        Rebuild(r);
        return;
    }

    std::map<state::Entity, JointInput> joints;
    for (const auto [e, joint] : r.view<const PhysicsJoint>().each()) joints.emplace(e, ReadJoint(s, r, e, joint));
    if (uint32_t(std::ranges::count_if(joints, [](const auto &entry) { return IsActiveJoint(entry.second); })) > s.World->Joints.Capacity) {
        Rebuild(r);
        return;
    }
    if (!released.empty()) RecomputeWorldTransforms(r, released, r.view<const BodyPoseCache>() | std::ranges::to<std::vector>(), {});
    if (removed.empty() && changed.empty() && added.empty() && joints == s.Joints) return;
    s.Invalidate();
    s.SampleOrder.reset();
    for (const auto entity : r.view<const PhysicsConstraintHandle>()) {
        const auto it = joints.find(entity);
        const bool active = it != joints.end() && IsActiveJoint(it->second);
        // Removing a body retires its world joints, which then rebuild on restart.
        if (active && s.World->Joints[r.get<const PhysicsConstraintHandle>(entity).ConstraintIndex].Active) continue;
        r.remove<PhysicsConstraintHandle>(entity);
        if (active) s.JointUpdates.insert(entity);
    }
    for (const auto &[entity, next] : joints) {
        const auto old = s.Joints.find(entity);
        bool update = old == s.Joints.end() || old->second != next;
        if (const auto *handle = r.try_get<const PhysicsConstraintHandle>(entity)) {
            const auto &joint = s.World->Joints[handle->ConstraintIndex];
            update |= reframed.contains(joint.BodyA) || reframed.contains(joint.BodyB);
        }
        if (update) s.JointUpdates.insert(entity);
    }
    s.Joints = std::move(joints);
}

void ApplySimulationSettings(state::Scene &r, const PhysicsSimulationSettings &settings) {
    auto &step = r.Context.get<PhysicsState>().Settings;
    step.Gravity = ToRbp(settings.Gravity);
    step.Iterations = std::max(1u, settings.SolverIterations);
}
std::optional<uint32_t> BakedThrough(const state::Scene &r) { return r.Context.get<PhysicsState>().Baked; }
uint32_t BodyCount(const state::Scene &r) { return uint32_t(r.Context.get<PhysicsState>().Bodies.size()); }
bool DoesFilterAllow(const state::Scene &r, state::Entity source, state::Entity target) {
    const auto &masks = r.Context.get<PhysicsState>().Masks;
    const auto a = masks.find(source), b = masks.find(target);
    return a == masks.end() || b == masks.end() || (a->second.Layer & b->second.Collides) != 0;
}
void InvalidateCache(state::Scene &r) { r.Context.get<PhysicsState>().CacheInvalid = true; }
bool AdvancePlayback(state::Scene &r, state::Entity viewport, int from_frame, int to_frame, int range_start_frame, int range_end_frame, float fps) {
    auto &s = r.Context.get<PhysicsState>();
    UpdateSettings(r, viewport, fps);
    if (std::exchange(s.CacheInvalid, false) || uint32_t(range_start_frame) != s.CacheStartFrame) s.Invalidate();
    s.CacheStartFrame = range_start_frame;
    s.CacheEndFrame = range_end_frame;
    if (s.Bodies.empty()) return false;
    // An edit restarts the simulation at once, and the bake waits for the next frame change, showing the cached start until then.
    const bool restarted = !s.Baked;
    if (restarted) Restart(s, r);
    if (from_frame == to_frame && !restarted) return false;
    if (from_frame != to_frame) BakeThrough(r, viewport, to_frame, fps);
    SamplePosesAtFrame(r, float(to_frame));
    const auto &contacts = s.ContactFrames[std::clamp(uint32_t(to_frame), s.CacheStartFrame, *s.Baked) - s.CacheStartFrame];
    r.Context.get<PhysicsContactImpacts>() = contacts.Impacts;
    r.Context.get<PhysicsSustainedContacts>() = contacts.Sustained;
    return true;
}

void BakeThrough(state::Scene &r, state::Entity viewport, int through_frame, float fps) {
    auto &s = r.Context.get<PhysicsState>();
    if (s.Bodies.empty()) return;
    UpdateSettings(r, viewport, fps);
    if (!s.Baked) Restart(s, r);
    const uint32_t target = std::min(uint32_t(std::max(through_frame, int(s.CacheStartFrame))), s.CacheEndFrame);
    if (*s.Baked >= target) return;
    // Prediction records future contacts without publishing them to the audio timeline.
    auto impacts = std::move(r.Context.get<PhysicsContactImpacts>());
    auto sustained = std::move(r.Context.get<PhysicsSustainedContacts>());
    while (*s.Baked < target) BakeFrame(r, viewport, s, *s.Baked + 1, fps);
    r.Context.get<PhysicsContactImpacts>() = std::move(impacts);
    r.Context.get<PhysicsSustainedContacts>() = std::move(sustained);
}

void SamplePosesAtFrame(state::Scene &r, float frame) {
    auto &s = r.Context.get<PhysicsState>();
    if (s.Bodies.empty() || !s.Baked) return;
    const float clamped = std::clamp(frame, float(s.CacheStartFrame), float(*s.Baked));
    const uint32_t lo = uint32_t(std::floor(clamped));
    const uint32_t hi = std::min(lo + 1, *s.Baked);
    const float t = clamped - float(lo);
    const auto lo_idx = lo - s.CacheStartFrame, hi_idx = hi - s.CacheStartFrame;
    // Update parents before restoring the independent poses of nested bodies.
    // The posed bodies in entity index order, as `owned` requires.
    const auto bodies = r.view<const BodyPoseCache>() | std::ranges::to<std::vector>();
    for (const auto entity : SampleOrder(s, r)) {
        const auto &cache = r.get<const BodyPoseCache>(entity);
        const auto &a = cache.Frames[lo_idx], &b = cache.Frames[hi_idx];
        const auto position = Mix(a.P, b.P, t);
        const auto rotation = Slerp(a.R, b.R, t);
        const auto &world = *WorldTransformOf(r, entity);
        // A body already at its sampled pose has a consistent subtree.
        if (world.P == position && world.R == rotation) continue;
        // The sampled pose recomposes the body's subtree, leaving the other posed bodies to their own samples.
        SetWorldTransform(r, entity, {position, rotation, world.S});
        RecomputeWorldTransforms(r, Children{&r, entity} | std::ranges::to<std::vector>(), bodies, {});
    }
}

void Init(state::Scene &r) {
    auto &s = r.Context.emplace<PhysicsState>();
    r.Context.emplace<PhysicsDefinitionUses>();
    if (r.Context.get<const mtl::Context>().Device->supportsFamily(MTL::GPUFamilyMetal4)) s.Context.emplace();
    r.Context.emplace<PhysicsContactImpacts>();
    r.Context.emplace<PhysicsSustainedContacts>();
    r.Context.emplace<MeshColliders>();
    r.on_destroy<PhysicsConstraintHandle, &OnDestroyPhysicsConstraint>();
    r.on_destroy<PhysicsMotion, &OnDestroyPhysicsInput>();
    r.on_destroy<ColliderShape, &OnDestroyPhysicsInput>();
    r.on_destroy<PhysicsJoint, &OnDestroyPhysicsInput>();
    r.on_destroy<PhysicsJointDef, &OnDestroyPhysicsInput>();
    reactive(r, Change::PhysicsInput)
        .on<PhysicsMotion>(On::Create | On::Update | On::Destroy)
        .on<PhysicsVelocity>(On::Create | On::Update | On::Destroy)
        .on<ColliderShape>(On::Create | On::Update | On::Destroy)
        .on<SceneParent>(On::Create | On::Update | On::Destroy)
        .on<ColliderMaterial>(On::Create | On::Update | On::Destroy)
        .on<TriggerTag>(On::Create | On::Destroy)
        .on<PhysicsJoint>(On::Create | On::Update | On::Destroy)
        .on<::PhysicsJointDef>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::PhysicsTransform).on<Transform>(On::Update);
    reactive(r, Change::PhysicsGeometry).on<MeshGeometryDirty>(On::Create | On::Update).on<MeshPositionsChanged>(On::Create | On::Update);
    reactive(r, Change::ColliderPolicy).on<::ColliderPolicy>(On::Create | On::Update);
    reactive(r, Change::Colliders).on<ColliderShape>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::PhysicsMaterialDef).on<::PhysicsMaterial>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::CollisionSystemDef).on<CollisionSystem>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::CollisionFilterDef).on<CollisionFilter>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::PhysicsDefinitionUses)
        .on<ColliderMaterial>(On::Create | On::Update | On::Destroy)
        .on<TriggerNodes>(On::Create | On::Update | On::Destroy)
        .on<CollisionFilter>(On::Create | On::Update | On::Destroy)
        .on<PhysicsJoint>(On::Create | On::Update | On::Destroy);
}
void Deinit(state::Scene &r) {
    r.Context.erase<MeshColliders>();
    r.Context.erase<PhysicsDefinitionUses>();
    r.Context.erase<PhysicsState>();
}
void Clear(state::Scene &r) {
    if (auto *s = r.Context.find<PhysicsState>()) ClearSimulation(*s, r);
}
} // namespace physics
