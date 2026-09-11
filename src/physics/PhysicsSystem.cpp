#include "PhysicsSystem.h"
#include "PhysicsChanges.h"
#include "PhysicsContact.h"
#include "Profile.h"
#include "RbpBody.h"
#include "RbpShape.h"
#include "Reactive.h"
#include "Replay.h"
#include "Solver.h"
#include "TransformMath.h"
#include "mesh/Mesh.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "scene/WorldTransform.h"
#include "viewport/ViewportEvents.h"

#include <algorithm>
#include <array>
#include <bit>
#include <map>
#include <ranges>
#include <set>
#include <stdexcept>

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
struct BodyInput {
    Transform Node;
    entt::entity Parent = null_entity;
    std::optional<PhysicsMotion> Motion;
    PhysicsVelocity Velocity;
    bool Sensor = false;
    std::vector<entt::entity> Colliders;
    bool operator==(const BodyInput &) const = default;
};
struct ColliderInput {
    entt::entity Owner;
    ColliderShape Shape;
    Transform Local{};
    PhysicsMaterial Material{};
    uint32_t Layer = ~0u, Collides = ~0u;
    bool HasFilter = false;
    bool operator==(const ColliderInput &) const = default;
};
struct JointInput {
    PhysicsJoint Joint;
    std::optional<PhysicsJointDef> Definition{};
    entt::entity Owner = null_entity, ConnectedOwner = null_entity;
    Transform Node;
    std::optional<Transform> Connected;
    bool operator==(const JointInput &) const = default;
};
struct SceneInput {
    std::map<entt::entity, BodyInput> Bodies;
    std::map<entt::entity, ColliderInput> Colliders;
    std::map<entt::entity, JointInput> Joints;
};
struct PhysicsState {
    rbp::mtl::Context Context;
    std::optional<rbp::Solver> Solver;
    std::optional<rbp::World> World;
    rbp::StepSettings Settings;
    SceneInput Input;
    std::set<entt::entity> JointUpdates;
    PhysicsSimulationSettings AppliedSettings;
    float CacheFps = 0;
    bool InputDirty = false;
    bool Evaluate = false;
    std::map<entt::entity, physics::RbpBody> Bodies;
    std::vector<entt::entity> Entities;
    std::map<entt::entity, rbp::CollisionMask> Masks;
    std::vector<rbp::SensorFollower> SensorFollowers;
    std::filesystem::path CapturePath;
    std::optional<rbp::replay::Writer> Capture;
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

    void Invalidate() {
        Baked.reset();
        Evaluate = true;
    }
};

rbp::Pose PoseOf(const Transform &t) { return rbp::At(ToRbp(t.P), ToRbp(numeric::Normalize(t.R))); }
entt::entity MotionOwner(const entt::registry &r, entt::entity e) {
    return FindAncestorIf(r, e, [&](auto node) { return r.all_of<PhysicsMotion>(node); });
}

void UpdateMasks(PhysicsState &s, const entt::registry &r) {
    std::map<entt::entity, uint32_t> bits;
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

Transform ComposeAuthored(const Transform &parent, Transform result, const mat4 &inverse) {
    // Preserve exact scale under rigid edits instead of remeasuring quaternion matrix columns.
    if (inverse == I4 && parent.S.x == parent.S.y && parent.S.y == parent.S.z) {
        result.P *= parent.S;
        result.S *= parent.S;
    } else result = ToTransform(ToMatrix(Transform{.S = parent.S}) * inverse * ToMatrix(result));
    result.P = parent.P + numeric::Rotate(parent.R, result.P);
    result.R = numeric::Normalize(parent.R * result.R);
    return result;
}

SceneInput ReadScene(const PhysicsState &s, const entt::registry &r) {
    SceneInput input;
    std::map<entt::entity, Transform> transforms;
    const auto transform = [&](this auto &self, entt::entity e) -> Transform {
        if (const auto it = transforms.find(e); it != transforms.end()) return it->second;
        Transform result;
        if (const auto *local = r.try_get<const Transform>(e)) {
            result = *local;
            if (const auto parent = ParentOrNull(r, e); parent != null_entity) {
                const auto *inverse = r.try_get<const ParentInverse>(e);
                result = ComposeAuthored(self(parent), *local, inverse ? inverse->M : I4);
            }
        }
        transforms.emplace(e, result);
        return result;
    };
    const auto add_body = [&](entt::entity entity) {
        if (input.Bodies.contains(entity)) return;
        const auto *motion = r.try_get<const PhysicsMotion>(entity);
        const bool sensor = r.all_of<TriggerTag, ColliderShape>(entity);
        if (!motion && !sensor && MotionOwner(r, entity) != null_entity) return;
        auto &body = input.Bodies[entity];
        body.Node = transform(entity);
        body.Parent = ParentOrNull(r, entity);
        if (motion) {
            body.Motion = *motion;
            if (const auto *velocity = r.try_get<const PhysicsVelocity>(entity)) body.Velocity = *velocity;
        }
        body.Sensor = sensor;
        if (r.all_of<ColliderShape>(entity)) body.Colliders.push_back(entity);
        if (motion && !sensor) {
            const auto gather = [&](this auto &self, entt::entity node) -> void {
                for (auto child : Children{&r, node}) {
                    if (r.all_of<PhysicsMotion>(child)) continue;
                    if (r.all_of<ColliderShape>(child) && !r.all_of<TriggerTag>(child)) body.Colliders.push_back(child);
                    self(child);
                }
            };
            gather(entity);
        }
        const auto local_transform = [&](this auto &self, entt::entity node) -> Transform {
            if (node == entity) return {.S = body.Node.S};
            const auto *inverse = r.try_get<const ParentInverse>(node);
            return ComposeAuthored(self(ParentOrNull(r, node)), r.get<const Transform>(node), inverse ? inverse->M : I4);
        };
        for (auto collider : body.Colliders) {
            ColliderInput leaf{.Owner = entity, .Shape = r.get<const ColliderShape>(collider), .Local = local_transform(collider)};
            if (const auto *material = r.try_get<const ColliderMaterial>(collider)) {
                if (const auto *definition = r.try_get<const PhysicsMaterial>(material->PhysicsMaterialEntity)) leaf.Material = *definition;
                if (const auto it = s.Masks.find(material->CollisionFilterEntity); it != s.Masks.end()) {
                    leaf.Layer = it->second.Layer;
                    leaf.Collides = it->second.Collides;
                    leaf.HasFilter = true;
                }
            }
            leaf.Material.Name.clear();
            input.Colliders.emplace(collider, std::move(leaf));
        }
    };
    for (auto e : SortedEntities(r.view<const PhysicsMotion>())) add_body(e);
    for (auto e : SortedEntities(r.view<const ColliderShape>())) add_body(e);
    for (auto [e, joint] : r.view<const PhysicsJoint>().each()) {
        JointInput value{.Joint = joint, .Node = transform(e), .Connected = r.valid(joint.ConnectedNode) ? std::optional{transform(joint.ConnectedNode)} : std::nullopt};
        const auto owner = [&](entt::entity node) { return FindAncestorIf(r, node, [&](auto ancestor) { return input.Bodies.contains(ancestor); }); };
        value.Owner = owner(e);
        value.ConnectedOwner = owner(joint.ConnectedNode);
        if (const auto *definition = r.try_get<const PhysicsJointDef>(joint.JointDefEntity)) {
            value.Definition = *definition;
            value.Definition->Name.clear();
        }
        input.Joints.emplace(e, std::move(value));
    }
    return input;
}

bool RequiresRebuild(const SceneInput &before, const SceneInput &after) {
    if (before.Bodies.size() != after.Bodies.size() || before.Colliders.size() != after.Colliders.size()) return true;
    for (const auto &[e, body] : after.Bodies) {
        const auto old = before.Bodies.find(e);
        if (old == before.Bodies.end() || old->second.Motion.has_value() != body.Motion.has_value() || old->second.Sensor != body.Sensor) return true;
    }
    for (const auto &[e, leaf] : after.Colliders) {
        const auto old = before.Colliders.find(e);
        if (old == before.Colliders.end() || old->second.Owner != leaf.Owner) return true;
    }
    return false;
}

bool IsActiveJoint(const JointInput &input) {
    return input.Definition && input.Connected && input.Owner != null_entity && input.Owner != input.ConnectedOwner;
}

void ApplyCollider(rbp::Shape &shape, entt::entity entity, const ColliderInput &input) {
    shape.UserData = uint64_t(uint32_t(entity)) + 1;
    shape.HasMaterial = true;
    shape.Surface = ToRbp(input.Material);
    shape.HasFilter = input.HasFilter;
    shape.Mask = {input.Layer, input.Collides};
}

void ClearContacts(PhysicsState &s, entt::registry &r) {
    s.Capture.reset();
    s.Contacts.clear();
    s.ContactFrames.clear();
    r.ctx().get<PhysicsContactImpacts>().Events.clear();
    auto &sustained = r.ctx().get<PhysicsSustainedContacts>();
    sustained.Active.clear();
    sustained.Step = ++s.ContactStep;
}

void ClearSimulation(PhysicsState &s, entt::registry &r) {
    s.Clearing = true;
    r.clear<PhysicsConstraintHandle>();
    r.clear<PhysicsBodyHandle>();
    r.clear<BodyPoseCache>();
    s.Clearing = false;
    s.Bodies.clear();
    s.Entities.clear();
    s.SensorFollowers.clear();
    s.World.reset();
    s.WorldAnchor = rbp::NoIndex;
    s.Baked.reset();
    ClearContacts(s, r);
}

void OnDestroyPhysicsBody(entt::registry &r, entt::entity e) {
    auto *s = r.ctx().find<PhysicsState>();
    if (!s || s->Clearing || !s->World) return;
    const auto it = s->Bodies.find(e);
    if (it == s->Bodies.end()) return;
    s->World->RemoveBody(it->second.Body);
    if (it->second.Shape != rbp::NoIndex) s->World->RemoveShape(it->second.Shape);
    s->Entities[it->second.Body] = null_entity;
    s->Bodies.erase(it);
    s->Baked.reset();
    s->Contacts.clear();
    r.ctx().get<PhysicsSustainedContacts>().Active.clear();
}

void OnDestroyPhysicsConstraint(entt::registry &r, entt::entity e) {
    auto *s = r.ctx().find<PhysicsState>();
    if (!s || s->Clearing || !s->World) return;
    s->World->RemoveJoint(r.get<const PhysicsConstraintHandle>(e).ConstraintIndex);
    s->Invalidate();
}

void OnDestroyPhysicsInput(entt::registry &r, entt::entity) {
    // Entity destruction can erase entries from reactive storage after component destruction signals.
    if (auto *s = r.ctx().find<PhysicsState>()) s->InputDirty = true;
}

rbp::WorldLimits Limits(const entt::registry &r) {
    const uint32_t colliders = uint32_t(r.view<const ColliderShape>().size());
    const uint32_t motions = uint32_t(r.view<const PhysicsMotion>().size());
    rbp::WorldLimits limits;
    limits.Bodies = std::max(1u, colliders + motions + 1);
    limits.Shapes = std::max(4u, 4 * colliders + motions + 4);
    const auto joints = uint32_t(r.view<const PhysicsJoint>().size());
    limits.Joints = std::max(8u, joints + joints / 2);
    limits.CompoundChildren = std::max(1u, 2 * colliders);
    uint64_t vertices = 1, triangles = 1;
    for (const auto [e, collider] : r.view<const ColliderShape>().each()) {
        const auto mesh = IsMeshBackedShape(collider.Shape) ? TryGetMesh(r, collider.MeshEntity) : std::nullopt;
        vertices += mesh && std::holds_alternative<physics::TriangleMesh>(collider.Shape) ? mesh->VertexCount() : rbp::MaxHullVertices;
        triangles += mesh ? uint64_t(mesh->TriangleIndexCount() / 3) : 4;
    }
    if (vertices * 3 > UINT32_MAX || triangles * 6 > UINT32_MAX) throw std::runtime_error("Physics geometry exceeds RBP pool indexing.");
    limits.ShapeVertices = uint32_t(vertices * 3);
    limits.HullFaces = std::max(1u, colliders * 384);
    limits.Triangles = uint32_t(triangles * 3);
    limits.BvhNodes = uint32_t(triangles * 6);
    return limits;
}

auto GeometryOverflows(const rbp::World &world) {
    const auto &o = world.Overflow;
    return std::array{o.Shapes, o.ShapeVertices, o.HullFaces, o.Triangles, o.BvhNodes, o.CompoundChildren};
}

physics::RbpBody CookBody(PhysicsState &s, const SceneInput &scene, entt::registry &r, entt::entity entity, const physics::RbpBody *previous = nullptr) {
    auto &world = *s.World;
    const auto &input = scene.Bodies.at(entity);
    const auto *motion = input.Motion ? &*input.Motion : nullptr;
    const auto &colliders = input.Colliders;
    std::vector<rbp::Index> shapes;
    shapes.reserve(colliders.size());
    try {
        for (auto collider : colliders) {
            const auto &leaf = scene.Colliders.at(collider);
            const auto &desc = leaf.Shape;
            const auto &transform = leaf.Local;
            auto local = PoseOf(transform);
            local.Position += rbp::Rotate(local.Orientation, ToRbp(desc.LocalOffset * transform.S));
            const auto mesh = IsMeshBackedShape(desc.Shape) ? TryGetMesh(r, desc.MeshEntity) : std::nullopt;
            const auto shape = physics::BuildRbpShape(world, desc.Shape, mesh ? &*mesh : nullptr, transform.S, local);
            shapes.push_back(shape);
            ApplyCollider(world.Shapes[shape], collider, leaf);
        }
        const auto body = physics::BuildRbpBody(world, shapes, input.Node, motion, &input.Velocity, input.Sensor, previous);
        for (auto shape : shapes) world.RemoveShape(shape);
        return body;
    } catch (...) {
        for (auto shape : shapes) world.RemoveShape(shape);
        throw;
    }
}

void BuildBody(PhysicsState &s, entt::registry &r, entt::entity entity) {
    if (s.Bodies.contains(entity)) return;
    const auto body = CookBody(s, s.Input, r, entity);
    s.Bodies.emplace(entity, body);
    if (s.Entities.size() <= body.Body) s.Entities.resize(body.Body + 1, null_entity);
    s.Entities[body.Body] = entity;
    r.emplace_or_replace<PhysicsBodyHandle>(entity, PhysicsBodyHandle{body.Body});
    if (s.Input.Bodies.at(entity).Motion) r.emplace_or_replace<BodyPoseCache>(entity, BodyPoseCache{{physics::RbpNodePose(body.InitialPose, body.Frame)}});
}

void BuildJoint(PhysicsState &s, entt::registry &r, entt::entity entity) {
    const auto it = s.Input.Joints.find(entity);
    if (it == s.Input.Joints.end() || !IsActiveJoint(it->second)) return;
    const auto &input = it->second;
    const auto &joint = input.Joint;
    const auto &def = *input.Definition;
    const auto owner = input.Owner, connected = input.ConnectedOwner;
    auto &world = *s.World;
    if (connected == null_entity && s.WorldAnchor == rbp::NoIndex) s.WorldAnchor = world.AddBody({});
    // KHR measures the connected frame in the joint node's frame. RBP measures A in B.
    rbp::JointDesc desc;
    desc.BodyA = connected == null_entity ? s.WorldAnchor : s.Bodies.at(connected).Body;
    desc.BodyB = s.Bodies.at(owner).Body;
    const auto a = PoseOf(*input.Connected), b = PoseOf(input.Node);
    desc.AtA = a.Position;
    desc.AtB = b.Position;
    desc.FrameA = a.Orientation;
    desc.FrameB = b.Orientation;
    desc.Collide = joint.EnableCollision;
    for (int axis = 0; axis < 3; ++axis) desc.Linear[axis] = rbp::AxisFree;
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

void Rebuild(entt::registry &r) {
    const profile::CpuScope scope{"PhysicsRebuild"};
    auto &s = r.ctx().get<PhysicsState>();
    ClearSimulation(s, r);
    if (!s.Solver) s.Solver.emplace(s.Context);
    s.World.emplace(s.Context, Limits(r));
    for (auto entity : SortedEntities(r.view<const PhysicsMotion>())) BuildBody(s, r, entity);
    for (auto entity : SortedEntities(r.view<const ColliderShape>()))
        if (s.Input.Bodies.contains(entity)) BuildBody(s, r, entity);
    for (auto entity : SortedEntities(r.view<const PhysicsJoint>())) BuildJoint(s, r, entity);
    s.JointUpdates.clear();
    s.Evaluate = true;
}

void Restart(PhysicsState &s, entt::registry &r) {
    const profile::CpuScope scope{"PhysicsReset"};
    ClearContacts(s, r);
    for (auto [entity, transform] : r.view<const Transform>().each())
        if (ParentOrNull(r, entity) == null_entity) UpdateWorldTransformRecursive(r, entity);
    for (const auto &[entity, body] : s.Bodies) {
        s.World->Poses[body.Body] = body.InitialPose;
        s.World->Velocities[body.Body] = body.InitialVelocity;
        if (auto *cache = r.try_get<BodyPoseCache>(entity)) cache->Frames = {physics::RbpNodePose(body.InitialPose, body.Frame)};
    }
    s.World->ResetDynamics();
    for (auto entity : s.JointUpdates) BuildJoint(s, r, entity);
    s.JointUpdates.clear();
    s.SensorFollowers.clear();
    for (const auto &[entity, body] : s.Bodies) {
        if (!r.all_of<TriggerTag>(entity) || r.all_of<PhysicsMotion>(entity)) continue;
        const auto owner = MotionOwner(r, GetParentEntity(r, entity));
        if (owner == null_entity) continue;
        const auto owner_body = s.Bodies.at(owner).Body;
        s.SensorFollowers.push_back({body.Body, owner_body, rbp::ComposePose(Inverse(s.World->Poses[owner_body]), s.World->Poses[body.Body])});
    }
    s.World->WeldStatic();
    s.Baked = s.CacheStartFrame;
    s.ContactFrames = {{r.ctx().get<PhysicsContactImpacts>(), r.ctx().get<PhysicsSustainedContacts>()}};
}

entt::entity EntityForBody(const PhysicsState &s, rbp::Index body) { return body < s.Entities.size() ? s.Entities[body] : null_entity; }
entt::entity Collider(uint64_t data) { return data ? entt::entity(uint32_t(data - 1)) : null_entity; }
rbp::float3 WorldPoint(const rbp::ContactSide &side) { return rbp::WorldPoint(side.InitialPose, side.Point); }
rbp::float3 PointVelocity(const rbp::ContactSide &side) { return side.Velocity.Linear + simd::cross(side.Velocity.Angular, rbp::Rotate(side.Pose.Orientation, side.Point)); }

void CollectSubstep(PhysicsState &s, entt::registry &r, std::map<ContactKey, ContactSum> &frame) {
    const profile::CpuScope scope{"PhysicsContacts"};
    auto events = s.World->TakeContactChanges();
    std::ranges::stable_sort(events, {}, Key);
    ++s.Substep;
    auto &impacts = r.ctx().get<PhysicsContactImpacts>().Events;
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

void StepSimulation(PhysicsState &s, entt::registry &r, float sim_dt, uint32_t substeps) {
    const profile::CpuScope scope{"PhysicsFrame"};
    auto &out = r.ctx().get<PhysicsSustainedContacts>();
    out.Active.clear();
    out.Step = ++s.ContactStep;
    r.ctx().get<PhysicsContactImpacts>().Events.clear();
    if (sim_dt <= 0) return;
    substeps = std::max(1u, substeps);
    s.Settings.DeltaTime = sim_dt / float(substeps);
    auto &world = *s.World;
    world.TrackContacts = !r.view<const ReportContacts>().empty();
    if (!s.CapturePath.empty()) {
        s.Capture.emplace(s.CapturePath, world, s.SensorFollowers);
        s.CapturePath.clear();
    }
    std::map<ContactKey, ContactSum> contacts;
    rbp::AdvanceResult completed;
    {
        const profile::CpuScope advance_scope{"RbpAdvance"};
        completed = s.Solver->Advance(world, s.Settings, substeps, s.SensorFollowers, [&](const rbp::StepResult &step) {
            if (s.Capture) s.Capture->Step(world, s.Settings, step);
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

void SyncBodyWorldTransform(entt::registry &r, entt::entity entity, const vec3 &pos, const quat &rot) {
    r.patch<WorldTransform>(entity, [&](WorldTransform &t) { t.P = pos; t.R = rot; });
    for (const auto child : Children{&r, entity}) UpdateWorldTransformRecursive(r, child);
}

void BakeFrame(entt::registry &r, entt::entity viewport, PhysicsState &s, uint32_t frame, float fps) {
    const auto &settings = r.get<const PhysicsSimulationSettings>(viewport);
    const float sim_dt = (fps > 0 ? 1.f / fps : 1.f / 60) * settings.TimeScale;
    StepSimulation(s, r, sim_dt, settings.SubstepsPerFrame);
    for (auto [entity, handle, cache] : r.view<const PhysicsBodyHandle, BodyPoseCache>().each()) {
        const auto &body = s.Bodies.at(entity);
        cache.Frames.push_back(physics::RbpNodePose(s.World->Poses[body.Body], body.Frame));
    }
    s.ContactFrames.push_back({std::move(r.ctx().get<PhysicsContactImpacts>()), std::move(r.ctx().get<PhysicsSustainedContacts>())});
    s.Baked = frame;
}

template<typename C>
void ClearDanglingRefs(entt::registry &r, entt::entity deleted, entt::entity C::*field) {
    for (auto [e, c] : r.view<C>().each())
        if (c.*field == deleted) r.patch<C>(e, [field](C &x) { x.*field = null_entity; });
}

template<typename C>
void ClearDanglingRefs(entt::registry &r, entt::entity deleted, std::vector<entt::entity> C::*field) {
    for (auto [e, c] : r.view<C>().each())
        if (std::ranges::contains(c.*field, deleted)) r.patch<C>(e, [field, deleted](C &x) { std::erase(x.*field, deleted); });
}

void UpdateSettings(entt::registry &r, entt::entity viewport, float fps) {
    auto &s = r.ctx().get<PhysicsState>();
    const auto &settings = r.get<const PhysicsSimulationSettings>(viewport);
    if (s.AppliedSettings == settings && s.CacheFps == fps) return;
    physics::ApplySimulationSettings(r, settings);
    s.AppliedSettings = settings;
    s.CacheFps = fps;
    s.Invalidate();
}

void ProcessChanges(entt::registry &r) {
    auto &s = r.ctx().get<PhysicsState>();
    const auto any = [&]<typename... T> { return (... || !reactive<T>(r).empty()); };
    if (!std::exchange(s.InputDirty, false) && !any.operator()<changes::PhysicsShape, changes::PhysicsMotion, changes::PhysicsPose, changes::PhysicsMaterial, changes::PhysicsTrigger, changes::PhysicsJoint, changes::PhysicsMaterialDef, changes::CollisionSystemDef, changes::CollisionFilterDef, changes::PhysicsJointDef, changes::PhysicsGeometry, changes::PhysicsHierarchy>()) return;
    for (auto e : reactive<changes::PhysicsMaterialDef>(r))
        if (!r.all_of<PhysicsMaterial>(e)) ClearDanglingRefs(r, e, &ColliderMaterial::PhysicsMaterialEntity);
    for (auto e : reactive<changes::CollisionSystemDef>(r))
        if (!r.all_of<CollisionSystem>(e)) {
            ClearDanglingRefs(r, e, &CollisionFilter::Systems);
            ClearDanglingRefs(r, e, &CollisionFilter::CollideSystems);
        }
    for (auto e : reactive<changes::CollisionFilterDef>(r))
        if (!r.all_of<CollisionFilter>(e)) {
            ClearDanglingRefs(r, e, &ColliderMaterial::CollisionFilterEntity);
            ClearDanglingRefs(r, e, &TriggerNodes::CollisionFilterEntity);
        }
    for (auto [entity, joint] : r.view<const PhysicsJoint>().each())
        if (joint.JointDefEntity != null_entity && !r.all_of<PhysicsJointDef>(joint.JointDefEntity))
            r.patch<PhysicsJoint>(entity, [](auto &j) { j.JointDefEntity = null_entity; });
    UpdateMasks(s, r);
    auto input = ReadScene(s, r);
    if (input.Bodies.empty()) {
        if (s.World) ClearSimulation(s, r);
        s.Input = std::move(input);
        return;
    }
    const auto joints = std::ranges::count_if(input.Joints, [](const auto &entry) { return IsActiveJoint(entry.second); });
    if (!s.World || RequiresRebuild(s.Input, input) || joints > s.World->Joints.Capacity) {
        s.Input = std::move(input);
        Rebuild(r);
        return;
    }
    std::set<entt::entity> recook, surfaces;
    for (const auto &[entity, leaf] : input.Colliders) {
        const auto &old = s.Input.Colliders.at(entity);
        if (old.Shape != leaf.Shape || old.Local != leaf.Local || (IsMeshBackedShape(leaf.Shape.Shape) && reactive<changes::PhysicsGeometry>(r).contains(leaf.Shape.MeshEntity))) recook.insert(leaf.Owner);
        if (old != leaf) surfaces.insert(leaf.Owner);
    }
    const bool changed = !recook.empty() || !surfaces.empty() || s.Input.Bodies != input.Bodies || s.Input.Joints != input.Joints;
    if (!changed) return;
    s.Invalidate();
    for (auto entity : r.view<const PhysicsConstraintHandle>()) {
        const auto it = input.Joints.find(entity);
        if (it == input.Joints.end() || !IsActiveJoint(it->second)) r.remove<PhysicsConstraintHandle>(entity);
    }
    std::set<rbp::Index> reframed;
    const auto overflows = GeometryOverflows(*s.World);
    try {
        for (const auto &[entity, next] : input.Bodies) {
            const bool replace = recook.contains(entity);
            if (!replace && s.Input.Bodies.at(entity) == next) continue;
            auto &body = s.Bodies.at(entity);
            const auto old_pose = body.InitialPose;
            const auto old_mass = s.World->Masses[body.Body];
            if (replace) body = CookBody(s, input, r, entity, &body);
            else physics::UpdateRbpBody(*s.World, body, next.Node, next.Motion ? &*next.Motion : nullptr, &next.Velocity);
            const auto mass = s.World->Masses[body.Body];
            if (replace || simd::any(old_pose.Position != body.InitialPose.Position) || simd::any(old_pose.Orientation != body.InitialPose.Orientation) || old_mass.InvMass != mass.InvMass || simd::any(old_mass.InvInertiaLocal != mass.InvInertiaLocal)) reframed.insert(body.Body);
        }
    } catch (...) {
        if (GeometryOverflows(*s.World) == overflows) throw;
        s.Input = std::move(input);
        Rebuild(r);
        return;
    }
    for (const auto &[entity, next] : input.Joints) {
        const auto old = s.Input.Joints.find(entity);
        bool update = old == s.Input.Joints.end() || old->second != next;
        if (const auto *handle = r.try_get<const PhysicsConstraintHandle>(entity)) {
            const auto &joint = s.World->Joints[handle->ConstraintIndex];
            update |= reframed.contains(joint.BodyA) || reframed.contains(joint.BodyB);
        }
        if (update) s.JointUpdates.insert(entity);
    }
    for (auto entity : surfaces) {
        if (recook.contains(entity)) continue;
        const auto &body = s.Bodies.at(entity);
        if (body.Shape == rbp::NoIndex) continue;
        const auto &compound = s.World->Shapes[body.Shape];
        for (uint32_t i = 0; i < compound.VertexCount; ++i) {
            auto &leaf = s.World->Shapes[s.World->Child(body.Shape, i)];
            const auto collider = entt::entity(uint32_t(leaf.UserData - 1));
            const auto &next = input.Colliders.at(collider);
            if (s.Input.Colliders.at(collider) != next) ApplyCollider(leaf, collider, next);
        }
    }
    s.Input = std::move(input);
}
} // namespace

namespace physics {
void CaptureReplay(entt::registry &r, const std::filesystem::path &path) {
    auto &s = r.ctx().get<PhysicsState>();
    s.CapturePath = path;
    s.Invalidate();
}

void ApplySimulationSettings(entt::registry &r, const PhysicsSimulationSettings &settings) {
    auto &step = r.ctx().get<PhysicsState>().Settings;
    step.Gravity = ToRbp(settings.Gravity);
    step.Iterations = std::max(1u, settings.SolverIterations);
}
std::optional<uint32_t> BakedThrough(const entt::registry &r) { return r.ctx().get<PhysicsState>().Baked; }
uint32_t BodyCount(const entt::registry &r) { return uint32_t(r.ctx().get<PhysicsState>().Bodies.size()); }
bool DoesFilterAllow(const entt::registry &r, entt::entity source, entt::entity target) {
    const auto &masks = r.ctx().get<PhysicsState>().Masks;
    const auto a = masks.find(source), b = masks.find(target);
    return a == masks.end() || b == masks.end() || (a->second.Layer & b->second.Collides) != 0;
}
bool AdvancePlayback(entt::registry &r, entt::entity viewport, int from_frame, int to_frame, int range_start_frame, int range_end_frame, float fps, bool cache_invalid) {
    auto &s = r.ctx().get<PhysicsState>();
    UpdateSettings(r, viewport, fps);
    if (cache_invalid || uint32_t(range_start_frame) != s.CacheStartFrame) s.Invalidate();
    s.CacheStartFrame = range_start_frame;
    s.CacheEndFrame = range_end_frame;
    if (s.Bodies.empty()) return false;
    if (!s.Baked) Restart(s, r);
    if (!std::exchange(s.Evaluate, false) && from_frame == to_frame) return false;
    BakeThrough(r, viewport, to_frame, fps);
    SamplePosesAtFrame(r, float(to_frame));
    const auto &contacts = s.ContactFrames[std::clamp(uint32_t(to_frame), s.CacheStartFrame, *s.Baked) - s.CacheStartFrame];
    r.ctx().get<PhysicsContactImpacts>() = contacts.Impacts;
    r.ctx().get<PhysicsSustainedContacts>() = contacts.Sustained;
    return true;
}

void BakeThrough(entt::registry &r, entt::entity viewport, int through_frame, float fps) {
    auto &s = r.ctx().get<PhysicsState>();
    if (s.Bodies.empty()) return;
    UpdateSettings(r, viewport, fps);
    if (!s.Baked) Restart(s, r);
    const uint32_t target = std::min(uint32_t(std::max(through_frame, int(s.CacheStartFrame))), s.CacheEndFrame);
    if (*s.Baked >= target) return;
    // Prediction records future contacts without publishing them to the audio timeline.
    auto impacts = std::move(r.ctx().get<PhysicsContactImpacts>());
    auto sustained = std::move(r.ctx().get<PhysicsSustainedContacts>());
    while (*s.Baked < target) BakeFrame(r, viewport, s, *s.Baked + 1, fps);
    r.ctx().get<PhysicsContactImpacts>() = std::move(impacts);
    r.ctx().get<PhysicsSustainedContacts>() = std::move(sustained);
}

void SamplePosesAtFrame(entt::registry &r, float frame) {
    auto &s = r.ctx().get<PhysicsState>();
    if (s.Bodies.empty() || !s.Baked) return;
    const float clamped = std::clamp(frame, float(s.CacheStartFrame), float(*s.Baked));
    const uint32_t lo = uint32_t(std::floor(clamped));
    const uint32_t hi = std::min(lo + 1, *s.Baked);
    const float t = clamped - float(lo);
    const auto lo_idx = lo - s.CacheStartFrame, hi_idx = hi - s.CacheStartFrame;
    // Update parents before restoring the independent poses of nested bodies.
    std::vector<std::pair<uint32_t, entt::entity>> entities;
    for (auto entity : r.view<const PhysicsBodyHandle, const BodyPoseCache>()) {
        uint32_t depth = 0;
        for (auto parent = ParentOrNull(r, entity); parent != null_entity; parent = ParentOrNull(r, parent)) ++depth;
        entities.emplace_back(depth, entity);
    }
    std::ranges::sort(entities);
    for (auto [depth, entity] : entities) {
        const auto &cache = r.get<const BodyPoseCache>(entity);
        const auto &a = cache.Frames[lo_idx], &b = cache.Frames[hi_idx];
        SyncBodyWorldTransform(r, entity, numeric::Mix(a.P, b.P, t), numeric::Slerp(a.R, b.R, t));
    }
}

void Init(entt::registry &r) {
    r.ctx().emplace<PhysicsState>();
    r.ctx().emplace<PhysicsContactImpacts>();
    r.ctx().emplace<PhysicsSustainedContacts>();
    r.on_destroy<PhysicsBodyHandle>().connect<&OnDestroyPhysicsBody>();
    r.on_destroy<PhysicsConstraintHandle>().connect<&OnDestroyPhysicsConstraint>();
    r.on_destroy<PhysicsJoint>().connect<&OnDestroyPhysicsInput>();
    r.on_destroy<PhysicsJointDef>().connect<&OnDestroyPhysicsInput>();
    r.on_destroy<SceneNode>().connect<&OnDestroyPhysicsInput>();
    track<changes::PhysicsMotion>(r).on<PhysicsMotion>(On::Create | On::Update | On::Destroy).on<PhysicsVelocity>(On::Create | On::Update | On::Destroy);
    track<changes::PhysicsShape>(r).on<ColliderShape>(On::Create | On::Update | On::Destroy);
    track<changes::PhysicsPose>(r).on<Transform>(On::Update);
    track<changes::PhysicsGeometry>(r).on<MeshGeometryDirty>(On::Create | On::Update).on<MeshPositionsChanged>(On::Create | On::Update);
    track<changes::PhysicsHierarchy>(r).on<SceneNode>(On::Update | On::Destroy).on<ParentInverse>(On::Create | On::Update | On::Destroy);
    track<changes::ColliderPolicy>(r).on<::ColliderPolicy>(On::Create | On::Update);
    track<changes::PhysicsMaterial>(r).on<ColliderMaterial>(On::Create | On::Update | On::Destroy);
    track<changes::PhysicsTrigger>(r).on<TriggerTag>(On::Create | On::Destroy);
    track<changes::PhysicsJoint>(r).on<PhysicsJoint>(On::Create | On::Update | On::Destroy);
    track<changes::PhysicsMaterialDef>(r).on<::PhysicsMaterial>(On::Create | On::Update | On::Destroy);
    track<changes::CollisionSystemDef>(r).on<CollisionSystem>(On::Create | On::Update | On::Destroy);
    track<changes::CollisionFilterDef>(r).on<CollisionFilter>(On::Create | On::Update | On::Destroy);
    track<changes::PhysicsJointDef>(r).on<::PhysicsJointDef>(On::Create | On::Update | On::Destroy);

    RegisterComponentEventHandler(r, ProcessChanges);
}
void Deinit(entt::registry &r) { r.ctx().erase<PhysicsState>(); }
void Clear(entt::registry &r) {
    if (auto *s = r.ctx().find<PhysicsState>()) ClearSimulation(*s, r);
}
} // namespace physics
