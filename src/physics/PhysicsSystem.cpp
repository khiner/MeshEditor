#include "PhysicsSystem.h"
#include "ColliderUpdate.h"
#include "PhysicsChanges.h"
#include "PhysicsContact.h"
#include "Profile.h"
#include "RbpBody.h"
#include "RbpShape.h"
#include "Solver.h"
#include "SortUnique.h"
#include "TransformMath.h"
#include "mesh/Mesh.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/Neighborhood.h"
#include "procedure/Cook.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "viewport/ViewportEvents.h"

#include <map>
#include <set>

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
struct BodyInput {
    Transform Node;
    state::Entity Parent = state::Null;
    std::optional<PhysicsMotion> Motion;
    PhysicsVelocity Velocity;
    bool Sensor = false;
    uint64_t Scenes = 0; // The bits of the scenes the body's root is in.
    std::vector<state::Entity> Colliders;
    bool operator==(const BodyInput &) const = default;
};
struct ColliderInput {
    ColliderShape Shape;
    Transform Local{};
    PhysicsMaterial Material{};
    uint32_t Layer = ~0u, Collides = ~0u;
    bool operator==(const ColliderInput &) const = default;
};
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
} // namespace
struct PhysicsState {
    // The solver runs on Metal 4 devices only, so this stays empty on other devices and no world is built.
    std::optional<rbp::mtl::Context> Context;
    std::optional<rbp::Solver> Solver;
    std::optional<rbp::World> World;
    rbp::StepSettings Settings;
    std::map<state::Entity, BodyRecord> Bodies;
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
    // Live scenes in entity index order, each scene's position its bit in a body's scene word.
    std::vector<state::Entity> Scenes;
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

    void Invalidate() { Baked.reset(); }
};
// Removes all bodies, constraints, and contacts while preserving initialization.
void Reset(PhysicsState &, state::Registry &);
namespace {

rbp::Pose PoseOf(const Transform &t) { return rbp::At(ToRbp(t.P), ToRbp(Normalize(t.R))); }
state::Entity MotionOwner(const state::Registry &r, state::Entity e) {
    return FindAncestorIf(r, e, [&](auto node) { return r.all_of<PhysicsMotion>(node); });
}

void UpdateMasks(PhysicsState &s, const state::Registry &r) {
    std::map<state::Entity, uint32_t> bits;
    for (const auto e : SortedEntities(r.view<const CollisionSystem>())) {
        if (bits.size() == 32) throw std::runtime_error("RBP supports at most 32 collision systems.");
        bits.emplace(e, uint32_t(1) << bits.size());
    }
    const auto mask = [&](const auto &systems) {
        uint32_t result = 0;
        for (const auto e : systems) {
            if (const auto it = bits.find(e); it != bits.end()) result |= it->second;
        }
        return result;
    };
    s.Masks.clear();
    for (auto [e, filter] : r.view<const CollisionFilter>().each()) {
        const auto collide = filter.Mode == CollideMode::All ? ~0u : filter.Mode == CollideMode::Allowlist ? mask(filter.CollideSystems) :
                                                                                                             ~mask(filter.CollideSystems);
        s.Masks.emplace(e, rbp::CollisionMask{mask(filter.Systems), collide});
    }
    s.Scenes = r.view<const Scene>() | std::views::take(MaxScenes) | std::ranges::to<std::vector>();
}

uint64_t SceneWord(const PhysicsState &s, const state::Registry &r, state::Entity e) {
    const auto *membership = RootMembership(r, e);
    if (!membership) return 0;
    uint64_t word = 0;
    for (const auto scene : membership->Scenes) {
        if (const auto it = std::ranges::find(s.Scenes, scene); it != s.Scenes.end()) word |= uint64_t(1) << (it - s.Scenes.begin());
    }
    return word;
}

Transform ComposeAuthored(const Transform &parent, Transform result) {
    // A uniform parent scale multiplies exactly, without remeasuring quaternion matrix columns.
    if (parent.S.x == parent.S.y && parent.S.y == parent.S.z) {
        result.P *= parent.S;
        result.S *= parent.S;
    } else {
        result = ToTransform(ToMatrix(Transform{.S = parent.S}) * ToMatrix(result));
    }
    result.P = parent.P + Rotate(parent.R, result.P);
    result.R = Normalize(parent.R * result.R);
    return result;
}

Transform AuthoredWorld(const state::Registry &r, state::Entity e) {
    const auto *local = r.try_get<const Transform>(e);
    if (!local) return {};
    const auto parent = ParentOrNull(r, e);
    return parent != state::Null ? ComposeAuthored(AuthoredWorld(r, parent), *local) : *local;
}

std::optional<BodyInput> ReadBody(const PhysicsState &s, const state::Registry &r, state::Entity entity, std::vector<ColliderInput> &leaves) {
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
    body.Scenes = SceneWord(s, r, entity);
    leaves.reserve(body.Colliders.size());
    for (const auto collider : body.Colliders) {
        auto &leaf = leaves.emplace_back(ColliderInput{.Shape = r.get<const ColliderShape>(collider), .Local = local_transform(collider)});
        if (const auto *material = r.try_get<const ColliderMaterial>(collider)) {
            if (const auto *definition = r.try_get<const PhysicsMaterial>(material->PhysicsMaterialEntity)) leaf.Material = *definition;
            if (const auto it = s.Masks.find(material->CollisionFilterEntity); it != s.Masks.end()) {
                leaf.Layer = it->second.Layer;
                leaf.Collides = it->second.Collides;
            }
        }
        leaf.Material.Name.clear();
    }
    return body;
}

JointInput ReadJoint(const PhysicsState &s, const state::Registry &r, state::Entity entity, const PhysicsJoint &joint) {
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
    // Each leaf carries its whole mask, so the body's own mask goes unused.
    shape.HasFilter = true;
    shape.Mask = {input.Layer, input.Collides};
}

void ClearContacts(PhysicsState &s, state::Registry &r) {
    s.Contacts.clear();
    s.ContactFrames.clear();
    r.Resources.get<PhysicsContactImpacts>().Events.clear();
    auto &sustained = r.Resources.get<PhysicsSustainedContacts>();
    sustained.Active.clear();
    sustained.Step = ++s.ContactStep;
}

} // namespace
void Reset(PhysicsState &s, state::Registry &r) {
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
namespace {

void OnDestroyPhysicsConstraint(state::Registry &r, state::Entity e) {
    auto *s = r.Resources.find<PhysicsState>();
    if (!s || s->Clearing || !s->World) return;
    s->World->RemoveJoint(r.get<const PhysicsConstraintHandle>(e).ConstraintIndex);
    s->Invalidate();
}

// A destroyed entity's reactive entries are discarded, so its motion, collider and joint inputs record their entity here.
void OnDestroyPhysicsInput(state::Registry &r, state::Entity e) {
    if (auto *s = r.Resources.find<PhysicsState>()) s->Destroyed.push_back(e);
}

// Every pool reserves twice its current need, with room for at least HeadroomColliders hull colliders, so added bodies build in place.
constexpr uint64_t HeadroomColliders = 16;

rbp::WorldLimits Limits(const state::Registry &r) {
    const uint32_t colliders = uint32_t(r.view<const ColliderShape>().size());
    const uint32_t motions = uint32_t(r.view<const PhysicsMotion>().size());
    const auto joints = uint32_t(r.view<const PhysicsJoint>().size());
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

// The record a leaf's mesh-backed shape reads, and whether the shape reads its triangles.
std::optional<std::pair<uint32_t, bool>> LeafMesh(const state::Registry &r, const ColliderInput &leaf) {
    const auto &shape = leaf.Shape;
    const auto mesh = IsMeshBackedShape(shape.Shape) ? TryGetMesh(r, shape.MeshEntity) : std::nullopt;
    if (!mesh) return {};
    return std::pair{mesh->GetStoreId(), std::holds_alternative<physics::TriangleMesh>(shape.Shape)};
}

// The ordinal positions, and the ordinal triangles a triangle mesh shape reads, of the records that leaves' mesh-backed shapes read.
struct LeafMeshes {
    std::map<uint32_t, std::vector<vec3>> Positions;
    std::map<uint32_t, std::vector<uint32_t>> Triangles;
};
// Reads the records of the leaves of every list in one gather each for positions and triangles.
LeafMeshes ReadLeafMeshes(const state::Registry &r, auto &&leaf_lists) {
    std::vector<uint32_t> positioned, triangulated;
    for (const auto &leaves : leaf_lists) {
        for (const auto &leaf : leaves) {
            if (const auto mesh = LeafMesh(r, leaf)) {
                positioned.push_back(mesh->first);
                if (mesh->second) triangulated.push_back(mesh->first);
            }
        }
    }
    SortUnique(positioned);
    SortUnique(triangulated);
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    auto positions = OrdinalPositions(meshes, pipelines, positioned);
    auto triangles = OrdinalTriangles(meshes, pipelines, triangulated);
    LeafMeshes read;
    for (uint32_t i = 0; i < positioned.size(); ++i) read.Positions.emplace(positioned[i], std::move(positions[i]));
    for (uint32_t i = 0; i < triangulated.size(); ++i) read.Triangles.emplace(triangulated[i], std::move(triangles[i]));
    return read;
}

// Builds the leaves' shapes into `shapes`, which the caller removes, each holding its collider's material and filter.
void BuildLeafShapes(rbp::World &world, const state::Registry &r, const LeafMeshes &meshes, const BodyInput &input, std::span<const ColliderInput> leaves, std::vector<rbp::Index> &shapes) {
    shapes.reserve(leaves.size());
    for (uint32_t i = 0; i < leaves.size(); ++i) {
        const auto &leaf = leaves[i];
        const auto &desc = leaf.Shape;
        auto local = PoseOf(leaf.Local);
        local.Position += rbp::Rotate(local.Orientation, ToRbp(desc.LocalOffset * leaf.Local.S));
        std::span<const vec3> positions;
        std::span<const uint32_t> triangles;
        if (const auto mesh = LeafMesh(r, leaf)) {
            positions = meshes.Positions.at(mesh->first);
            if (mesh->second) triangles = meshes.Triangles.at(mesh->first);
        }
        const auto shape = physics::BuildRbpShape(world, desc.Shape, positions, triangles, leaf.Local.S, local);
        shapes.push_back(shape);
        ApplyCollider(world.Shapes[shape], input.Colliders[i], leaf);
    }
}

physics::RbpBody CookBody(PhysicsState &s, const state::Registry &r, const LeafMeshes &meshes, const BodyInput &input, std::span<const ColliderInput> leaves, std::optional<const physics::RbpBody &> previous = {}) {
    auto &world = *s.World;
    std::vector<rbp::Index> shapes;
    try {
        BuildLeafShapes(world, r, meshes, input, leaves, shapes);
        const auto body = physics::BuildRbpBody(world, shapes, input.Node, input.Motion, input.Velocity, input.Sensor, previous);
        world.RemoveShapes(shapes);
        return body;
    } catch (...) {
        world.RemoveShapes(shapes);
        throw;
    }
}

// Gives the bodies whose colliders fit or cook from time-dependent meshes the collider geometry of the frame.
void ShapeTimeBodies(PhysicsState &s, state::Registry &r, int frame) {
    if (r.view<const procedure::TimeReads>().empty()) return;
    std::vector<state::Entity> bodies, meshes;
    for (const auto mesh : TimeDependentMeshes(r)) {
        for (const auto collider : MeshCollidersOf(r, mesh)) {
            if (const auto owner = s.LeafOwners.find(collider); owner != s.LeafOwners.end()) {
                bodies.push_back(owner->second);
                meshes.push_back(mesh);
            }
        }
    }
    if (bodies.empty()) return;
    SortUnique(bodies);
    SortUnique(meshes);
    // A procedure without a result at the frame leaves its mesh without geometry.
    std::vector<std::vector<vec3>> positions(meshes.size());
    std::vector<std::vector<uint32_t>> triangles(meshes.size());
    ReadAtFrame(r, meshes, frame, [&](std::span<const uint32_t> records) {
        const auto &store = r.Resources.get<const MeshStore>();
        const auto &pipelines = GetMeshPipelines(r);
        const auto valid = std::views::iota(0u, uint32_t(records.size())) | std::views::filter([&](uint32_t i) { return records[i] != InvalidOffset; }) | std::ranges::to<std::vector>();
        const auto ids = valid | std::views::transform([&](uint32_t i) { return records[i]; }) | std::ranges::to<std::vector>();
        auto read_positions = OrdinalPositions(store, pipelines, ids);
        auto read_triangles = OrdinalTriangles(store, pipelines, ids);
        for (uint32_t k = 0; k < valid.size(); ++k) {
            positions[valid[k]] = std::move(read_positions[k]);
            triangles[valid[k]] = std::move(read_triangles[k]);
        }
    });
    // Each time leaf reads its mesh's geometry at the frame under the drawn record's id.
    LeafMeshes frame_meshes;
    std::map<state::Entity, AABB> bounds;
    for (uint32_t i = 0; i < meshes.size(); ++i) {
        const auto drawn = TryGetMesh(r, meshes[i])->GetStoreId();
        frame_meshes.Positions.emplace(drawn, positions[i]);
        frame_meshes.Triangles.emplace(drawn, triangles[i]);
        if (positions[i].empty()) continue;
        AABB box{positions[i].front(), positions[i].front()};
        for (const auto &p : positions[i]) box = {Min(box.Min, p), Max(box.Max, p)};
        bounds.emplace(meshes[i], box);
    }
    auto &world = *s.World;
    for (const auto entity : bodies) {
        auto &record = s.Bodies.at(entity);
        std::vector<ColliderInput> leaves;
        auto input = ReadBody(s, r, entity, leaves);
        if (!input || input->Colliders != record.Input.Colliders) continue;
        for (uint32_t i = 0; i < leaves.size(); ++i) {
            const auto *listed = r.try_get<const ColliderListedUnder>(input->Colliders[i]);
            if (!listed || !std::ranges::binary_search(meshes, listed->MeshEntity)) continue;
            const auto box = bounds.find(listed->MeshEntity);
            if (auto shape = DeriveColliderShape(r, input->Colliders[i], box != bounds.end() ? std::optional{box->second} : std::nullopt)) leaves[i].Shape = std::move(*shape);
        }
        std::vector<rbp::Index> shapes;
        try {
            BuildLeafShapes(world, r, frame_meshes, *input, leaves, shapes);
            physics::ReshapeRbpBody(world, record.Cooked, shapes);
        } catch (...) {
            world.RemoveShapes(shapes);
            throw;
        }
        world.RemoveShapes(shapes);
    }
}

void ReleaseLeaves(PhysicsState &s, state::Entity body, std::span<const state::Entity> colliders) {
    for (const auto collider : colliders) {
        if (const auto leaf = s.LeafOwners.find(collider); leaf != s.LeafOwners.end() && leaf->second == body) s.LeafOwners.erase(leaf);
    }
}

void BuildBody(PhysicsState &s, state::Registry &r, const LeafMeshes &meshes, state::Entity entity, BodyInput input, std::vector<ColliderInput> leaves) {
    const auto body = CookBody(s, r, meshes, input, leaves);
    s.World->SetBodyScope(body.Body, input.Scenes);
    for (const auto collider : input.Colliders) s.LeafOwners[collider] = entity;
    if (s.Entities.size() <= body.Body) s.Entities.resize(body.Body + 1, state::Null);
    s.Entities[body.Body] = entity;
    r.emplace_or_replace<PhysicsBodyHandle>(entity);
    if (input.Motion) r.emplace_or_replace<BodyPoseCache>(entity, BodyPoseCache{{physics::RbpNodePose(body.InitialPose, body.Frame)}});
    s.Bodies.emplace(entity, BodyRecord{body, std::move(input), std::move(leaves)});
}

// A body as its components describe it, and whether its shapes cook again.
struct Pending {
    state::Entity Entity;
    BodyInput Input;
    std::vector<ColliderInput> Leaves;
    bool Recook = false;
};

void BuildBodies(PhysicsState &s, state::Registry &r) {
    std::vector<Pending> pending;
    std::set<state::Entity> read;
    std::vector<ColliderInput> leaves;
    const auto read_body = [&](state::Entity entity) {
        if (!read.insert(entity).second) return;
        if (auto input = ReadBody(s, r, entity, leaves)) pending.push_back({entity, std::move(*input), std::move(leaves)});
    };
    for (const auto entity : SortedEntities(r.view<const PhysicsMotion>())) read_body(entity);
    for (const auto entity : SortedEntities(r.view<const ColliderShape>())) read_body(entity);
    const auto meshes = ReadLeafMeshes(r, pending | std::views::transform(&Pending::Leaves));
    for (auto &[entity, input, body_leaves, _] : pending) BuildBody(s, r, meshes, entity, std::move(input), std::move(body_leaves));
}

void BuildJoint(PhysicsState &s, state::Registry &r, state::Entity entity) {
    const auto it = s.Joints.find(entity);
    if (it == s.Joints.end() || !IsActiveJoint(it->second)) return;
    const auto &input = it->second;
    const auto &joint = input.Joint;
    const auto &def = *input.Definition;
    const auto owner = input.Owner, connected = input.ConnectedOwner;
    auto &world = *s.World;
    if (connected == state::Null && s.WorldAnchor == rbp::NoIndex) s.WorldAnchor = world.AddBody({});
    // KHR measures the connected frame in the joint node's frame.
    // RBP measures A in B.
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
        for (const auto axis : axes) {
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
            } else {
                modes[axis] = rbp::AxisLimited;
            }
            (linear ? desc.LinearLimitLow : desc.LimitLow)[axis] = low;
            (linear ? desc.LinearLimitHigh : desc.LimitHigh)[axis] = high;
            (linear ? desc.LinearStiffness : desc.AngularStiffness)[axis] = limit.Stiffness.value_or(INFINITY);
            (linear ? desc.LinearDamping : desc.AngularDamping)[axis] = limit.Damping;
        };
        if (std::popcount(mask) > 1 && !(low == 0 && high == 0)) {
            const auto axis = uint32_t(std::countr_zero(mask));
            configure(axis, true);
            (linear ? desc.LinearLimitAxes : desc.AngularLimitAxes)[axis] = mask;
        } else {
            for (uint32_t axis = 0; axis < 3; ++axis) {
                if (mask & (1u << axis)) configure(axis, false);
            }
        }
    }
    for (const auto &drive : def.Drives) {
        if (drive.Axis > 2) throw std::invalid_argument("A physics joint drive axis must be X, Y or Z.");
        const bool linear = drive.Type == PhysicsDriveType::Linear;
        float mass = 1;
        if (drive.Mode == PhysicsDriveMode::Acceleration) {
            const auto axis = rbp::Rotate(b.Orientation, rbp::float3{drive.Axis == 0 ? 1.f : 0, drive.Axis == 1 ? 1.f : 0, drive.Axis == 2 ? 1.f : 0});
            float inverse = 0;
            for (const auto body : {desc.BodyA, desc.BodyB}) {
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

void Rebuild(state::Registry &r) {
    const profile::CpuScope scope{"PhysicsRebuild"};
    auto &s = r.Resources.get<PhysicsState>();
    Reset(s, r);
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

const std::vector<state::Entity> &SampleOrder(PhysicsState &s, const state::Registry &r) {
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

void Restart(PhysicsState &s, state::Registry &r) {
    const profile::CpuScope scope{"PhysicsReset"};
    ShapeTimeBodies(s, r, int(s.CacheStartFrame));
    ClearContacts(s, r);
    RecomputeWorldTransforms(r, SampleOrder(s, r), {}, {});
    for (const auto &[entity, record] : s.Bodies) {
        const auto &body = record.Cooked;
        s.World->Poses[body.Body] = body.InitialPose;
        s.World->Velocities[body.Body] = body.InitialVelocity;
        if (auto *cache = r.try_edit<BodyPoseCache>(entity)) cache->Frames = {physics::RbpNodePose(body.InitialPose, body.Frame)};
    }
    s.World->ResetDynamics();
    for (const auto entity : s.JointUpdates) BuildJoint(s, r, entity);
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
    s.ContactFrames = {{r.Resources.get<PhysicsContactImpacts>(), r.Resources.get<PhysicsSustainedContacts>()}};
}

state::Entity EntityForBody(const PhysicsState &s, rbp::Index body) { return body < s.Entities.size() ? s.Entities[body] : state::Null; }
state::Entity Collider(uint64_t data) { return data ? state::Entity(uint32_t(data - 1)) : state::Null; }
rbp::float3 WorldPoint(const rbp::ContactSide &side) { return rbp::WorldPoint(side.InitialPose, side.Point); }
rbp::float3 PointVelocity(const rbp::ContactSide &side) { return side.Velocity.Linear + simd::cross(side.Velocity.Angular, rbp::Rotate(side.Pose.Orientation, side.Point)); }

void CollectSubstep(PhysicsState &s, state::Registry &r, std::map<ContactKey, ContactSum> &frame) {
    const profile::CpuScope scope{"PhysicsContacts"};
    auto events = s.World->TakeContactChanges();
    std::ranges::stable_sort(events, {}, Key);
    ++s.Substep;
    auto &impacts = r.Resources.get<PhysicsContactImpacts>().Events;
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
            for (const bool side_a : {true, false}) {
                impacts.push_back({
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
    }
    std::erase_if(s.Contacts, [&](const auto &entry) { return entry.second.Seen != s.Substep; });
}

void StepSimulation(PhysicsState &s, state::Registry &r, float sim_dt, uint32_t substeps) {
    const profile::CpuScope scope{"PhysicsFrame"};
    auto &out = r.Resources.get<PhysicsSustainedContacts>();
    out.Active.clear();
    out.Step = ++s.ContactStep;
    r.Resources.get<PhysicsContactImpacts>().Events.clear();
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
        const auto it = s.Contacts.find(key);
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

void BakeFrame(state::Registry &r, PhysicsState &s, uint32_t frame, float fps) {
    // Frame k simulates with the collider geometry of frame k.
    ShapeTimeBodies(s, r, int(frame));
    const auto &settings = r.get<const PhysicsSimulationSettings>(state::Document);
    const float sim_dt = (fps > 0 ? 1.f / fps : 1.f / 60) * settings.TimeScale;
    StepSimulation(s, r, sim_dt, settings.SubstepsPerFrame);
    for (auto [entity, cache] : r.view<const PhysicsBodyHandle, BodyPoseCache>().each()) {
        const auto &body = s.Bodies.at(entity).Cooked;
        cache.Frames.push_back(physics::RbpNodePose(s.World->Poses[body.Body], body.Frame));
    }
    s.ContactFrames.push_back({std::move(r.Resources.get<PhysicsContactImpacts>()), std::move(r.Resources.get<PhysicsSustainedContacts>())});
    s.Baked = frame;
}

void UpdateSettings(state::Registry &r, float fps) {
    auto &s = r.Resources.get<PhysicsState>();
    const auto &settings = r.get<const PhysicsSimulationSettings>(state::Document);
    if (s.AppliedSettings == settings && s.CacheFps == fps) return;
    physics::ApplySimulationSettings(r, settings);
    s.AppliedSettings = settings;
    s.CacheFps = fps;
    s.Invalidate();
}
} // namespace

namespace physics {
void ProcessChanges(state::Registry &r) {
    const auto &changes = r.Resources.get<const PhysicsChanges>();
    auto &s = r.Resources.get<PhysicsState>();
    const auto &inputs = changes.Input, &moved = changes.Transform, &geometry = changes.Geometry;
    const bool definitions = AnyChanged(changes.MaterialDef, changes.CollisionSystemDef, changes.CollisionFilterDef);
    const bool scenes = !changes.Scenes.empty();
    r.Resources.get<PhysicsChanges>().Scenes.clear();
    if (inputs.empty() && moved.empty() && geometry.empty() && !definitions && !scenes && s.Destroyed.empty()) return;
    const auto destroyed = std::exchange(s.Destroyed, {});
    if (r.view<const PhysicsMotion>().empty() && r.view<const ColliderShape>().empty()) {
        if (s.World) Reset(s, r);
        return;
    }
    if (!s.World) {
        Rebuild(r);
        return;
    }
    if (definitions || scenes) UpdateMasks(s, r);

    // `reached` contains every node the walk visits and every destroyed input, which the joints below read.
    std::vector<state::Entity> candidates, reached;
    const auto touch = [&](state::Entity e) {
        if (const auto owner = s.LeafOwners.find(e); owner != s.LeafOwners.end()) candidates.push_back(owner->second);
        if (s.Bodies.contains(e)) candidates.push_back(e);
        if (!r.valid(e) || !r.any_of<PhysicsMotion, ColliderShape>(e)) return;
        candidates.push_back(e);
        if (const auto owner = MotionOwner(r, e); owner != state::Null) candidates.push_back(owner);
    };
    const auto touch_subtree = [&](this const auto &self, state::Entity e) -> void {
        touch(e);
        reached.push_back(e);
        if (!r.valid(e)) return;
        for (const auto child : Children{&r, e}) self(child);
    };
    for (const auto e : inputs) touch_subtree(e);
    for (const auto e : moved) touch_subtree(e);
    for (const auto e : destroyed) touch(e);
    reached.append_range(destroyed);
    // A body takes the geometry that the timeline's frame or time alone changed in its bake, frame by frame.
    const auto &time_cooks = r.Resources.get<const TimeCooks>().Meshes;
    const auto time_leaf = [&](state::Entity collider) {
        const auto *listed = r.valid(collider) ? r.try_get<const ColliderListedUnder>(collider) : nullptr;
        return listed && std::ranges::binary_search(time_cooks, listed->MeshEntity) && s.LeafOwners.contains(collider);
    };
    for (const auto mesh_entity : geometry) {
        for (const auto collider : MeshCollidersOf(r, mesh_entity)) {
            if (!time_leaf(collider)) touch(collider);
        }
    }
    if (definitions) {
        for (const auto e : r.view<const ColliderMaterial>()) touch(e);
    }
    if (scenes) {
        for (const auto e : s.Bodies | std::views::keys) touch(e);
    }
    SortUnique(candidates);

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
        if (!input || input->Sensor != it->second.Input.Sensor) {
            removed.push_back(e);
            if (input) added.push_back({e, std::move(*input), std::move(leaves)});
            continue;
        }
        const auto &record = it->second;
        // Shapes that the time alone changed leave the body as its bake holds it.
        if (*input == record.Input && leaves.size() == record.Leaves.size() && std::ranges::all_of(std::views::iota(0u, uint32_t(leaves.size())), [&](uint32_t i) {
                auto leaf = leaves[i];
                if (time_leaf(input->Colliders[i])) leaf.Shape = record.Leaves[i].Shape;
                return leaf == record.Leaves[i];
            })) continue;
        const bool recook = input->Colliders != record.Input.Colliders || std::ranges::any_of(leaves, mesh_edited) ||
            !std::ranges::equal(leaves, record.Leaves, [](const auto &a, const auto &b) { return a.Shape == b.Shape && a.Local == b.Local; });
        if (recook || *input != record.Input || leaves != record.Leaves) changed.push_back({e, std::move(*input), std::move(leaves), recook});
    }
    // Motion bodies join first, each group in entity order, as a full build adds them.
    std::ranges::stable_partition(added, [](const Pending &p) { return p.Input.Motion.has_value(); });

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

    std::vector<std::span<const ColliderInput>> cooked;
    for (const auto &pending : changed) {
        if (pending.Recook) cooked.emplace_back(pending.Leaves);
    }
    for (const auto &pending : added) cooked.emplace_back(pending.Leaves);
    const auto meshes = ReadLeafMeshes(r, cooked);

    std::set<rbp::Index> reframed;
    const auto overflows = PoolOverflows(*s.World);
    try {
        for (auto &[e, input, next_leaves, recook] : changed) {
            auto &record = s.Bodies.at(e);
            auto &body = record.Cooked;
            const auto old_pose = body.InitialPose;
            const auto old_mass = s.World->Masses[body.Body];
            if (input.Scenes != record.Input.Scenes) s.World->SetBodyScope(body.Body, input.Scenes);
            if (recook) {
                body = CookBody(s, r, meshes, input, next_leaves, body);
            } else {
                if (input != record.Input) physics::UpdateRbpBody(*s.World, body, input.Node, input.Motion, input.Velocity);
                // Compound children follow the collider order, so each changed leaf updates its surface in place.
                for (uint32_t i = 0; i < next_leaves.size(); ++i) {
                    if (next_leaves[i] != record.Leaves[i]) ApplyCollider(s.World->Shapes[s.World->Child(body.Shape, i)], input.Colliders[i], next_leaves[i]);
                }
            }
            const auto mass = s.World->Masses[body.Body];
            if (recook || simd::any(old_pose.Position != body.InitialPose.Position) || simd::any(old_pose.Orientation != body.InitialPose.Orientation) || old_mass.InvMass != mass.InvMass || simd::any(old_mass.InvInertiaLocal != mass.InvInertiaLocal)) reframed.insert(body.Body);
            if (input.Motion && !record.Input.Motion) r.emplace<BodyPoseCache>(e, BodyPoseCache{{physics::RbpNodePose(body.InitialPose, body.Frame)}});
            else if (!input.Motion && record.Input.Motion && r.remove<BodyPoseCache>(e)) released.push_back(e);
            ReleaseLeaves(s, e, record.Input.Colliders);
            for (const auto collider : input.Colliders) s.LeafOwners[collider] = e;
            record.Input = std::move(input);
            record.Leaves = std::move(next_leaves);
        }
        for (auto &[e, input, next_leaves, _] : added) BuildBody(s, r, meshes, e, std::move(input), std::move(next_leaves));
    } catch (...) {
        if (PoolOverflows(*s.World) == overflows) throw;
        Rebuild(r);
        return;
    }

    const auto reach_subtree = [&](this const auto &self, state::Entity e) -> void {
        reached.push_back(e);
        for (const auto child : Children{&r, e}) self(child);
    };
    for (const auto e : removed) {
        if (r.valid(e)) reach_subtree(e);
    }
    for (const auto &pending : added) reach_subtree(pending.Entity);
    SortUnique(reached);

    const auto reaches = [&](state::Entity e) { return std::ranges::binary_search(reached, e); };
    const auto left = [&](state::Entity e) { return std::ranges::binary_search(removed, e); };
    std::vector<state::Entity> reread, reframed_joints;
    for (const auto &[entity, input] : s.Joints) {
        const auto connected = input.Joint.ConnectedNode;
        if (reaches(connected) || reaches(input.Joint.JointDefEntity) || left(input.Owner) || left(input.ConnectedOwner) || input.Connected.has_value() != r.valid(connected)) reread.push_back(entity);
        if (const auto *handle = reframed.empty() || !r.valid(entity) ? nullptr : r.try_get<const PhysicsConstraintHandle>(entity)) {
            const auto &joint = s.World->Joints[handle->ConstraintIndex];
            if (reframed.contains(joint.BodyA) || reframed.contains(joint.BodyB)) reframed_joints.push_back(entity);
        }
    }
    for (const auto e : reached) {
        if (s.Joints.contains(e) || (r.valid(e) && r.all_of<PhysicsJoint>(e))) reread.push_back(e);
    }
    SortUnique(reread);
    std::vector<state::Entity> updated;
    bool erased = false;
    for (const auto entity : reread) {
        const auto *joint = r.valid(entity) ? r.try_get<const PhysicsJoint>(entity) : nullptr;
        const auto it = s.Joints.find(entity);
        if (!joint) {
            if (it != s.Joints.end()) {
                s.Joints.erase(it);
                erased = true;
            }
            continue;
        }
        auto next = ReadJoint(s, r, entity, *joint);
        if (it == s.Joints.end()) s.Joints.emplace(entity, std::move(next));
        else if (it->second != next) it->second = std::move(next);
        else continue;
        updated.push_back(entity);
    }
    if (!updated.empty() && uint32_t(std::ranges::count_if(s.Joints, [](const auto &entry) { return IsActiveJoint(entry.second); })) > s.World->Joints.Capacity) {
        Rebuild(r);
        return;
    }
    if (!released.empty()) RecomputeWorldTransforms(r, released, r.view<const BodyPoseCache>() | std::ranges::to<std::vector>(), {});
    if (removed.empty() && changed.empty() && added.empty() && updated.empty() && !erased) return;
    s.Invalidate();
    s.SampleOrder.reset();
    for (const auto entity : reread) {
        const auto *handle = r.valid(entity) ? r.try_get<const PhysicsConstraintHandle>(entity) : nullptr;
        if (!handle) continue;
        const auto it = s.Joints.find(entity);
        const bool active = it != s.Joints.end() && IsActiveJoint(it->second);
        if (active && s.World->Joints[handle->ConstraintIndex].Active) continue;
        r.remove<PhysicsConstraintHandle>(entity);
        if (active) s.JointUpdates.insert(entity);
    }
    s.JointUpdates.insert_range(updated);
    s.JointUpdates.insert_range(reframed_joints);
}

void ApplySimulationSettings(state::Registry &r, const PhysicsSimulationSettings &settings) {
    auto &step = r.Resources.get<PhysicsState>().Settings;
    step.Gravity = ToRbp(settings.Gravity);
    step.Iterations = std::max(1u, settings.SolverIterations);
}
std::optional<uint32_t> BakedThrough(const state::Registry &r) { return r.Resources.get<PhysicsState>().Baked; }
uint32_t BodyCount(const state::Registry &r) { return uint32_t(r.Resources.get<PhysicsState>().Bodies.size()); }
bool DoesFilterAllow(const state::Registry &r, state::Entity source, state::Entity target) {
    const auto &masks = r.Resources.get<PhysicsState>().Masks;
    const auto a = masks.find(source), b = masks.find(target);
    return a == masks.end() || b == masks.end() || (a->second.Layer & b->second.Collides) != 0;
}
void InvalidateCache(state::Registry &r) { r.Resources.get<PhysicsState>().CacheInvalid = true; }
bool AdvancePlayback(state::Registry &r, int from_frame, int to_frame, int range_start_frame, int range_end_frame, float fps) {
    auto &s = r.Resources.get<PhysicsState>();
    UpdateSettings(r, fps);
    if (std::exchange(s.CacheInvalid, false) || uint32_t(range_start_frame) != s.CacheStartFrame) s.Invalidate();
    s.CacheStartFrame = range_start_frame;
    s.CacheEndFrame = range_end_frame;
    if (s.Bodies.empty()) return false;
    const bool restarted = !s.Baked;
    if (restarted) Restart(s, r);
    if (from_frame == to_frame && !restarted) return false;
    if (from_frame != to_frame) BakeThrough(r, to_frame, fps);
    SamplePosesAtFrame(r, float(to_frame));
    const auto &contacts = s.ContactFrames[std::clamp(uint32_t(to_frame), s.CacheStartFrame, *s.Baked) - s.CacheStartFrame];
    r.Resources.get<PhysicsContactImpacts>() = contacts.Impacts;
    r.Resources.get<PhysicsSustainedContacts>() = contacts.Sustained;
    return true;
}

void BakeThrough(state::Registry &r, int through_frame, float fps) {
    auto &s = r.Resources.get<PhysicsState>();
    if (s.Bodies.empty()) return;
    UpdateSettings(r, fps);
    if (!s.Baked) Restart(s, r);
    const uint32_t target = std::min(uint32_t(std::max(through_frame, int(s.CacheStartFrame))), s.CacheEndFrame);
    if (*s.Baked >= target) return;
    auto impacts = std::move(r.Resources.get<PhysicsContactImpacts>());
    auto sustained = std::move(r.Resources.get<PhysicsSustainedContacts>());
    while (*s.Baked < target) BakeFrame(r, s, *s.Baked + 1, fps);
    r.Resources.get<PhysicsContactImpacts>() = std::move(impacts);
    r.Resources.get<PhysicsSustainedContacts>() = std::move(sustained);
}

void SamplePosesAtFrame(state::Registry &r, float frame) {
    auto &s = r.Resources.get<PhysicsState>();
    if (s.Bodies.empty() || !s.Baked) return;
    const float clamped = std::clamp(frame, float(s.CacheStartFrame), float(*s.Baked));
    const auto lo = uint32_t(std::floor(clamped));
    const uint32_t hi = std::min(lo + 1, *s.Baked);
    const float t = clamped - float(lo);
    const auto lo_idx = lo - s.CacheStartFrame, hi_idx = hi - s.CacheStartFrame;
    std::vector<state::Entity> moved_children;
    for (const auto entity : SampleOrder(s, r)) {
        const auto &cache = r.get<const BodyPoseCache>(entity);
        const auto &a = cache.Frames[lo_idx], &b = cache.Frames[hi_idx];
        const auto position = Mix(a.P, b.P, t);
        const auto rotation = Slerp(a.R, b.R, t);
        const auto &world = *WorldTransformOf(r, entity);
        const auto scale = s.Bodies.at(entity).Input.Node.S;
        if (world.P == position && world.R == rotation && world.S == scale) continue;
        SetWorldTransform(r, entity, {position, rotation, scale});
        moved_children.append_range(Children{&r, entity});
    }
    // The posed bodies in entity index order, as `owned` requires.
    RecomputeWorldTransforms(r, moved_children, r.view<const BodyPoseCache>() | std::ranges::to<std::vector>(), {});
}

void Init(state::Registry &r) {
    auto &s = r.Resources.emplace<PhysicsState>();
    s.Context.emplace();
    r.Resources.emplace<PhysicsContactImpacts>();
    r.Resources.emplace<PhysicsSustainedContacts>();
    RegisterMeshColliderHandlers(r);
    r.on_destroy<PhysicsConstraintHandle, &OnDestroyPhysicsConstraint>();
    r.on_destroy<PhysicsMotion, &OnDestroyPhysicsInput>();
    r.on_destroy<ColliderShape, &OnDestroyPhysicsInput>();
    r.on_destroy<PhysicsJoint, &OnDestroyPhysicsInput>();
    r.on_destroy<PhysicsJointDef, &OnDestroyPhysicsInput>();
    auto &changes = r.Resources.emplace<PhysicsChanges>();
    changes.Input.bind(r)
        .on<PhysicsMotion>(On::Create | On::Update | On::Destroy)
        .on<PhysicsVelocity>(On::Create | On::Update | On::Destroy)
        .on<ColliderShape>(On::Create | On::Update | On::Destroy)
        .on<SceneParent>(On::Create | On::Update | On::Destroy)
        .on<SceneMembership>(On::Create | On::Update | On::Destroy)
        .on<ColliderMaterial>(On::Create | On::Update | On::Destroy)
        .on<TriggerTag>(On::Create | On::Destroy)
        .on<PhysicsJoint>(On::Create | On::Update | On::Destroy)
        .on<::PhysicsJointDef>(On::Create | On::Update | On::Destroy);
    changes.Transform.bind(r).on<Transform>(On::Update);
    changes.Geometry.bind(r).on<MeshGeometryDirty>(On::Create | On::Update).on<MeshPositionsChanged>(On::Create | On::Update);
    changes.ColliderPolicy.bind(r).on<::ColliderPolicy>(On::Create | On::Update);
    changes.Colliders.bind(r).on<ColliderShape>(On::Create | On::Update | On::Destroy);
    changes.MaterialDef.bind(r).on<::PhysicsMaterial>(On::Create | On::Update | On::Destroy);
    changes.CollisionSystemDef.bind(r).on<CollisionSystem>(On::Create | On::Update | On::Destroy);
    changes.CollisionFilterDef.bind(r).on<CollisionFilter>(On::Create | On::Update | On::Destroy);
    changes.Scenes.bind(r, state::Clearing::Explicit).on<Scene>(On::Create | On::Destroy);
    changes.BodyMesh.bind(r).on<PhysicsBodyHandle>(On::Create | On::Destroy).on<ColliderShape>(On::Create | On::Update | On::Destroy);
}
void Deinit(state::Registry &r) {
    r.Resources.erase<PhysicsState>();
}
} // namespace physics
