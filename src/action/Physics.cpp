#include "action/Physics.h"
#include "Variant.h"
#include "action/ScopeResolve.h"
#include "scene/Entity.h"
#include "state/Scene.h"

namespace action::physics {
namespace {
template<typename T> void Rename(state::Scene &r, state::Entity e, const std::string &name) {
    r.patch<T>(e, [&](T &x) { x.Name = name; });
}
template<typename T> void AddNamed(state::Scene &r, std::string_view prefix) {
    r.emplace<T>(r.create(), T{.Name = std::string{prefix} + ' ' + std::to_string(r.view<T>().size())});
}
} // namespace

void Apply(state::Scene &r, state::Entity, const Action &action) {
    std::visit(
        overloaded{
            [&](const SetMotionType &a) {
                using Type = SetMotionType::Type;
                // Selected scope fans out to the selected entities that already take part in physics.
                ForEachScopeTarget(
                    a.Scope, state::Null, state::Null,
                    [&] { return FindActiveEntity(r); },
                    [&](auto &&fn) {
                        for (const auto e : r.view<Selected>())
                            if (r.any_of<ColliderShape, PhysicsMotion>(e)) fn(e);
                    },
                    [&](state::Entity e) {
                        const bool want_motion = a.Value == Type::Kinematic || a.Value == Type::Dynamic;
                        const bool want_collider = a.Value == Type::Static || want_motion;
                        if (!want_motion) r.remove<PhysicsMotion>(e);
                        if (!want_collider) r.remove<ColliderShape>(e);
                        if (want_collider && !r.all_of<ColliderShape>(e)) {
                            r.emplace<ColliderShape>(e);
                            r.emplace<ColliderPolicy>(e);
                        }
                        if (want_motion) {
                            const bool is_kinematic = a.Value == Type::Kinematic;
                            if (!r.all_of<PhysicsMotion>(e)) r.emplace<PhysicsMotion>(e, PhysicsMotion{.IsKinematic = is_kinematic});
                            else r.patch<PhysicsMotion>(e, [is_kinematic](PhysicsMotion &m) { m.IsKinematic = is_kinematic; });
                        }
                    }
                );
            },
            [&](const SetMotion &a) {
                ForEachComponentTarget<PhysicsMotion>(r, a.Scope, state::Null, state::Null, [&](state::Entity e) { r.replace<PhysicsMotion>(e, *a.Value); });
            },
            [&](const SetColliderShape &a) {
                ForEachComponentTarget<ColliderShape>(r, a.Scope, state::Null, state::Null, [&](state::Entity e) {
                    const auto owner_mesh = FindMeshEntity(r, e);
                    r.patch<ColliderShape>(e, [&](ColliderShape &cs) {
                        cs.Shape = a.Shape;
                        if (IsMeshBackedShape(a.Shape) && cs.MeshEntity == state::Null) cs.MeshEntity = owner_mesh;
                    });
                    if (a.LockKind) r.patch<ColliderPolicy>(e, [](ColliderPolicy &p) { p.LockedKind = true; });
                });
            },
            [&](AddTrigger) {
                const auto e = FindActiveEntity(r);
                r.emplace<ColliderShape>(e);
                r.emplace<ColliderPolicy>(e);
                r.emplace<TriggerTag>(e);
            },
            [&](RemoveTriggerNodes) { r.remove<TriggerNodes>(FindActiveEntity(r)); },
            [&](const SetTrigger &a) {
                const auto e = FindActiveEntity(r);
                if (a.Value) r.emplace_or_replace<TriggerTag>(e);
                else r.remove<TriggerTag>(e);
            },
            [&](AddPhysicsMaterial) { AddNamed<PhysicsMaterial>(r, "Material"); },
            [&](AddCollisionSystem) { AddNamed<CollisionSystem>(r, "System"); },
            [&](AddCollisionFilter) { AddNamed<CollisionFilter>(r, "Filter"); },
            [&](AddJointDef) { AddNamed<PhysicsJointDef>(r, "Joint"); },
            [&](const RenamePhysicsMaterial &a) { Rename<PhysicsMaterial>(r, a.Entity, a.Name); },
            [&](const RenameCollisionSystem &a) { Rename<CollisionSystem>(r, a.Entity, a.Name); },
            [&](const RenameCollisionFilter &a) { Rename<CollisionFilter>(r, a.Entity, a.Name); },
            [&](const RenameJointDef &a) { Rename<PhysicsJointDef>(r, a.Entity, a.Name); },
            [&](const ToggleFilterEntity &a) {
                r.patch<CollisionFilter>(a.FilterEntity, [&](CollisionFilter &f) {
                    auto &vec = a.Which == ToggleFilterEntity::List::Systems ? f.Systems : f.CollideSystems;
                    if (a.Add) {
                        if (std::find(vec.begin(), vec.end(), a.SystemEntity) == vec.end()) vec.emplace_back(a.SystemEntity);
                    } else std::erase(vec, a.SystemEntity);
                });
            },
            [&]<typename T>(const SetJointVecItem<T> &a) {
                r.patch<PhysicsJointDef>(a.JointDefEntity, [&](PhysicsJointDef &d) { (d.*JointVecMember<T>)[a.Index] = *a.Value; });
            },
            [&]<typename T>(const AddJointVecItem<T> &a) {
                r.patch<PhysicsJointDef>(a.JointDefEntity, [&](PhysicsJointDef &d) { (d.*JointVecMember<T>).emplace_back(); });
            },
            [&]<typename T>(const DeleteJointVecItem<T> &a) {
                r.patch<PhysicsJointDef>(a.JointDefEntity, [&](PhysicsJointDef &d) {
                    auto &vec = d.*JointVecMember<T>;
                    vec.erase(vec.begin() + a.Index);
                });
            },
        },
        action
    );
}
} // namespace action::physics
