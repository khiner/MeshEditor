#include "action/Physics.h"
#include "TypeList.h"

#include "action/ScopeResolve.h"

namespace action::physics {
namespace {
template<typename T> void AddNamed(state::Registry &r, std::string_view prefix) {
    r.emplace<T>(r.create(), T{.Name = std::string{prefix} + ' ' + std::to_string(r.view<T>().size())});
}
} // namespace

void Apply(state::Registry &r, const Action &action) {
    std::visit(
        overloaded{
            [&](const SetMotionType &a) {
                using Type = SetMotionType::Type;
                // Apply Selected scope only to selected entities already participating in physics.
                ForEachTarget(
                    a.Target,
                    [&] { return FindActiveEntity(r); },
                    [&](auto &&fn) {
                        for (const auto e : SelectedIn(r)) {
                            if (r.any_of<ColliderShape, PhysicsMotion>(e)) fn(e);
                        }
                    },
                    [&](state::Entity e) {
                        e = HierarchySource(r, e);
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
            [&](const SetColliderShape &a) {
                ForEachComponentTarget<ColliderShape>(r, a.Target, [&](state::Entity e) {
                    const auto owner_mesh = FindMeshEntity(r, e);
                    r.patch<ColliderShape>(e, [&](ColliderShape &cs) {
                        cs.Shape = a.Shape;
                        if (IsMeshBackedShape(a.Shape) && cs.MeshEntity == state::Null) cs.MeshEntity = owner_mesh;
                    });
                    if (a.LockKind) r.patch<ColliderPolicy>(e, [](ColliderPolicy &p) { p.LockedKind = true; });
                });
            },
            [&](const AddTrigger &a) {
                ForEachComponentTarget<Transform>(r, a.Target, [&](state::Entity e) {
                    r.get_or_emplace<ColliderShape>(e);
                    r.get_or_emplace<ColliderPolicy>(e);
                    r.emplace_or_replace<TriggerTag>(e);
                });
            },
            [&](const RemoveTriggerNodes &a) {
                ForEachComponentTarget<TriggerNodes>(r, a.Target, [&](state::Entity e) { r.remove<TriggerNodes>(e); });
            },
            [&](const SetTrigger &a) {
                ForEachComponentTarget<ColliderShape>(r, a.Target, [&](state::Entity e) {
                    if (a.Value) r.emplace_or_replace<TriggerTag>(e);
                    else r.remove<TriggerTag>(e);
                });
            },
            [&](AddPhysicsMaterial) { AddNamed<PhysicsMaterial>(r, "Material"); },
            [&](AddCollisionSystem) { AddNamed<CollisionSystem>(r, "System"); },
            [&](AddCollisionFilter) { AddNamed<CollisionFilter>(r, "Filter"); },
            [&](AddJointDef) { AddNamed<PhysicsJointDef>(r, "Joint"); },
            [&](const RenameResource &a) {
                const auto rename = [&]<typename T>() { if (r.all_of<T>(a.Entity)) r.patch<T>(a.Entity, [&](auto &v) { v.Name = a.Name; }); };
                [&]<typename... T>(TypeList<T...>) { (rename.template operator()<T>(), ...); }(TypeList<PhysicsMaterial, CollisionSystem, CollisionFilter, PhysicsJointDef>{});
            },
            [&](const ToggleFilterEntity &a) {
                r.patch<CollisionFilter>(a.FilterEntity, [&](CollisionFilter &f) {
                    auto &vec = a.Which == ToggleFilterEntity::List::Systems ? f.Systems : f.CollideSystems;
                    if (a.Add) {
                        if (!std::ranges::contains(vec, a.SystemEntity)) vec.emplace_back(a.SystemEntity);
                    } else {
                        std::erase(vec, a.SystemEntity);
                    }
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
