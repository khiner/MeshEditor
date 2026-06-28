#include "action/Physics.h"
#include "action/Dispatch.h"
#include "action/ScopeResolve.h"
#include "scene/Entity.h"

namespace action::physics {
void Apply(entt::registry &r, entt::entity viewport, const Action &action) {
    // Selected scope fans out to every selected entity matching `accept`, otherwise the active entity.
    auto for_each_physics_target = [&](Scope scope, auto &&accept, auto &&fn) {
        if (scope != Scope::Selected && scope != Scope::SelectedDelta) {
            if (const auto e = FindActiveEntity(r); e != null_entity) fn(e);
        } else
            for (const auto e : r.view<Selected>())
                if (accept(e)) fn(e);
    };
    std::visit(
        overloaded{
            [&](const CreateNamed &a) { ApplyCreateNamed(r, a.ComponentType, a.Prefix); },
            [&](const SetName &a) { ApplySetName(r, a.ComponentType, a.Entity, a.Name); },
            [&](const SetMotionType &a) {
                using Type = SetMotionType::Type;
                const auto accept = [&](entt::entity e) { return r.any_of<ColliderShape, PhysicsMotion>(e); };
                for_each_physics_target(a.Scope, accept, [&](entt::entity e) {
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
                });
            },
            [&](const SetColliderShape &a) {
                const auto accept = [&](entt::entity e) { return r.all_of<ColliderShape>(e); };
                for_each_physics_target(a.Scope, accept, [&](entt::entity e) {
                    const auto owner_mesh = FindMeshEntity(r, e);
                    r.patch<ColliderShape>(e, [&](ColliderShape &cs) {
                        cs.Shape = a.Shape;
                        if (IsMeshBackedShape(a.Shape) && cs.MeshEntity == null_entity) cs.MeshEntity = owner_mesh;
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
            [&]<typename Field>(const Update<Field> &a) { ApplyUpdate(r, viewport, a); },
            [&](const Replace<PhysicsMotion> &a) { ForEachReplaceTarget<PhysicsMotion>(r, a.Scope, a.Entity, [&](entt::entity e) { r.emplace_or_replace<PhysicsMotion>(e, *a.Value); }); },
        },
        action
    );
}
} // namespace action::physics
