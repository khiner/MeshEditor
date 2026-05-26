#include "action/Physics.h"
#include "action/Dispatch.h"
#include "scene/Entity.h"

#include <format>

namespace {
// Components targeted by the physics enum Updates (CollideMode, PhysicsCombineMode, drive types).
using UpdateComponents = action::TypeList<
    PhysicsMaterial, CollisionFilter, ColliderMaterial, ColliderPolicy,
    PhysicsMotion, PhysicsVelocity, PhysicsJoint, TriggerNodes, PhysicsSimulationSettings>;
using NamedComponents = action::TypeList<PhysicsMaterial, CollisionSystem, CollisionFilter, PhysicsJointDef>;
} // namespace

namespace action::physics {
void Apply(entt::registry &r, entt::entity viewport, const Action &action) {
    std::visit(
        overloaded{
            [&](const CreateNamed &a) {
                DispatchByTypeHash(NamedComponents{}, a.ComponentType, [&]<typename T> {
                    r.emplace<T>(r.create(), T{.Name = std::format("{} {}", a.Prefix, r.view<T>().size())});
                });
            },
            [&](const SetName &a) {
                DispatchByTypeHash(NamedComponents{}, a.ComponentType, [&]<typename T> {
                    r.patch<T>(a.Entity, [&](T &x) { x.Name = a.Name; });
                });
            },
            [&](const SetMotionType &a) {
                using Type = SetMotionType::Type;
                const auto e = FindActiveEntity(r);
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
            },
            [&](const SetColliderShape &a) {
                const auto e = FindActiveEntity(r);
                const auto owner_mesh = FindMeshEntity(r, e);
                r.patch<ColliderShape>(e, [&](ColliderShape &cs) {
                    cs.Shape = a.Shape;
                    if (IsMeshBackedShape(a.Shape) && cs.MeshEntity == null_entity) cs.MeshEntity = owner_mesh;
                });
                if (a.LockKind) r.patch<ColliderPolicy>(e, [](ColliderPolicy &p) { p.LockedKind = true; });
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
                    auto &vec = f.*(a.Field);
                    if (a.Add) {
                        if (std::find(vec.begin(), vec.end(), a.SystemEntity) == vec.end()) vec.push_back(a.SystemEntity);
                    } else std::erase(vec, a.SystemEntity);
                });
            },
            [&]<typename T>(const SetJointVecItem<T> &a) {
                r.patch<PhysicsJointDef>(a.JointDefEntity, [&](PhysicsJointDef &d) { (d.*(a.Field))[a.Index] = *a.Value; });
            },
            [&]<typename T>(const AddJointVecItem<T> &a) {
                r.patch<PhysicsJointDef>(a.JointDefEntity, [&](PhysicsJointDef &d) { (d.*(a.Field)).push_back({}); });
            },
            [&]<typename T>(const DeleteJointVecItem<T> &a) {
                r.patch<PhysicsJointDef>(a.JointDefEntity, [&](PhysicsJointDef &d) {
                    auto &vec = d.*(a.Field);
                    vec.erase(vec.begin() + a.Index);
                });
            },
            [&]<typename Field>(const Update<Field> &a) { ApplyUpdate<UpdateComponents>(r, viewport, a); },
            [&](const Replace<PhysicsMotion> &a) { r.emplace_or_replace<PhysicsMotion>(a.Entity, *a.Value); },
            [&](const ReplaceActive<PhysicsMotion> &a) { r.emplace_or_replace<PhysicsMotion>(FindActiveEntity(r), *a.Value); },
        },
        action
    );
}
} // namespace action::physics
