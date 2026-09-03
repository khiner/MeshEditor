#pragma once

#include "Variant.h"
#include "action/Core.h"
#include "physics/PhysicsTypes.h"

namespace action {
// Shared ownership bounds the action variant size.
template<>
struct Replace<PhysicsMotion> {
    Scope Scope{Scope::Entity};
    entt::entity Entity{null_entity};
    std::unique_ptr<PhysicsMotion> Value;
};
} // namespace action

namespace action::physics {
// AddTrigger and RemoveTriggerNodes target the active entity.
// SetMotionType and SetColliderShape use Scope.
struct SetMotionType {
    enum class Type : uint8_t {
        None,
        Static,
        Kinematic,
        Dynamic
    };
    Type Value;
    Scope Scope{Scope::Active};
};

// Set `LockKind` when changing the collider-shape alternative.
struct SetColliderShape {
    PhysicsShape Shape;
    bool LockKind;
    Scope Scope{Scope::Active};
};

struct AddTrigger {};
struct RemoveTriggerNodes {};

// `Add` appends a missing node or removes all occurrences.
struct ToggleFilterEntity {
    enum class List : uint8_t { Systems,
                                CollideSystems };
    entt::entity FilterEntity;
    List Which;
    entt::entity SystemEntity;
    bool Add;
};

// Maps a joint item type to its PhysicsJointDef vector.
template<typename T> inline constexpr std::vector<T> PhysicsJointDef::*JointVecMember = nullptr;
template<> inline constexpr std::vector<PhysicsJointLimit> PhysicsJointDef::*JointVecMember<PhysicsJointLimit> = &PhysicsJointDef::Limits;
template<> inline constexpr std::vector<PhysicsJointDrive> PhysicsJointDef::*JointVecMember<PhysicsJointDrive> = &PhysicsJointDef::Drives;

template<typename T>
struct SetJointVecItem {
    entt::entity JointDefEntity;
    uint32_t Index;
    std::unique_ptr<T> Value;
};
template<typename T>
struct AddJointVecItem {
    entt::entity JointDefEntity;
};
template<typename T>
struct DeleteJointVecItem {
    entt::entity JointDefEntity;
    uint32_t Index;
};

using Actions = std::variant<
    SetName, SetMotionType, SetColliderShape, AddTrigger, RemoveTriggerNodes,
    CreateNamed, ToggleFilterEntity,
    SetJointVecItem<PhysicsJointLimit>, AddJointVecItem<PhysicsJointLimit>, DeleteJointVecItem<PhysicsJointLimit>,
    SetJointVecItem<PhysicsJointDrive>, AddJointVecItem<PhysicsJointDrive>, DeleteJointVecItem<PhysicsJointDrive>>;

using Action = MergedVariantT<
    Actions,
    Update<CollideMode>, Update<PhysicsCombineMode>, Update<PhysicsDriveType>, Update<PhysicsDriveMode>,
    Replace<PhysicsMotion>>;

void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::physics
