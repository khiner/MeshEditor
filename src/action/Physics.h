#pragma once

#include "action/Core.h"
#include "physics/PhysicsTypes.h"

#include <string>

namespace action::physics {
// AddTrigger, RemoveTriggerNodes and SetTrigger target the active entity.
// SetMotionType, SetColliderShape and SetMotion use Scope.
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
// Replaces the whole motion record. Shared ownership bounds the action variant size.
struct SetMotion {
    std::unique_ptr<PhysicsMotion> Value;
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
// Marks the active entity's ColliderShape as a sensor.
struct SetTrigger {
    bool Value;
};

// Document-level resources, created with an ordinal name.
struct AddPhysicsMaterial {};
struct AddCollisionSystem {};
struct AddCollisionFilter {};
struct AddJointDef {};
struct RenamePhysicsMaterial {
    state::Entity Entity;
    std::string Name;
};
struct RenameCollisionSystem {
    state::Entity Entity;
    std::string Name;
};
struct RenameCollisionFilter {
    state::Entity Entity;
    std::string Name;
};
struct RenameJointDef {
    state::Entity Entity;
    std::string Name;
};

// `Add` appends a missing node or removes all occurrences.
struct ToggleFilterEntity {
    enum class List : uint8_t { Systems,
                                CollideSystems };
    state::Entity FilterEntity;
    List Which;
    state::Entity SystemEntity;
    bool Add;
};

// Maps a joint item type to its PhysicsJointDef vector.
template<typename T> inline constexpr std::vector<T> PhysicsJointDef::*JointVecMember = nullptr;
template<> inline constexpr std::vector<PhysicsJointLimit> PhysicsJointDef::*JointVecMember<PhysicsJointLimit> = &PhysicsJointDef::Limits;
template<> inline constexpr std::vector<PhysicsJointDrive> PhysicsJointDef::*JointVecMember<PhysicsJointDrive> = &PhysicsJointDef::Drives;

template<typename T>
struct SetJointVecItem {
    state::Entity JointDefEntity;
    uint32_t Index;
    std::unique_ptr<T> Value;
};
template<typename T>
struct AddJointVecItem {
    state::Entity JointDefEntity;
};
template<typename T>
struct DeleteJointVecItem {
    state::Entity JointDefEntity;
    uint32_t Index;
};

using Action = std::variant<
    SetMotionType, SetMotion, SetColliderShape, AddTrigger, RemoveTriggerNodes, SetTrigger,
    AddPhysicsMaterial, AddCollisionSystem, AddCollisionFilter, AddJointDef,
    RenamePhysicsMaterial, RenameCollisionSystem, RenameCollisionFilter, RenameJointDef,
    ToggleFilterEntity,
    SetJointVecItem<PhysicsJointLimit>, AddJointVecItem<PhysicsJointLimit>, DeleteJointVecItem<PhysicsJointLimit>,
    SetJointVecItem<PhysicsJointDrive>, AddJointVecItem<PhysicsJointDrive>, DeleteJointVecItem<PhysicsJointDrive>>;

void Apply(state::Scene &, state::Entity viewport, const Action &);
} // namespace action::physics
