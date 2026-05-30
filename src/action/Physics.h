#pragma once

#include "Variant.h"
#include "action/Core.h"
#include "physics/PhysicsTypes.h"

namespace action {
// Heap-allocate big types to keep the variant small.
template<>
struct Replace<PhysicsMotion> {
    entt::entity Entity;
    std::unique_ptr<PhysicsMotion> Value;
};
template<>
struct ReplaceActive<PhysicsMotion> {
    std::unique_ptr<PhysicsMotion> Value;
};
} // namespace action

namespace action::physics {
// SetMotionType / SetColliderShape / AddTrigger / RemoveTriggerNodes target the active entity.
struct SetMotionType {
    enum class Type : uint8_t {
        None,
        Static,
        Kinematic,
        Dynamic
    };
    Type Value;
};

// `LockKind=true` locks `ColliderPolicy.LockedKind`; set it when the variant alternative changed.
struct SetColliderShape {
    PhysicsShape Shape;
    bool LockKind;
};

struct AddTrigger {};
struct RemoveTriggerNodes {};

// `Add=true` appends iff not present; `Add=false` erases all occurrences.
struct ToggleFilterEntity {
    enum class List : uint8_t { Systems,
                                CollideSystems };
    entt::entity FilterEntity;
    List Which;
    entt::entity SystemEntity;
    bool Add;
};

// Maps a joint-vec-item element type to the PhysicsJointDef vector it targets.
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
    Replace<PhysicsMotion>, ReplaceActive<PhysicsMotion>>;

void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::physics
