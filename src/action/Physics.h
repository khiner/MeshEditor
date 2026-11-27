#pragma once

#include "action/Core.h"
#include "action/Emit.h"
#include "physics/PhysicsTypes.h"

namespace action::physics {
// Entity operations apply to their Target, defaulting to the active entity.
struct SetMotionType {
    enum class Type : uint8_t {
        None,
        Static,
        Kinematic,
        Dynamic
    };
    Type Value;
    Target Target{OnActive{}};
};
// Set `LockKind` when changing the collider-shape alternative.
struct SetColliderShape {
    PhysicsShape Shape;
    bool LockKind;
    Target Target{OnActive{}};
};

struct AddTrigger {
    Target Target{OnActive{}};
};
struct RemoveTriggerNodes {
    Target Target{OnActive{}};
};
// Marks the targeted ColliderShape as a sensor.
struct SetTrigger {
    bool Value;
    Target Target{OnActive{}};
};

// Document-level resources, created with an ordinal name.
struct AddPhysicsMaterial {};
struct AddCollisionSystem {};
struct AddCollisionFilter {};
struct AddJointDescription {};

// `Add` appends a missing node or removes all occurrences.
struct ToggleFilterEntity {
    enum class List : uint8_t { Systems,
                                CollideSystems };
    state::Entity FilterEntity;
    List Which;
    state::Entity SystemEntity;
    bool Add;
};

// Maps a joint item type to its JointDescription vector.
template<typename T> inline constexpr std::vector<T> JointDescription::*JointVecMember = nullptr;
template<> inline constexpr std::vector<PhysicsJointLimit> JointDescription::*JointVecMember<PhysicsJointLimit> = &JointDescription::Limits;
template<> inline constexpr std::vector<PhysicsJointDrive> JointDescription::*JointVecMember<PhysicsJointDrive> = &JointDescription::Drives;

template<typename T>
struct SetJointVecItem {
    state::Entity JointDescriptionEntity;
    uint32_t Index;
    std::unique_ptr<T> Value;
};
template<typename T>
struct AddJointVecItem {
    state::Entity JointDescriptionEntity;
};
template<typename T>
struct DeleteJointVecItem {
    state::Entity JointDescriptionEntity;
    uint32_t Index;
};

using Action = std::variant<
    SetMotionType, SetColliderShape, AddTrigger, RemoveTriggerNodes, SetTrigger,
    AddPhysicsMaterial, AddCollisionSystem, AddCollisionFilter, AddJointDescription,
    ToggleFilterEntity,
    SetJointVecItem<PhysicsJointLimit>, AddJointVecItem<PhysicsJointLimit>, DeleteJointVecItem<PhysicsJointLimit>,
    SetJointVecItem<PhysicsJointDrive>, AddJointVecItem<PhysicsJointDrive>, DeleteJointVecItem<PhysicsJointDrive>>;

void Apply(state::Registry &, const Action &);
} // namespace action::physics
namespace action {
void Append(Destination, physics::Action);
}
