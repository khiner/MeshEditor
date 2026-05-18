#pragma once

#include "Variant.h"
#include "action/Core.h"
#include "physics/PhysicsTypes.h"

#include <entt/core/type_info.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

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
struct SetName {
    entt::entity Entity;
    entt::id_type ComponentType;
    std::string Name;
};

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

// Create a new named entity carrying component `ComponentType`, named "<Prefix> <ordinal>".
struct CreateNamed {
    entt::id_type ComponentType;
    std::string_view Prefix;
};

template<typename T>
inline SetName SetNameOf(entt::entity e, std::string name) { return {e, entt::type_hash<T>::value(), std::move(name)}; }
template<typename T>
constexpr CreateNamed CreateNamedOf(std::string_view prefix) { return {entt::type_hash<T>::value(), prefix}; }

// `Add=true` appends iff not present; `Add=false` erases all occurrences.
struct ToggleFilterEntity {
    entt::entity FilterEntity;
    std::vector<entt::entity> CollisionFilter::*Field;
    entt::entity SystemEntity;
    bool Add;
};

template<typename T>
struct SetJointVecItem {
    entt::entity JointDefEntity;
    std::vector<T> PhysicsJointDef::*Field;
    uint32_t Index;
    std::unique_ptr<T> Value;
};
template<typename T>
struct AddJointVecItem {
    entt::entity JointDefEntity;
    std::vector<T> PhysicsJointDef::*Field;
};
template<typename T>
struct DeleteJointVecItem {
    entt::entity JointDefEntity;
    std::vector<T> PhysicsJointDef::*Field;
    uint32_t Index;
};

using Actions = std::variant<
    SetName, SetMotionType, SetColliderShape, AddTrigger, RemoveTriggerNodes,
    CreateNamed, ToggleFilterEntity,
    SetJointVecItem<PhysicsJointLimit>, AddJointVecItem<PhysicsJointLimit>, DeleteJointVecItem<PhysicsJointLimit>,
    SetJointVecItem<PhysicsJointDrive>, AddJointVecItem<PhysicsJointDrive>, DeleteJointVecItem<PhysicsJointDrive>>;

using Action = MergedVariantT<
    Actions, action::Core,
    action::Update<CollideMode>, action::Update<PhysicsCombineMode>,
    action::Replace<PhysicsMotion>, action::ReplaceActive<PhysicsMotion>>;
} // namespace action::physics
