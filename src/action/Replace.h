#pragma once

#include <entt/entity/fwd.hpp>

#include <memory>

struct PhysicsMotion;

namespace action {
template<typename T>
struct Replace {
    entt::entity Entity;
    T Value;
};

// Specialization keeps the variant alt small for heavy components.
template<>
struct Replace<PhysicsMotion> {
    entt::entity Entity;
    std::unique_ptr<PhysicsMotion> Value;
};

// Like Replace<T> but carries no entity — the dispatcher resolves the target via FindActiveEntity(R).
// Separate type (vs an optional field on Replace<T>) so we don't waste bytes storing entt::null.
template<typename T>
struct ReplaceActive {
    T Value;
};
template<>
struct ReplaceActive<PhysicsMotion> {
    std::unique_ptr<PhysicsMotion> Value;
};
} // namespace action
