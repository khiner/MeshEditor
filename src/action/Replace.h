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
} // namespace action
