#pragma once

#include <cstdint> // uint32_t
#include <memory> // std::allocator

// Lighter weight declarations for entity/registry than <entt/entity/fwd.hpp>
// Plus a null_entity alias for entt::null (Entity.cpp asserts this matches entt::null)
namespace entt {
enum class entity : uint32_t;
template<typename Entity, typename> class basic_registry;
using registry = basic_registry<entity, std::allocator<entity>>;
} // namespace entt

constexpr auto null_entity = static_cast<entt::entity>(-1);
