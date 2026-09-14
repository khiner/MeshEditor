#pragma once
#include <cstdint>
#include <functional>
namespace state {
enum class Entity : uint32_t {};
inline constexpr Entity Null = Entity{UINT32_MAX};
constexpr uint32_t Integral(Entity e) { return uint32_t(e); }
constexpr uint32_t Index(Entity e) { return uint32_t(e) & 0xfffffu; }
constexpr uint32_t Generation(Entity e) { return uint32_t(e) >> 20; }
constexpr Entity MakeEntity(uint32_t index, uint32_t generation) { return Entity{index | (generation << 20)}; }
struct Scene;
} // namespace state
template<> struct std::hash<state::Entity> {
    size_t operator()(state::Entity e) const { return state::Integral(e); }
};

inline constexpr state::Entity null_entity = state::Null;
