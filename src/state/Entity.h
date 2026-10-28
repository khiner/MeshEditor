#pragma once

#include <functional>
namespace state {
enum class Entity : uint32_t {};
inline constexpr Entity Null = Entity{UINT32_MAX};
// Entity zero holds the document components and is the first allocation after every Reset.
inline constexpr Entity Document = Entity{0};
constexpr uint32_t Integral(Entity e) { return uint32_t(e); }
constexpr uint32_t Index(Entity e) { return uint32_t(e) & 0xfffffu; }
constexpr uint32_t Generation(Entity e) { return uint32_t(e) >> 20; }
constexpr Entity MakeEntity(uint32_t index, uint32_t generation) { return Entity{index | (generation << 20)}; }
// An entity reference constrained to identities carrying Component.
template<typename Component> struct EntityRef {
    using ComponentType = Component;
    Entity Value{Null};
    bool operator==(const EntityRef &) const = default;
};
template<typename T> inline constexpr bool IsEntityRef = false;
template<typename C> inline constexpr bool IsEntityRef<EntityRef<C>> = true;
struct Registry;
} // namespace state
template<> struct std::hash<state::Entity> {
    size_t operator()(state::Entity e) const { return state::Integral(e); }
};
