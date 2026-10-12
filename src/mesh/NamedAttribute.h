#pragma once

#include "Range.h"
#include "gpu/NamedAttributeRef.h"
#include "mesh/GeometrySelection.h"

#include <concepts>
#include <cstring>
#include <span>
#include <string_view>
#include <type_traits>

// A name fits Blender's 64-byte attribute name with its terminator.
using AttributeName = std::array<char, 64>;
// The bits of a value's components, zero beyond them.
using AttributeValue = std::array<uint32_t, 4>;

template<typename T>
concept AttributeScalar = std::same_as<T, float> || std::same_as<T, vec2> || std::same_as<T, vec3> || std::same_as<T, vec4> || std::same_as<T, int32_t> || std::same_as<T, bool>;
template<AttributeScalar T>
constexpr AttributeType AttributeTypeOf = std::same_as<T, float> ? AttributeType::Float :
    std::same_as<T, vec2>                                        ? AttributeType::Vec2 :
    std::same_as<T, vec3>                                        ? AttributeType::Vec3 :
    std::same_as<T, vec4>                                        ? AttributeType::Vec4 :
    std::same_as<T, int32_t>                                     ? AttributeType::Int :
                                                                   AttributeType::Bool;
constexpr std::array AttributeTypes{AttributeType::Float, AttributeType::Vec2, AttributeType::Vec3, AttributeType::Vec4, AttributeType::Int, AttributeType::Bool};
// Calls `fn` with the value type of `type` as its template argument.
decltype(auto) WithAttributeScalar(AttributeType type, auto &&fn) {
    switch (type) {
        case AttributeType::Float: return fn.template operator()<float>();
        case AttributeType::Vec2: return fn.template operator()<vec2>();
        case AttributeType::Vec3: return fn.template operator()<vec3>();
        case AttributeType::Vec4: return fn.template operator()<vec4>();
        case AttributeType::Int: return fn.template operator()<int32_t>();
        case AttributeType::Bool: return fn.template operator()<bool>();
    }
    std::unreachable();
}
// One value per element of a canonical element block, where a bool block is the selection mask layout.
template<AttributeScalar T> using AttributeBlock = std::conditional_t<std::same_as<T, bool>, SelectionBlock, std::array<T, MeshElementBlockSize>>;

// A record's named attribute, whose id is its slot in the store's attribute table.
// A free slot has no record.
struct NamedAttribute {
    uint32_t Record{InvalidOffset};
    AttributeDomain Domain{};
    AttributeType Type{};
    AttributeName Name{};
    Range Values{}; // The value blocks of the record's local blocks in its domain, in their order
    AttributeValue Default{};
    std::string_view NameView() const { return {Name.data(), strnlen(Name.data(), Name.size())}; }
};

// An attribute's values at its record's handles in its domain, a block per local block.
template<AttributeScalar T>
struct NamedAttributeView {
    std::span<const AttributeBlock<T>> Values;
    T operator[](uint32_t handle) const {
        const auto &block = Values[handle / MeshElementBlockSize];
        const auto slot = handle % MeshElementBlockSize;
        if constexpr (std::same_as<T, bool>) return (block[slot / 32u] >> (slot % 32u)) & 1u;
        else return block[slot];
    }
};
