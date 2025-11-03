#pragma once

#include <cstdint>
#include <limits>

namespace he {
using uint = uint32_t;

constexpr uint null{std::numeric_limits<uint>::max()};

// Tag types for type-safe handles
namespace tag {
struct Vertex {};
struct Halfedge {};
struct Edge {};
struct Face {};

} // namespace tag

// Generic handle template
template<typename Tag>
struct Handle {
    uint Index{null};

    uint operator*() const { return Index; }
    auto operator<=>(const Handle &) const = default;
    operator bool() const { return Index != null; }
};

using VH = Handle<tag::Vertex>;
using HH = Handle<tag::Halfedge>;
using EH = Handle<tag::Edge>;
using FH = Handle<tag::Face>;
} // namespace he
