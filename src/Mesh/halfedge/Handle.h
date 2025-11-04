#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string>

namespace he {
using uint = uint32_t;

constexpr uint null{std::numeric_limits<uint>::max()};

enum class Element {
    None,
    Vertex, // Vertices are not duplicated. Each vertex uses the vertex normal.
    Edge, // Vertices are duplicated. Each vertex uses the vertex normal.
    Face, // Vertices are duplicated for each face. Each vertex uses the face normal.
};

constexpr std::array Elements{Element::Vertex, Element::Edge, Element::Face};

constexpr std::string_view label(Element element) {
    switch (element) {
        case Element::Vertex: return "vertex";
        case Element::Edge: return "edge";
        case Element::Face: return "face";
        case Element::None: return "none";
    }
}

// Tag types for type-safe handles
namespace tag {
struct Vertex {};
struct Edge {};
struct Face {};

struct Halfedge {};
} // namespace tag

// Generic handle template
template<typename Tag>
struct Handle {
    uint Index{null};

    uint operator*() const { return Index; }
    auto operator<=>(const Handle &) const = default;
    operator bool() const { return Index != null; }

    constexpr Element GetElement() const {
        if constexpr (std::is_same_v<Tag, tag::Vertex>) return Element::Vertex;
        if constexpr (std::is_same_v<Tag, tag::Edge>) return Element::Edge;
        if constexpr (std::is_same_v<Tag, tag::Face>) return Element::Face;
        return Element::None;
    }
};

using VH = Handle<tag::Vertex>;
using HH = Handle<tag::Halfedge>;
using EH = Handle<tag::Edge>;
using FH = Handle<tag::Face>;

// Type-erased handle with comparison/conversion to typed handles
struct AnyHandle {
    AnyHandle(Element element = Element::None, uint index = null) : Element(element), Index(index) {}
    template<typename Tag> AnyHandle(Handle<Tag> h) : Element(h.GetElement()), Index(*h) {}

    he::Element Element;
    uint Index;

    uint operator*() const { return Index; }
    bool operator==(const AnyHandle &other) const { return Element == other.Element && Index == other.Index; }
    operator bool() const { return Index != null; }

    bool operator==(VH vh) const { return Element == he::Element::Vertex && Index == *vh; }
    bool operator==(EH eh) const { return Element == he::Element::Edge && Index == *eh; }
    bool operator==(FH fh) const { return Element == he::Element::Face && Index == *fh; }

    // Implicit conversion to typed handles
    operator VH() const { return {Element == he::Element::Vertex ? Index : null}; }
    operator EH() const { return {Element == he::Element::Edge ? Index : null}; }
    operator FH() const { return {Element == he::Element::Face ? Index : null}; }
};

struct AnyHandleHash {
    size_t operator()(const AnyHandle &h) const {
        return std::hash<uint>{}(static_cast<uint>(h.Element)) ^ (std::hash<uint>{}(h.Index) << 1);
    }
};
} // namespace he
