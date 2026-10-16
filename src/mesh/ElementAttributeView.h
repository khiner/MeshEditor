#pragma once
#include "mesh/ElementView.h"

#include <ranges>

// A record's layer values by handle, empty for a record without the layer.
template<typename T>
struct ElementAttributeView {
    std::span<const T> Values; // From the record's handle 0
    bool empty() const { return Values.empty(); }
    T GetOr(uint32_t handle, T fallback = {}) const { return handle < Values.size() ? Values[handle] : fallback; }
    const T &operator[](uint32_t handle) const { return Values[handle]; }
};

// Derived triangles name their polygon corners explicitly, by record-relative handle.
// Tessellation and allocation order do not affect the corner handles.
struct TriangleCorners {
    ElementView<uvec3> Values;
    std::span<const uint32_t> Handles; // Borrowed triangle handles; TriangleTable maps them into Values.Values.
    std::span<const uint32_t> TriangleTable;
    std::span<const uint32_t> CornerTable; // Resolves corner handles to arena addresses, and empty for packed corners whose handles are their addresses.
    size_t size() const { return Handles.empty() ? Values.size() : Handles.size(); }
    uvec3 Triangle(uint32_t triangle) const { return Handles.empty() ? Values[triangle] : Values.Values[Resolve<MeshElementBlockSize>(TriangleTable, ElementHandle{Handles[triangle]}).Value()]; }
    uint32_t operator[](uint32_t corner) const { return Triangle(corner / 3u)[corner % 3u]; }
    uint32_t Address(uint32_t corner) const {
        const auto handle = (*this)[corner];
        return CornerTable.empty() ? handle : Resolve<MeshElementBlockSize>(CornerTable, ElementHandle{handle}).Value();
    }
};

// Borrow corners and their vertex handles.
// External triangle soups may supply their packed indices directly.
// Both inputs use the same geometry algorithms.
struct TriangleVertexView {
    std::span<const uint32_t> Vertices; // The corners' vertex handles by corner address, or the packed indices
    TriangleCorners Corners;
    size_t First{}, Count{size_t(-1)};
    TriangleVertexView() = default;
    TriangleVertexView(TriangleCorners corners, std::span<const uint32_t> vertices) : Vertices(vertices), Corners(corners) {}
    template<std::ranges::contiguous_range R>
        requires std::same_as<std::ranges::range_value_t<R>, uint32_t>
    TriangleVertexView(const R &indices) : Vertices(indices) {}
    size_t size() const { return Count != size_t(-1) ? Count : Corners.Values.empty() ? Vertices.size() :
                                                                                        Corners.size() * 3u; }
    bool empty() const { return size() == 0; }
    uint32_t operator[](size_t i) const { return Vertices[Corners.Values.empty() ? First + i : Corners.Address(uint32_t(First + i))]; }
    std::array<uint32_t, 3> TriangleAt(size_t triangle) const {
        assert(triangle * 3u + 2u < size());
        const size_t first = First + triangle * 3u;
        if (Corners.Values.empty()) return {Vertices[first], Vertices[first + 1u], Vertices[first + 2u]};
        assert(first % 3u == 0u);
        return {Vertices[Corners.Address(uint32_t(first))], Vertices[Corners.Address(uint32_t(first + 1u))], Vertices[Corners.Address(uint32_t(first + 2u))]};
    }
    // The vertex handles of the triangle at a triangle handle, or of the packed triangle at that index.
    std::array<uint32_t, 3> TriangleAtHandle(uint32_t handle) const {
        if (Corners.Values.empty()) {
            const size_t first = size_t(handle) * 3u;
            return {Vertices[first], Vertices[first + 1u], Vertices[first + 2u]};
        }
        const auto corners = Corners.Values.Values[Resolve<MeshElementBlockSize>(Corners.TriangleTable, ElementHandle{handle}).Value()];
        const auto address = [&](uint32_t corner) { return Corners.CornerTable.empty() ? corner : Resolve<MeshElementBlockSize>(Corners.CornerTable, ElementHandle{corner}).Value(); };
        return {Vertices[address(corners[0])], Vertices[address(corners[1])], Vertices[address(corners[2])]};
    }
    TriangleVertexView subspan(size_t first, size_t count) const {
        auto result = *this;
        result.First += first;
        result.Count = count;
        return result;
    }
};

// Face ownership follows the first corner.
// No owner mirror is stored.
struct TriangleFaceView {
    TriangleCorners Corners;
    std::span<const uint32_t> HalfedgeFaces; // By corner address
    uint32_t operator[](uint32_t triangle) const { return HalfedgeFaces[Corners.Address(triangle * 3u)]; }
};

// CPU consumers that operate on draw triangles borrow attributes, which their corners' handles index.
// No expanded fan-order payload is materialized.
template<typename T>
struct CornerAttributeView {
    ElementAttributeView<T> Attribute;
    TriangleCorners Corners;
    bool empty() const { return Attribute.empty(); }
    const T &operator[](uint32_t corner) const { return Attribute[Corners[corner]]; }
    T GetOr(uint32_t corner, T fallback = {}) const { return Attribute.GetOr(Corners[corner], fallback); }
};
