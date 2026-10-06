#pragma once
#include "mesh/ElementView.h"

#include <ranges>

template<typename T>
struct ElementAttributeView {
    std::span<const uint32_t> Blocks;
    std::span<const T> Values;
    bool empty() const { return Values.empty(); }
    T GetOr(uint32_t handle, T fallback = {}) const {
        if (handle / MeshElementBlockSize >= Blocks.size() || !Blocks[handle / MeshElementBlockSize]) return fallback;
        return (*this)[handle];
    }
    uint32_t Index(uint32_t handle) const {
        const auto block = Blocks[handle / MeshElementBlockSize];
        assert(block != 0u);
        return (block - 1u) * MeshElementBlockSize + handle % MeshElementBlockSize;
    }
    const T &operator[](uint32_t handle) const { return Values[Index(handle)]; }
};

// Derived triangles name their polygon corners explicitly. Tessellation and
// allocation order do not affect the canonical attribute handles.
struct TriangleCorners {
    ElementView<uvec3> Values;
    std::span<const uint32_t> Handles; // Optional canonical triangle enumeration, borrowed from render membership.
    size_t size() const { return Handles.empty() ? Values.size() : Handles.size(); }
    uint32_t operator[](uint32_t corner) const { return Values[Handles.empty() ? corner / 3u : Handles[corner / 3u]][corner % 3u]; }
};

// Borrow canonical corners and vertices.
// External triangle soups may supply their packed indices directly.
// Both inputs use the same geometry algorithms.
struct TriangleVertexView {
    std::span<const uint32_t> Vertices;
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
    uint32_t operator[](size_t i) const { return Vertices[Corners.Values.empty() ? First + i : Corners[uint32_t(First + i)]]; }
    std::array<uint32_t, 3> TriangleAt(size_t triangle) const {
        assert(triangle * 3u + 2u < size());
        const size_t first = First + triangle * 3u;
        if (Corners.Values.empty()) return {Vertices[first], Vertices[first + 1u], Vertices[first + 2u]};
        assert(first % 3u == 0u);
        const auto &corners = Corners.Values[Corners.Handles.empty() ? uint32_t(first / 3u) : Corners.Handles[first / 3u]];
        return {Vertices[corners[0]], Vertices[corners[1]], Vertices[corners[2]]};
    }
    std::array<uint32_t, 3> TriangleAtHandle(uint32_t handle) const {
        if (Corners.Values.empty()) {
            const size_t first = size_t(handle) * 3u;
            return {Vertices[first], Vertices[first + 1u], Vertices[first + 2u]};
        }
        const auto corners = Corners.Values.Values[handle];
        return {Vertices[corners[0]], Vertices[corners[1]], Vertices[corners[2]]};
    }
    TriangleVertexView subspan(size_t first, size_t count) const {
        auto result = *this;
        result.First += first;
        result.Count = count;
        return result;
    }
};

// Face ownership follows the first canonical corner.
// No owner mirror is stored.
struct TriangleFaceView {
    TriangleCorners Corners;
    std::span<const uint32_t> HalfedgeFaces;
    uint32_t operator[](uint32_t triangle) const { return HalfedgeFaces[Corners[triangle * 3u]]; }
};

// CPU consumers that operate on draw triangles borrow canonical attributes.
// No expanded fan-order payload is materialized.
template<typename T>
struct CornerAttributeView {
    ElementAttributeView<T> Attribute;
    TriangleCorners Corners;
    bool empty() const { return Attribute.empty(); }
    const T &operator[](uint32_t corner) const { return Attribute[Corners[corner]]; }
    T GetOr(uint32_t corner, T fallback = {}) const { return Attribute.GetOr(Corners[corner], fallback); }
};
