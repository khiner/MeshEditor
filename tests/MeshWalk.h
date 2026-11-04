#pragma once

#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"

#include <ranges>

// Halfedge traversal and geometry reads over a mesh record's connectivity, for test inspection.
namespace test {
inline Mesh::VH ToVertex(const Mesh &mesh, Mesh::HH h) {
    return Mesh::VH{mesh.GetStore().Arenas().FaceCorners.Buffer.GetSpan<uint32_t>()[mesh.GetConnectivity().HalfedgeAddress(*h).Value()]};
}
// A boundary halfedge has no opposite, so its origin comes from the previous halfedge in the face loop.
inline Mesh::VH FromVertex(const Mesh &mesh, Mesh::HH h) {
    const auto &c = mesh.GetConnectivity();
    if (const auto opposite = c.Opposite(h)) return ToVertex(mesh, opposite);
    const auto previous = c.Previous(h);
    return previous ? ToVertex(mesh, previous) : Mesh::VH{};
}
inline Mesh::HH Halfedge(const Mesh &mesh, Mesh::EH e) { return mesh.GetConnectivity().EdgeHalfedge(*e); }
// The edge's endpoints, from and to along its halfedge.
inline std::array<Mesh::VH, 2> EdgeVertices(const Mesh &mesh, Mesh::EH e) {
    const auto h = Halfedge(mesh, e);
    return {FromVertex(mesh, h), ToVertex(mesh, h)};
}
inline Mesh::HH Opposite(const Mesh &mesh, Mesh::HH h) { return mesh.GetConnectivity().Opposite(h); }
inline Mesh::FH Face(const Mesh &mesh, Mesh::HH h) { return mesh.GetConnectivity().FaceOf(h); }
// The arena address of a mesh's element, which indexes its arena values.
inline uint32_t Address(const Mesh &mesh, Mesh::EH e) { return mesh.GetConnectivity().EdgeAddress(*e).Value(); }
inline uint32_t Address(const Mesh &mesh, Mesh::FH f) { return mesh.GetConnectivity().FaceAddress(*f).Value(); }
inline const vec3 &Position(const Mesh &mesh, Mesh::VH v) { return mesh.GetStore().Arenas().Vertices.Get({mesh.GetConnectivity().VertexAddress(*v).Value(), 1})[0].Position; }
inline vec3 Normal(const Mesh &mesh, Mesh::FH f) { return mesh.GetStore().Arenas().BaseFaceNormals.Get({mesh.GetConnectivity().FaceAddress(*f).Value(), 1})[0]; }
inline vec3 EdgeCenter(const Mesh &mesh, Mesh::EH e) {
    const auto [from, to] = EdgeVertices(mesh, e);
    return .5f * (Position(mesh, from) + Position(mesh, to));
}
inline std::array<vec3, 3> TrianglePositions(const Mesh &mesh, uvec3 corners) {
    const auto at = [&mesh](uint32_t c) { return Position(mesh, ToVertex(mesh, Mesh::HH{c})); };
    return {at(corners.x), at(corners.y), at(corners.z)};
}

// Halfedges from a start, each the step from the one before, until the start returns or the step runs out.
struct Circulation {
    using Step = Mesh::HH (*)(const Mesh &, Mesh::HH);
    struct Iterator {
        using difference_type = std::ptrdiff_t;
        using value_type = Mesh::HH;
        const Mesh *M{};
        Step Advance{};
        Mesh::HH Current{}, Start{};
        Mesh::HH operator*() const { return Current; }
        Iterator &operator++() {
            Current = Advance(*M, Current);
            if (Current == Start) Current = Mesh::HH{};
            return *this;
        }
        Iterator operator++(int) {
            auto before = *this;
            ++*this;
            return before;
        }
        bool operator==(const Iterator &other) const { return Current == other.Current; }
    };
    const Mesh *M;
    Step Advance;
    Mesh::HH Start;
    Iterator begin() const { return {M, Advance, Start, Start}; }
    Iterator end() const { return {M, Advance, Mesh::HH{}, Start}; }
};
inline Mesh::HH NextInFace(const Mesh &mesh, Mesh::HH h) { return mesh.GetConnectivity().Next(h); }
inline Circulation FaceHalfedges(const Mesh &mesh, Mesh::FH f) { return {&mesh, NextInFace, mesh.GetConnectivity().FaceHalfedge(*f)}; }
inline auto FaceVertices(const Mesh &mesh, Mesh::FH f) {
    return FaceHalfedges(mesh, f) | std::views::transform([&mesh](Mesh::HH h) { return ToVertex(mesh, h); });
}
inline uint32_t Valence(const Mesh &mesh, Mesh::FH f) { return uint32_t(std::ranges::distance(FaceHalfedges(mesh, f))); }
inline vec3 Centroid(const Mesh &mesh, Mesh::FH f) {
    vec3 centroid{0};
    uint32_t count = 0;
    for (const auto v : FaceVertices(mesh, f)) {
        centroid += Position(mesh, v);
        ++count;
    }
    return centroid / float(count);
}
// The face polygon's vector area, half the sum of its consecutive corners' cross products.
inline vec3 AreaVector(const Mesh &mesh, Mesh::FH f) {
    std::vector<vec3> corners;
    for (const auto v : FaceVertices(mesh, f)) corners.push_back(Position(mesh, v));
    vec3 area{};
    for (size_t i = 0; i < corners.size(); ++i) area += Cross(corners[i], corners[(i + 1) % corners.size()]);
    return .5f * area;
}
} // namespace test
