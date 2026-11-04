#pragma once

#include "MeshChecks.h"
#include "gpu/PolygonTriangulation.h"
#include "mesh/MeshData.h"

#include <manifold/polygon.h>
#include <numeric>

// Agreement of a mesh record with manifold on the same input, for test targets that link manifold.
namespace test {
using Triangles = std::vector<std::array<uint32_t, 3>>;

// The triangle starting at its least corner, keeping its winding.
inline std::array<uint32_t, 3> Canonical(std::array<uint32_t, 3> triangle) {
    std::ranges::rotate(triangle, std::ranges::min_element(triangle));
    return triangle;
}

// manifold::Triangulate's sorted canonical triangles of a loop, projected into the plane through its first corner normal to its vector area.
inline Triangles ManifoldTriangles(std::span<const vec3> loop) {
    float magnitude = 0.f;
    std::vector<numeric::dvec3> points;
    for (const auto p : loop) {
        magnitude = std::max({magnitude, std::abs(p.x), std::abs(p.y), std::abs(p.z)});
        points.emplace_back(p);
    }
    numeric::dvec3 area{};
    for (size_t i = 0; i < points.size(); ++i) area += Cross(points[i], points[(i + 1) % points.size()]);
    const auto normal = Normalize(area);
    // A right-handed frame keeps the loop's winding about the normal, which manifold reads as counterclockwise.
    const auto u = Normalize(Cross(std::abs(normal.x) < .5 ? numeric::dvec3{1, 0, 0} : numeric::dvec3{0, 1, 0}, normal)), v = Cross(normal, u);
    manifold::SimplePolygon polygon;
    for (const auto p : points) polygon.emplace_back(Dot(p - points[0], u), Dot(p - points[0], v));
    Triangles triangles;
    for (const auto t : manifold::Triangulate({polygon}, double(RelativePrecision * magnitude))) triangles.push_back(Canonical({uint32_t(t.x), uint32_t(t.y), uint32_t(t.z)}));
    std::ranges::sort(triangles);
    return triangles;
}

// Whether the face's render triangles are manifold's triangles of its loop from the same first corner.
inline bool TriangulatesLikeManifold(const Mesh &mesh, Mesh::FH face) {
    std::vector<Mesh::HH> corners;
    std::vector<vec3> loop;
    for (const auto h : FaceHalfedges(mesh, face)) {
        corners.push_back(h);
        loop.push_back(Position(mesh, ToVertex(mesh, h)));
    }
    const auto local = [&](uint32_t h) { return uint32_t(std::ranges::find(corners, Mesh::HH{h}) - corners.begin()); };
    Triangles actual;
    for (const auto t : RenderTriangles(mesh.GetStore(), mesh, face, uint32_t(corners.size() - 2u))) actual.push_back(Canonical({local(t.x), local(t.y), local(t.z)}));
    std::ranges::sort(actual);
    return actual == ManifoldTriangles(loop);
}

// One-face meshes whose triangulations rows compare with manifold's.
inline std::vector<MeshData> TriangulationPolygons() {
    const auto face = [](std::vector<vec3> positions) {
        std::vector<uint32_t> loop(positions.size());
        std::iota(loop.begin(), loop.end(), 0u);
        return MeshData{std::move(positions), {loop}};
    };
    const auto tilted = [&](std::vector<vec2> loop) {
        std::vector<vec3> positions;
        for (const auto p : loop) positions.emplace_back(p.x, p.y, .3f * p.x + .2f * p.y);
        return face(std::move(positions));
    };
    return {
        tilted({{0.f, 0.f}, {2.f, .1f}, {2.2f, 1.5f}, {-.1f, 1.2f}}),
        tilted({{0.f, 0.f}, {1.5f, -.2f}, {2.6f, .7f}, {2.4f, 1.9f}, {1.1f, 2.4f}, {-.3f, 1.3f}}),
        tilted({{0.f, 0.f}, {3.f, 0.f}, {3.f, 1.f}, {1.7f, 1.2f}, {2.6f, 2.1f}, {2.9f, 3.2f}, {.4f, 3.f}, {1.2f, 1.6f}}),
        face({{0.f, 0.f, 0.f}, {1.f, 0.f, .3f}, {1.1f, 1.f, 0.f}, {0.f, .9f, .4f}}),
        face({{0.f, 0.f, 0.f}, {2.f, .5f, .2f}, {0.f, 1.f, .5f}, {.6f, .5f, -.3f}}),
    };
}
} // namespace test
