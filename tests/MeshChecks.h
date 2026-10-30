#pragma once

#include "TestMeshOrdinals.h"
#include "mesh/MeshStore.h"

#include <boost/ut.hpp>

#include <map>

// Whole-record invariants over a mesh's selection index, render primitives and polygon tessellation.
namespace test {
inline std::string Describe(std::span<const vec3> points) {
    std::string text;
    for (const auto p : points) text += std::format("({:g} {:g} {:g}) ", p.x, p.y, p.z);
    return text;
}
// The selection roots, hidden sets, bounds and boundary edges of a record agree with a recount over its live elements.
inline void ExpectSelectionIndex(const MeshStore &meshes, uint32_t id) {
    if (!meshes.Get(id).SelectionSummary.Count) return;
    const Mesh mesh{meshes, id};
    const std::array elements{Element::Vertex, Element::Edge, Element::Face};
    const std::array domains{MeshStore::ElementDomain::Vertex, MeshStore::ElementDomain::Edge, MeshStore::ElementDomain::Face};
    for (uint32_t d = 0u; d < elements.size(); ++d) {
        const auto selection = meshes.GetSelectedElements(id, elements[d]);
        const auto hidden = meshes.GetHiddenElements(id, elements[d]);
        std::vector<uint32_t> expected_hidden, actual_hidden;
        std::vector<uint32_t> expected, actual;
        uint32_t live_count = 0u;
        vec3 sum{};
        AABB bounds;
        WithDomain(meshes.Arenas(), domains[d], [&](const auto &arena) {
            const auto set = DomainSet(meshes.Get(id), domains[d]);
            arena.ForEach(set, [&](ElementHandle handle, ElementAddress at) {
                const auto v = handle.Value, address = at.Value();
                ++live_count;
                const bool selected = selection.Bits[address / 32u] & (1u << (address % 32u));
                if (selected) expected.push_back(v);
                if (hidden.Contains(v)) {
                    expected_hidden.push_back(v);
                    boost::ut::expect(!selected);
                }
                if (d == 0u) {
                    const auto p = mesh.GetPosition(he::VH{v});
                    bounds.Min = Min(bounds.Min, p);
                    bounds.Max = Max(bounds.Max, p);
                    if (selected) sum += p;
                } else if (d == 2u) {
                    for (const auto vertex : mesh.fv_range(he::FH{v})) {
                        const auto p = mesh.GetPosition(vertex);
                        bounds.Min = Min(bounds.Min, p);
                        bounds.Max = Max(bounds.Max, p);
                    }
                }
            });
        });
        std::ranges::sort(expected);
        selection.ForEach([&](uint32_t v) { actual.push_back(v); });
        boost::ut::expect(actual == expected);
        boost::ut::expect(selection.Count() == expected.size());
        boost::ut::expect(selection.First() == (expected.empty() ? std::optional<uint32_t>{} : expected.front()));
        boost::ut::expect(selection.Last() == (expected.empty() ? std::optional<uint32_t>{} : expected.back()));
        const auto &root = meshes.GetSelectionRoot(id, elements[d]);
        boost::ut::expect(root.LiveCount == live_count);
        std::ranges::sort(expected_hidden);
        hidden.ForEach([&](uint32_t h) { actual_hidden.push_back(h); });
        boost::ut::expect(actual_hidden == expected_hidden);
        boost::ut::expect(root.Hidden == expected_hidden.size());
        boost::ut::expect(hidden.Count() == expected_hidden.size());
        boost::ut::expect(Length(root.PositionSum - sum) < 0.0001f * (1.f + Length(sum)));
        boost::ut::expect(root.Bounds == bounds) << "domain" << d << "root" << Describe(std::array{root.Bounds.Min, root.Bounds.Max}) << "elements" << Describe(std::array{bounds.Min, bounds.Max});
    }
    std::vector<uint32_t> expected_boundary, actual_boundary;
    for (const auto e : mesh.edges())
        if (!mesh.GetOppositeHalfedge(mesh.GetHalfedge(e, 0u))) expected_boundary.push_back(*e);
    std::ranges::sort(expected_boundary);
    meshes.GetBoundaryEdges(id).ForEach([&](uint32_t e) { actual_boundary.push_back(e); });
    boost::ut::expect(expected_boundary == actual_boundary);
}

inline std::array<uint32_t, 3> RenderedPrimitiveCounts(const MeshStore &meshes, uint32_t id) {
    std::array<uint32_t, 3> counts{};
    const auto set = meshes.Get(id).MeshletData;
    meshes.Render().Meshlets.ForEach(set, [&](ElementHandle, ElementAddress address) {
        const auto &cluster = meshes.Render().Meshlets.Get({address.Value(), 1u})[0];
        counts[cluster.Topology] += cluster.TriangleCount;
    });
    return counts;
}

// The rendered wires and points match the mesh's loose edges and isolated vertices, and so do the triangles of a record without coarser cluster levels.
inline void ExpectRenderedGeometry(const MeshStore &meshes, const Mesh &mesh) {
    std::array<uint32_t, 3> expected{mesh.TriangleIndexCount() / 3u, 0u, 0u};
    for (const auto edge : mesh.edges()) expected[1] += !mesh.GetConnectivity().FaceOf(mesh.GetHalfedge(edge, 0u));
    for (const auto vertex : mesh.vertices()) expected[2] += mesh.GetConnectivity().Incoming(vertex).y == 0u;
    auto actual = RenderedPrimitiveCounts(meshes, mesh.GetStoreId());
    if (meshes.ClusterGroupCount(meshes.Get(mesh.GetStoreId())) > 0u) actual[0] = expected[0];
    boost::ut::expect(actual == expected) << "rendered" << actual[0] << actual[1] << actual[2] << "elements" << expected[0] << expected[1] << expected[2];
}

// The face's first `count` render triangles, whose corners are halfedge handles.
// Face triangles hold triangle handles, which a copied record places at other arena addresses.
inline std::vector<uvec3> RenderTriangles(const MeshStore &meshes, const Mesh &mesh, he::FH face, uint32_t count) {
    const auto &a = meshes.Arenas();
    const auto first = a.FaceTriangles.Get({test::Address(mesh, face), 1u})[0];
    std::vector<uvec3> triangles;
    a.Triangles.ForEachAddressRun(meshes.Get(mesh.GetStoreId()).TriangleData, ElementHandle{first}, count, [&](Range run) { triangles.append_range(a.Triangles.Get(run)); });
    return triangles;
}
// Whether the planar loop neither revisits a vertex nor crosses itself, since no tessellation covers a pinched or twisted polygon once.
inline bool Simple(std::span<const vec3> points, vec3 normal, float extent) {
    const auto n = points.size();
    for (size_t i = 0u; i < n; ++i)
        for (size_t j = 0u; j < i; ++j)
            if (Length(points[i] - points[j]) <= 1e-6f * extent) return false;
    const auto u = Normalize(points[1] - points[0]), v = Cross(normal, u);
    const auto plane = [&](vec3 p) { return std::array{Dot(p - points[0], u), Dot(p - points[0], v)}; };
    const auto orient = [](std::array<float, 2> a, std::array<float, 2> b, std::array<float, 2> c) { return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]); };
    for (size_t i = 0u; i < n; ++i)
        for (size_t j = i + 2u; j < n; ++j) {
            if (i == 0u && j + 1u == n) continue;
            const auto a = plane(points[i]), b = plane(points[(i + 1u) % n]), c = plane(points[j]), d = plane(points[(j + 1u) % n]);
            if (orient(a, b, c) * orient(a, b, d) < 0.f && orient(c, d, a) * orient(c, d, b) < 0.f) return false;
        }
    return true;
}
// Validate the render triangles independently of the triangulation algorithm.
// Every triangle corner is a corner of the face and the triangles' directed edges cancel against the face loop.
// A planar simple face's triangles also wind with the face and cover its area exactly once.
inline void ExpectPolygonTessellation(const MeshStore &meshes, const Mesh &mesh) {
    const auto position = [&](he::VH vertex) { return mesh.GetPosition(vertex); };
    for (const auto face : mesh.faces()) {
        std::vector<vec3> points;
        for (const auto vertex : mesh.fv_range(face)) points.push_back(position(vertex));
        const auto origin = points.front();
        vec3 area{};
        for (uint32_t i = 1u; i + 1u < points.size(); ++i) area += Cross(points[i] - origin, points[i + 1u] - origin);
        // A collapsed polygon has no area to tessellate.
        const auto length = Length(area);
        if (!(length > 0.f)) continue;
        const auto normal = area / length;
        float extent = 0.f, height = 0.f;
        for (const auto p : points) {
            extent = std::max(extent, Length(p - origin));
            height = std::max(height, std::abs(Dot(p - origin, normal)));
        }
        const bool planar = height <= 1e-5f * extent && Simple(points, normal, extent);
        double triangle_area = 0.0;
        std::map<std::pair<uint32_t, uint32_t>, int> boundary;
        const auto edge = [&](uint32_t a, uint32_t b, int weight) { boundary[std::minmax(a, b)] += a < b ? weight : -weight; };
        std::set<uint32_t> corners;
        for (const auto h : mesh.fh_range(face)) {
            corners.insert(*h);
            edge(*mesh.GetConnectivity().Previous(h), *h, -1);
        }
        for (const auto triangle : RenderTriangles(meshes, mesh, face, uint32_t(points.size() - 2u))) {
            for (uint32_t k = 0u; k < 3u; ++k) {
                boost::ut::expect(corners.contains(triangle[k]));
                edge(triangle[k], triangle[(k + 1u) % 3u], 1);
            }
            const auto a = position(mesh.GetToVertex(he::HH{triangle.x}));
            const auto b = position(mesh.GetToVertex(he::HH{triangle.y}));
            const auto c = position(mesh.GetToVertex(he::HH{triangle.z}));
            const auto signed_area = Dot(Cross(b - a, c - a), normal);
            if (planar) boost::ut::expect(signed_area >= -length * 1e-5f) << "triangle winds outside polygon" << Describe(points) << "triangle" << Describe(std::array{a, b, c}) << "area" << signed_area;
            triangle_area += std::abs(signed_area);
        }
        if (planar) boost::ut::expect(std::abs(triangle_area - length) < length * 1e-4) << "render triangles overlap or leave a gap" << Describe(points) << "triangles" << triangle_area << "polygon" << length;
        for (const auto &[ends, count] : boundary) boost::ut::expect(count == 0) << "triangle edge differs from polygon boundary" << ends.first << ends.second;
    }
}
} // namespace test
