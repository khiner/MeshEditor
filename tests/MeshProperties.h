#pragma once

#include "MeshWalk.h"

#include <set>
#include <span>
#include <vector>

// Geometric properties of a mesh record that hold whatever order an operator builds it in.
namespace test {
// The signed volume each closed component of faces joined across their edges encloses, in the order of their lowest faces.
inline std::vector<double> ClosedComponentVolumes(const Mesh &mesh) {
    std::vector<double> volumes;
    std::vector<bool> visited(mesh.FaceCount());
    for (const auto seed : mesh.faces()) {
        if (visited[mesh.FaceOrdinal(seed)]) continue;
        visited[mesh.FaceOrdinal(seed)] = true;
        std::vector<Mesh::FH> component{seed};
        bool closed = true;
        for (size_t i = 0; i < component.size(); ++i) {
            for (const auto h : FaceHalfedges(mesh, component[i])) {
                const auto opposite = Opposite(mesh, h);
                const auto face = opposite ? Face(mesh, opposite) : Mesh::FH{};
                if (!face) closed = false;
                else if (!visited[mesh.FaceOrdinal(face)]) {
                    visited[mesh.FaceOrdinal(face)] = true;
                    component.push_back(face);
                }
            }
        }
        if (!closed) continue;
        // Measuring from a corner of the component keeps precision.
        const numeric::dvec3 origin{Position(mesh, *FaceVertices(mesh, seed).begin())};
        double volume = 0.0;
        for (const auto face : component) {
            std::vector<numeric::dvec3> corners;
            for (const auto v : FaceVertices(mesh, face)) corners.push_back(numeric::dvec3{Position(mesh, v)} - origin);
            for (size_t k = 1; k + 1 < corners.size(); ++k) volume += Dot(corners[0], Cross(corners[k], corners[k + 1]));
        }
        volumes.push_back(volume / 6.0);
    }
    return volumes;
}
// Whether every closed component encloses positive signed volume.
inline bool ClosedComponentsOutward(const Mesh &mesh) {
    return std::ranges::all_of(ClosedComponentVolumes(mesh), [](double volume) { return volume > 0.0; });
}

// Whether each edge between two of the triangles passes the empty-circle test in the plane normal to their summed area vectors, as Beautify projects it.
// An edge whose rotation folds the quad or collapses a triangle in that plane has no alternative and passes, and a collapsed triangle beside a valid rotation fails.
inline bool Delaunay(const Mesh &mesh, std::span<const uint32_t> faces) {
    const std::set<uint32_t> selected{faces.begin(), faces.end()};
    const auto triangle = [&](Mesh::FH face) { return face && selected.contains(*face) && Valence(mesh, face) == 3u; };
    for (const auto handle : selected) {
        if (!triangle(Mesh::FH{handle})) continue;
        for (const auto h : FaceHalfedges(mesh, Mesh::FH{handle})) {
            const auto opposite = Opposite(mesh, h);
            const auto other = opposite ? Face(mesh, opposite) : Mesh::FH{};
            if (!triangle(other) || *other <= handle) continue;
            // The edge runs from p to q, with r across it in this face and s in the other, all relative to p.
            const numeric::dvec3 p{Position(mesh, ToVertex(mesh, opposite))};
            const auto at = [&](Mesh::HH corner) { return numeric::dvec3{Position(mesh, ToVertex(mesh, corner))} - p; };
            const auto q = at(h), r = at(NextInFace(mesh, h)), s = at(NextInFace(mesh, opposite));
            const auto normal = Cross(q, r) + Cross(s, q);
            const auto scale = Length(normal);
            const auto along = q - normal * (Dot(q, normal) / (scale * scale));
            if (!(scale > 0.0) || !(Length(along) > 0.0)) continue;
            const auto u = along / Length(along), v = Cross(normal / scale, u);
            const auto plane = [&](numeric::dvec3 point) { return std::array{Dot(point, u), Dot(point, v)}; };
            const std::array<double, 2> p2{0.0, 0.0};
            const auto q2 = plane(q), r2 = plane(r), s2 = plane(s);
            const auto orient = [](std::array<double, 2> a, std::array<double, 2> b, std::array<double, 2> c) { return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]); };
            const auto sign = [&](double area) { return area > 1e-5 * scale ? 1 : area < -1e-5 * scale ? -1 :
                                                                                                         0; };
            const auto old_a = orient(p2, q2, r2), old_b = orient(q2, p2, s2), new_a = orient(q2, r2, s2), new_b = orient(r2, p2, s2);
            if (!(sign(old_a) + sign(old_b))) continue;
            if ((new_a >= 0.0) != (new_b >= 0.0) || !sign(new_a) || !sign(new_b) || (old_a >= 0.0) != (old_b >= 0.0)) continue;
            if (!sign(old_a) || !sign(old_b)) return false;
            // s lies inside the circle through p, q and r when the lifted determinant has the sign of their winding.
            const auto row = [&](std::array<double, 2> a) { return std::array{a[0] - s2[0], a[1] - s2[1], (a[0] - s2[0]) * (a[0] - s2[0]) + (a[1] - s2[1]) * (a[1] - s2[1])}; };
            const auto a = row(p2), b = row(q2), c = row(r2);
            const auto lifted = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
            const auto extent = std::max({Length(q), Length(r), Length(s)});
            if ((old_a > 0.0 ? lifted : -lifted) > 1e-5 * extent * extent * extent * extent) return false;
        }
    }
    return true;
}

// Whether the mesh has at most `faces` faces and every vertex of each surface lies within `tolerance` of the other surface's render triangles.
inline bool SimplifiedWithin(const Mesh &mesh, const Mesh &before, uint32_t faces, float tolerance) {
    if (mesh.FaceCount() > faces) return false;
    // The nearest point on a triangle comes from Ericson's closest point regions.
    const auto within = [tolerance](const Mesh &from, const Mesh &to) {
        std::vector<std::array<numeric::dvec3, 3>> triangles;
        for (const auto corners : to.DerivedTriangles()) {
            const auto [a, b, c] = TrianglePositions(to, corners);
            triangles.push_back({numeric::dvec3{a}, numeric::dvec3{b}, numeric::dvec3{c}});
        }
        const auto closest = [](numeric::dvec3 p, numeric::dvec3 a, numeric::dvec3 b, numeric::dvec3 c) {
            const auto ab = b - a, ac = c - a, ap = p - a;
            const auto d1 = Dot(ab, ap), d2 = Dot(ac, ap);
            if (d1 <= 0.0 && d2 <= 0.0) return a;
            const auto bp = p - b;
            const auto d3 = Dot(ab, bp), d4 = Dot(ac, bp);
            if (d3 >= 0.0 && d4 <= d3) return b;
            const auto vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return a + ab * (d1 / (d1 - d3));
            const auto cp = p - c;
            const auto d5 = Dot(ab, cp), d6 = Dot(ac, cp);
            if (d6 >= 0.0 && d5 <= d6) return c;
            const auto vb = d5 * d2 - d1 * d6;
            if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return a + ac * (d2 / (d2 - d6));
            const auto va = d3 * d6 - d5 * d4;
            if (va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0) return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
            const auto denominator = va + vb + vc;
            return a + ab * (vb / denominator) + ac * (vc / denominator);
        };
        for (const auto vertex : from.vertices()) {
            const numeric::dvec3 p{Position(from, vertex)};
            double nearest = 1e300;
            for (const auto &[a, b, c] : triangles) nearest = std::min(nearest, Length(closest(p, a, b, c) - p));
            if (!(nearest <= double(tolerance))) return false;
        }
        return true;
    };
    return within(mesh, before) && within(before, mesh);
}
// The volume a closed manifold surface encloses, from the signed tetrahedra its triangles span with the origin.
inline std::optional<double> EnclosedVolume(const Mesh &mesh) {
    if (!mesh.HasClosedSurface()) return std::nullopt;
    double volume = 0.0;
    for (const auto corners : mesh.DerivedTriangles()) {
        const auto [a, b, c] = TrianglePositions(mesh, corners);
        volume += Dot(numeric::dvec3{a}, Cross(numeric::dvec3{b}, numeric::dvec3{c})) / 6.0;
    }
    return std::abs(volume);
}
} // namespace test
