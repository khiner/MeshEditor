#pragma once

#include "PrimitiveType.h"
#include "mesh/MeshAttributes.h"
#include "mesh/MeshData.h"

#include <glm/geometric.hpp>

#include <ranges>
#include <unordered_map>

namespace primitive {
using std::views::transform, std::ranges::iota_view, std::ranges::to;

inline MeshData CreateMesh(const Rect &p) {
    const auto x = p.HalfExtents.x, z = p.HalfExtents.y;
    return {
        {{-x, 0, z}, {x, 0, z}, {x, 0, -z}, {-x, 0, -z}},
        {{0, 1, 2, 3}}
    };
}

inline MeshData CreateMesh(const Circle &p) {
    const auto [radius, n] = p;
    std::vector vertices =
        iota_view{0u, n} |
        transform([radius, n](uint32_t i) { return vec3{radius * __cospi(2.f * i / n), radius * __sinpi(2.f * i / n), 0}; }) |
        to<std::vector>();
    vertices.emplace_back(0, 0, 0); // Center vertex

    return {
        std::move(vertices),
        iota_view{0u, n} | transform([n](uint32_t i) { return std::vector<uint32_t>{i, (i + 1) % n, n}; }) | to<std::vector>()
    };
}

inline MeshData CreateMesh(const Cuboid &p) {
    const auto x = p.HalfExtents.x, y = p.HalfExtents.y, z = p.HalfExtents.z;
    return {
        {
            {-x, -y, -z},
            {x, -y, -z},
            {x, y, -z},
            {-x, y, -z},
            {-x, -y, z},
            {x, -y, z},
            {x, y, z},
            {-x, y, z},
        },
        {
            {0, 3, 2, 1}, // front
            {4, 5, 6, 7}, // back
            {0, 1, 5, 4}, // bottom
            {3, 7, 6, 2}, // top
            {0, 4, 7, 3}, // left
            {1, 2, 6, 5}, // right
        }
    };
}

inline MeshData CreateMesh(const IcoSphere &p) {
    const auto [radius, recursion_level] = p;
    static const float t = (1.f + sqrt(5.f)) / 2.f;
    // clang-format off
    std::vector<vec3> vertices{
        {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
        {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
        {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1},
    };
    for (auto &v : vertices) v = glm::normalize(v);

    std::vector<std::vector<uint32_t>> indices{
        {0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
        {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
        {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1},
    };
    // clang-format on

    std::unordered_map<uint32_t, uint32_t> cache; // Edge midpoint index cache.
    auto AddMidVertex = [&](uint32_t p1, uint32_t p2) -> uint32_t {
        const uint32_t key = (std::min(p1, p2) << 16) + std::max(p1, p2);
        if (const auto found = cache.find(key); found != cache.end()) return found->second;

        vertices.emplace_back(glm::normalize((vertices[p1] + vertices[p2]) / 2.f));

        const uint32_t i = vertices.size() - 1;
        cache[key] = i;
        return i;
    };

    for (uint32_t r = 0; r < recursion_level; ++r) {
        std::vector<std::vector<uint32_t>> new_indices;
        new_indices.reserve(indices.size() * 4);
        for (const auto &tri : indices) {
            const uint32_t a = tri[0], b = tri[1], c = tri[2];
            const uint32_t ab = AddMidVertex(a, b), bc = AddMidVertex(b, c), ca = AddMidVertex(c, a);
            new_indices.insert(new_indices.end(), {{a, ab, ca}, {b, bc, ab}, {c, ca, bc}, {ab, bc, ca}});
        }
        indices = std::move(new_indices);
    }

    for (auto &v : vertices) v *= radius;

    return {std::move(vertices), std::move(indices)};
}

inline MeshData CreateMesh(const UVSphere &p) {
    const auto [radius, n_slices, n_stacks] = p;
    std::vector<vec3> vertices;
    vertices.reserve(2 + n_slices * (n_stacks - 2)); // +/- 2 for the poles
    vertices.emplace_back(0, radius, 0); // Top pole
    // Vertices (excluding poles)
    for (uint32_t i = 1; i < n_stacks; ++i) {
        const float pp = float(i) / float(n_stacks);
        for (uint32_t j = 0; j < n_slices; ++j) {
            const float tt = 2.f * j / n_slices;
            vertices.emplace_back(vec3{__sinpi(pp) * __cospi(tt), __cospi(pp), __sinpi(pp) * __sinpi(tt)} * radius);
        }
    }
    vertices.emplace_back(0, -radius, 0); // Bottom pole

    std::vector<std::vector<uint32_t>> indices;
    indices.reserve(2 * n_slices + n_slices * (n_stacks - 2)); // Top + bottom triangles + quads
    // Top triangles
    for (uint32_t i = 0; i < n_slices; ++i) indices.push_back({0, 1 + (i + 1) % n_slices, 1 + i});
    // Quads per stack / slice
    for (uint32_t j = 0; j < n_stacks - 2; ++j) {
        const uint32_t j0 = 1 + j * n_slices, j1 = 1 + (j + 1) * n_slices;
        for (uint32_t i = 0; i < n_slices; ++i) {
            indices.push_back({
                j0 + i,
                j0 + (i + 1) % n_slices,
                j1 + (i + 1) % n_slices,
                j1 + i,
            });
        }
    }
    // Bottom triangles
    const uint32_t bottom_i = vertices.size() - 1;
    for (uint32_t i = 0; i < n_slices; ++i) {
        indices.push_back({
            bottom_i,
            1 + (n_stacks - 2) * n_slices + i,
            1 + (n_stacks - 2) * n_slices + (i + 1) % n_slices,
        });
    }

    return {std::move(vertices), std::move(indices)};
}

inline MeshData CreateMesh(const Torus &p) {
    const auto [major_radius, minor_radius, n_major, n_minor] = p;
    std::vector<vec3> vertices;
    vertices.reserve(n_major * n_minor);
    for (uint32_t i = 0; i < n_major; ++i) {
        const float t = 2.f * i / n_major;
        for (uint32_t j = 0; j < n_minor; ++j) {
            const float pp = 2.f * j / n_minor;
            const float r = major_radius + minor_radius * __cospi(pp);
            vertices.emplace_back(r * __sinpi(t), minor_radius * __sinpi(pp), r * __cospi(t));
        }
    }

    std::vector<std::vector<uint32_t>> indices;
    indices.reserve(n_major * n_minor);
    // Generate quads for the torus surface, maintaining original winding order
    for (uint32_t i = 0; i < n_major; ++i) {
        for (uint32_t j = 0; j < n_minor; ++j) {
            indices.push_back({
                i * n_minor + j,
                ((i + 1) % n_major) * n_minor + j,
                ((i + 1) % n_major) * n_minor + (j + 1) % n_minor,
                i * n_minor + (j + 1) % n_minor,
            });
        }
    }

    return {std::move(vertices), std::move(indices)};
}

inline MeshData CreateMesh(const Cylinder &p) {
    const auto [radius, height, slices] = p;
    std::vector<vec3> vertices(2 * slices);
    for (uint32_t i = 0; i < slices; i++) {
        const float a = 2.f * i / slices;
        const float x = __cospi(a), z = __sinpi(a);
        vertices[i] = {x * radius, -height / 2, z * radius}; // bottom face
        vertices[i + slices] = {x * radius, height / 2, z * radius}; // top face
    }

    std::vector<std::vector<uint32_t>> faces(slices + 2);
    // Bottom n-gon
    faces[0] = iota_view{0u, slices} | to<std::vector>();
    // Side quads
    for (uint32_t i = 0; i < slices; ++i) {
        faces[i + 1] = {
            i, // bottom
            i + slices, // top
            (i + 1) % slices + slices, // top
            (i + 1) % slices, // bottom
        };
    }
    // Top n-gon, reversed for winding order
    faces[slices + 1] = iota_view{0u, slices} |
        transform([slices](int i) { return slices + i; }) |
        std::views::reverse | to<std::vector>();

    return {std::move(vertices), std::move(faces)};
}

inline MeshData CreateMesh(const Cone &p) {
    const auto [radius, height, slices] = p;
    std::vector vertices =
        iota_view{0u, slices} | transform([&](uint32_t i) {
            return vec3{radius * __cospi(2.f * i / slices), -height / 2, radius * __sinpi(2.f * i / slices)};
        }) |
        to<std::vector>(); // Base
    vertices.emplace_back(0, height / 2, 0); // Top

    std::vector<std::vector<uint32_t>> indices = // Side triangles
        iota_view{0u, slices} |
        transform([slices](uint32_t i) { return std::vector<uint32_t>{slices, (i + 1) % slices, i}; }) | to<std::vector>();
    indices.emplace_back(iota_view{0u, slices} | to<std::vector>()); // Base n-gon

    return {std::move(vertices), std::move(indices)};
}

inline MeshData CreateMesh(const PrimitiveShape &shape) {
    return std::visit([](const auto &s) -> MeshData { return CreateMesh(s); }, shape);
}

// Expanded bone octahedron data: mesh with per-face normals + wire/adjacency indices.
struct BoneOctahedronData {
    MeshData Mesh;
    MeshVertexAttributes Attrs;
    std::vector<uint32_t> AdjacencyIndices; // 48 indices for 12 edges × 4 (line adjacency)
};

// Octahedral bone primitive with diagonal-ring shape.
// Head at origin, tail at (0, length, 0). 6 unique positions + 24 expanded face verts = 30 total.
// Unique positions [0..5] are used by wire/adjacency draws. Face verts [6..29] have per-face normals.
inline BoneOctahedronData BoneOctahedron(float length = 1.0f) {
    const float d = length * 0.1f;
    const std::array<vec3, 6> unique{
        vec3{0, 0, 0}, // 0: head
        vec3{d, d, d}, // 1
        vec3{d, d, -d}, // 2
        vec3{-d, d, -d}, // 3
        vec3{-d, d, d}, // 4
        vec3{0, length, 0}, // 5: tail
    };

    // 8 triangles: bottom 4 + top 4 (Blender's bone_octahedral_solid_tris)
    // clang-format off
    constexpr std::array<std::array<uint32_t, 3>, 8> tris{{
        {2, 1, 0}, {3, 2, 0}, {4, 3, 0}, {1, 4, 0}, // bottom
        {5, 1, 2}, {5, 2, 3}, {5, 3, 4}, {5, 4, 1}, // top
    }};
    // clang-format on

    // Per-face normals (Blender's bone_octahedral_solid_normals)
    constexpr float Rsqrt2 = 0.7071068f; // 1/sqrt(2)
    constexpr std::array<vec3, 8> face_normals{{
        {Rsqrt2, -Rsqrt2, 0}, // bottom-right
        {0, -Rsqrt2, -Rsqrt2}, // bottom-back
        {-Rsqrt2, -Rsqrt2, 0}, // bottom-left
        {0, -Rsqrt2, Rsqrt2}, // bottom-front
        {0.9940f, 0.1100f, 0}, // top-right
        {0, 0.1100f, -0.9940f}, // top-back
        {-0.9940f, 0.1100f, 0}, // top-left
        {0, 0.1100f, 0.9940f}, // top-front
    }};

    // Build 30-vertex buffer: [0..5] = unique positions, [6..29] = expanded face verts
    std::vector<vec3> positions;
    positions.reserve(30);
    for (const auto &p : unique) positions.push_back(p);
    std::vector<vec3> normals(6, vec3{0}); // placeholder normals for unique verts
    normals.reserve(30);

    std::vector<std::vector<uint32_t>> faces;
    faces.reserve(8);
    for (uint32_t f = 0; f < 8; ++f) {
        const uint32_t base = 6 + f * 3;
        for (uint32_t v = 0; v < 3; ++v) {
            positions.push_back(unique[tris[f][v]]);
            normals.push_back(face_normals[f]);
        }
        faces.push_back({base, base + 1, base + 2});
    }

    // 12 edges for adjacency-based silhouette detection.
    // Each edge has 4 indices: [adj_left, edge_v0, edge_v1, adj_right]
    // adj_left: vertex where face cycle goes edge_v0 → adj_left → edge_v1
    // adj_right: vertex where face cycle goes edge_v0 → edge_v1 → adj_right
    // This ensures cross(adj_left - edge_v0, edge_v1 - edge_v0) gives outward-pointing normals.
    // Derived from face table: {2,1,0},{3,2,0},{4,3,0},{1,4,0},{5,1,2},{5,2,3},{5,3,4},{5,4,1}
    // clang-format off
    const std::vector<uint32_t> adjacency{
        // Mid-ring edges
        0, 1, 2, 5, // edge 1-2: f0 cycle 0->2->1->0 has 1->0->2, f4 cycle has 1->2->5
        0, 2, 3, 5, // edge 2-3: f1 cycle has 2->0->3, f5 cycle has 2->3->5
        0, 3, 4, 5, // edge 3-4: f2 cycle has 3->0->4, f6 cycle has 3->4->5
        0, 4, 1, 5, // edge 4-1: f3 cycle has 4->0->1, f7 cycle has 4->1->5
        // Head edges (from vertex 0)
        2, 0, 1, 4, // edge 0-1: f0 cycle has 0->2->1, f3 cycle has 0->1->4
        3, 0, 2, 1, // edge 0-2: f1 cycle has 0->3->2, f0 cycle has 0->2->1
        4, 0, 3, 2, // edge 0-3: f2 cycle has 0->4->3, f1 cycle has 0->3->2
        1, 0, 4, 3, // edge 0-4: f3 cycle has 0->1->4, f2 cycle has 0->4->3
        // Tail edges (from vertex 5)
        4, 5, 1, 2, // edge 5-1: f7 cycle has 5->4->1, f4 cycle has 5->1->2
        1, 5, 2, 3, // edge 5-2: f4 cycle has 5->1->2, f5 cycle has 5->2->3
        2, 5, 3, 4, // edge 5-3: f5 cycle has 5->2->3, f6 cycle has 5->3->4
        3, 5, 4, 1, // edge 5-4: f6 cycle has 5->3->4, f7 cycle has 5->4->1
    };
    // clang-format on

    MeshVertexAttributes attrs;
    attrs.Normals = std::move(normals);

    return {
        .Mesh{std::move(positions), std::move(faces)},
        .Attrs = std::move(attrs),
        .AdjacencyIndices = adjacency,
    };
}

// Billboard disc for bone joint spheres.
struct BoneSphereData {
    MeshData Mesh;
    std::vector<uint32_t> OutlineIndices; // Line list ring
};

// Disc primitive for ray-traced joint spheres. Circle of N segments in XY plane.
inline BoneSphereData BoneSphereDisc(float radius = 0.05f, uint32_t segments = 32) {
    // Generate ring vertices + center
    std::vector<vec3> positions;
    positions.reserve(segments + 1);
    for (uint32_t i = 0; i < segments; ++i) {
        float angle = 2.f * 3.14159265f * float(i) / float(segments);
        positions.push_back({radius * std::cos(angle), radius * std::sin(angle), 0});
    }
    positions.push_back({0, 0, 0}); // center

    // Triangle fan as triangle list
    std::vector<std::vector<uint32_t>> faces;
    faces.reserve(segments);
    for (uint32_t i = 0; i < segments; ++i) {
        faces.push_back({i, (i + 1) % segments, segments});
    }

    // Outline ring as line list
    std::vector<uint32_t> outline;
    outline.reserve(segments * 2);
    for (uint32_t i = 0; i < segments; ++i) {
        outline.push_back(i);
        outline.push_back((i + 1) % segments);
    }

    return {
        .Mesh{std::move(positions), std::move(faces)},
        .OutlineIndices = std::move(outline),
    };
}
} // namespace primitive
