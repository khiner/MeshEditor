#pragma once

#include "PrimitiveType.h"
#include "mesh/MeshData.h"
#include "numeric/vec2.h"

#include <array>
#include <ranges>
#include <unordered_map>

using std::views::transform, std::ranges::iota_view, std::ranges::to;

inline MeshData Rect(vec2 half_extents = {0.5, 0.5}) {
    const auto x = half_extents.x, y = half_extents.y;
    return {
        {{-x, -y, 0}, {x, -y, 0}, {x, y, 0}, {-x, y, 0}},
        {{0, 1, 2, 3}}
    };
}

inline MeshData Circle(float radius = 0.5, uint n = 32) {
    std::vector<vec3> vertices =
        iota_view{0u, n} |
        transform([radius, n](uint i) { return vec3{radius * __cospi(2.f * i / n), radius * __sinpi(2.f * i / n), 0}; }) |
        to<std::vector>();
    vertices.emplace_back(0, 0, 0); // Center vertex

    return {
        std::move(vertices),
        iota_view{0u, n} | transform([n](uint i) { return std::vector<uint>{i, (i + 1) % n, n}; }) | to<std::vector>()
    };
}

inline MeshData Cuboid(vec3 half_extents = {0.5, 0.5, 0.5}) {
    const auto x = half_extents.x, y = half_extents.y, z = half_extents.z;
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

inline MeshData IcoSphere(float radius = 0.5, uint recursion_level = 3) {
    static const float t = (1.f + sqrt(5.f)) / 2.f;
    // clang-format off
    std::vector<vec3> vertices{
        {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
        {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
        {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1},
    };
    for (auto &v : vertices) v = glm::normalize(v);

    std::vector<std::vector<uint>> indices{
        {0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
        {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
        {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1},
    };
    // clang-format on

    std::unordered_map<uint, uint> cache; // Edge midpoint index cache.
    auto AddMidVertex = [&](uint p1, uint p2) -> uint {
        const uint key = (std::min(p1, p2) << 16) + std::max(p1, p2);
        if (const auto found = cache.find(key); found != cache.end()) return found->second;

        vertices.emplace_back(glm::normalize((vertices[p1] + vertices[p2]) / 2.f));

        const uint i = vertices.size() - 1;
        cache[key] = i;
        return i;
    };

    for (uint r = 0; r < recursion_level; ++r) {
        std::vector<std::vector<uint>> new_indices;
        new_indices.reserve(indices.size() * 4);
        for (const auto &tri : indices) {
            const uint a = tri[0], b = tri[1], c = tri[2];
            const uint ab = AddMidVertex(a, b), bc = AddMidVertex(b, c), ca = AddMidVertex(c, a);
            new_indices.insert(new_indices.end(), {{a, ab, ca}, {b, bc, ab}, {c, ca, bc}, {ab, bc, ca}});
        }
        indices = std::move(new_indices);
    }

    for (auto &v : vertices) v *= radius;

    return {std::move(vertices), std::move(indices)};
}

inline MeshData UVSphere(float radius = 0.5, uint n_slices = 32, uint n_stacks = 16) {
    std::vector<vec3> vertices;
    vertices.reserve(2 + n_slices * (n_stacks - 2)); // +/- 2 for the poles
    vertices.emplace_back(0, radius, 0); // Top pole
    // Vertices (excluding poles)
    for (uint i = 1; i < n_stacks; ++i) {
        const float p = float(i) / float(n_stacks);
        for (uint j = 0; j < n_slices; ++j) {
            const float t = 2.f * j / n_slices;
            vertices.emplace_back(vec3{__sinpi(p) * __cospi(t), __cospi(p), __sinpi(p) * __sinpi(t)} * radius);
        }
    }
    vertices.emplace_back(0, -radius, 0); // Bottom pole

    std::vector<std::vector<uint>> indices;
    indices.reserve(2 * n_slices + n_slices * (n_stacks - 2)); // Top + bottom triangles + quads
    // Top triangles
    for (uint i = 0; i < n_slices; ++i) indices.push_back({0, 1 + (i + 1) % n_slices, 1 + i});
    // Quads per stack / slice
    for (uint j = 0; j < n_stacks - 2; ++j) {
        const uint j0 = 1 + j * n_slices, j1 = 1 + (j + 1) * n_slices;
        for (uint i = 0; i < n_slices; ++i) {
            indices.push_back({
                j0 + i,
                j0 + (i + 1) % n_slices,
                j1 + (i + 1) % n_slices,
                j1 + i,
            });
        }
    }
    // Bottom triangles
    const uint bottom_i = vertices.size() - 1;
    for (uint i = 0; i < n_slices; ++i) {
        indices.push_back({
            bottom_i,
            1 + (n_stacks - 2) * n_slices + i,
            1 + (n_stacks - 2) * n_slices + (i + 1) % n_slices,
        });
    }

    return {std::move(vertices), std::move(indices)};
}

inline MeshData Torus(float major_radius = 0.5, float minor_radius = 0.2, uint n_major = 32, uint n_minor = 16) {
    std::vector<vec3> vertices;
    vertices.reserve(n_major * n_minor);
    for (uint i = 0; i < n_major; ++i) {
        const float t = 2.f * i / n_major;
        for (uint j = 0; j < n_minor; ++j) {
            const float p = 2.f * j / n_minor;
            const float r = major_radius + minor_radius * __cospi(p);
            vertices.emplace_back(r * __sinpi(t), minor_radius * __sinpi(p), r * __cospi(t));
        }
    }

    std::vector<std::vector<uint>> indices;
    indices.reserve(n_major * n_minor);
    // Generate quads for the torus surface, maintaining original winding order
    for (uint i = 0; i < n_major; ++i) {
        for (uint j = 0; j < n_minor; ++j) {
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

inline MeshData Cylinder(float radius = 0.5, float height = 1, uint slices = 32) {
    std::vector<vec3> vertices(2 * slices);
    for (uint i = 0; i < slices; i++) {
        const float a = 2.f * i / slices;
        const float x = __cospi(a), z = __sinpi(a);
        vertices[i] = {x * radius, -height / 2, z * radius}; // bottom face
        vertices[i + slices] = {x * radius, height / 2, z * radius}; // top face
    }

    std::vector<std::vector<uint>> faces(slices + 2);
    // Bottom n-gon
    faces[0] = iota_view{0u, slices} | to<std::vector>();
    // Side quads
    for (uint i = 0; i < slices; ++i) {
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

inline MeshData Cone(float radius = 0.5, float height = 1, uint slices = 32) {
    std::vector<vec3> vertices = // Base
        iota_view{0u, slices} | transform([&](uint i) {
            return vec3{radius * __cospi(2.f * i / slices), -height / 2, radius * __sinpi(2.f * i / slices)};
        }) |
        to<std::vector>();
    vertices.emplace_back(0, height / 2, 0); // Top

    std::vector<std::vector<uint>> indices = // Side triangles
        iota_view{0u, slices} |
        transform([slices](uint i) { return std::vector<uint>{slices, (i + 1) % slices, i}; }) | to<std::vector>();
    indices.emplace_back(iota_view{0u, slices} | to<std::vector>()); // Base n-gon

    return {std::move(vertices), std::move(indices)};
}

constexpr std::string ToString(PrimitiveType type) {
    using enum PrimitiveType;
    switch (type) {
        case Rect: return "Rect";
        case Circle: return "Circle";
        case Cube: return "Cube";
        case IcoSphere: return "IcoSphere";
        case UVSphere: return "UVSphere";
        case Torus: return "Torus";
        case Cylinder: return "Cylinder";
        case Cone: return "Cone";
    }
}

inline MeshData CreateDefaultPrimitive(PrimitiveType type) {
    switch (type) {
        case PrimitiveType::Rect: return Rect();
        case PrimitiveType::Cube: return Cuboid();
        case PrimitiveType::IcoSphere: return IcoSphere();
        case PrimitiveType::Circle: return Circle();
        case PrimitiveType::UVSphere: return UVSphere();
        case PrimitiveType::Torus: return Torus();
        case PrimitiveType::Cylinder: return Cylinder();
        case PrimitiveType::Cone: return Cone();
    }
}

constexpr std::array PrimitiveTypes{
    PrimitiveType::Rect,
    PrimitiveType::Circle,
    PrimitiveType::Cube,
    PrimitiveType::IcoSphere,
    PrimitiveType::UVSphere,
    PrimitiveType::Torus,
    PrimitiveType::Cylinder,
    PrimitiveType::Cone,
};
