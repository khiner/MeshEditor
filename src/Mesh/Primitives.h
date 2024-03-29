#pragma once

#include "numeric/vec2.h"

#include "Primitive.h"
#include "mesh/Mesh.h"

inline Mesh Rect(vec2 half_extents = {0.5, 0.5}) {
    const auto x = half_extents.x, y = half_extents.y;
    return {
        {{-x, -y, 0}, {x, -y, 0}, {x, y, 0}, {-x, y, 0}},
        {{0, 1, 2, 3}}
    };
}

inline Mesh Circle(float radius = 0.5, uint n = 32) {
    std::vector<vec3> vertices;
    std::vector<std::vector<uint>> indices;
    vertices.reserve(n + 1);
    indices.reserve(n);

    for (uint i = 0; i < n; ++i) {
        const float theta = 2 * M_PI * i / n;
        vertices.emplace_back(radius * vec3{cos(theta), sin(theta), 0});
        indices.push_back({i, (i + 1) % n, n});
    }
    vertices.emplace_back(0, 0, 0);
    return {std::move(vertices), std::move(indices)};
}

inline Mesh Cuboid(vec3 half_extents = {0.5, 0.5, 0.5}) {
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

inline Mesh IcoSphere(float radius = 0.5, uint recursion_level = 3) {
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
        for (auto &tri : indices) {
            const uint a = tri[0], b = tri[1], c = tri[2];
            const uint ab = AddMidVertex(a, b), bc = AddMidVertex(b, c), ca = AddMidVertex(c, a);
            new_indices.insert(new_indices.end(), {{a, ab, ca}, {b, bc, ab}, {c, ca, bc}, {ab, bc, ca}});
        }
        indices = std::move(new_indices);
    }

    for (auto &v : vertices) v *= radius;

    return Mesh{std::move(vertices), std::move(indices)};
}

inline Mesh UVSphere(float radius = 0.5, uint n_slices = 32, uint n_stacks = 16) {
    std::vector<vec3> vertices;
    std::vector<std::vector<uint>> indices;
    vertices.reserve(2 + n_slices * (n_stacks - 2)); // +/- 2 for the poles
    indices.reserve(2 * n_slices + n_slices * (n_stacks - 2)); // Top + bottom triangles + quads

    vertices.emplace_back(0, radius, 0); // Top pole

    // Vertices (excluding poles)
    for (uint i = 1; i < n_stacks; ++i) {
        const float phi = M_PI * float(i) / float(n_stacks);
        for (uint j = 0; j < n_slices; ++j) {
            const float theta = 2 * M_PI * float(j) / float(n_slices);
            vertices.emplace_back(vec3{sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta)} * radius);
        }
    }

    vertices.emplace_back(0, -radius, 0); // Bottom pole

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

inline Mesh Torus(float major_radius = 0.5, float minor_radius = 0.2, uint n_major = 32, uint n_minor = 16) {
    std::vector<vec3> vertices;
    std::vector<std::vector<uint>> indices;
    vertices.reserve(n_major * n_minor);
    indices.reserve(n_major * n_minor);

    for (uint i = 0; i < n_major; ++i) {
        const float theta = 2 * M_PI * float(i) / float(n_major);
        for (uint j = 0; j < n_minor; ++j) {
            const float phi = 2 * M_PI * float(j) / float(n_minor);
            const float radial_distance = major_radius + minor_radius * cos(phi);
            vertices.emplace_back(radial_distance * sin(theta), minor_radius * sin(phi), radial_distance * cos(theta));
        }
    }

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

inline Mesh Cylinder(float radius = 0.5, float height = 1, uint slices = 32) {
    std::vector<vec3> vertices(2 * slices);
    std::vector<std::vector<uint>> faces(slices + 2);

    for (uint i = 0; i < slices; i++) {
        const float angle = 2 * M_PI * float(i) / float(slices);
        const float x = cos(angle), z = sin(angle);
        vertices[i] = {x * radius, -height / 2, z * radius}; // bottom face
        vertices[i + slices] = {x * radius, height / 2, z * radius}; // top face
    }

    // Bottom n-gon
    faces[0].reserve(slices);
    for (uint i = 0; i < slices; i++) faces[0].emplace_back(i);
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
    faces[slices + 1].reserve(slices);
    for (int i = slices - 1; i >= 0; --i) faces[slices + 1].emplace_back(i + slices);

    return {std::move(vertices), std::move(faces)};
}

inline Mesh Cone(float radius = 0.5, float height = 1, uint slices = 32) {
    std::vector<vec3> vertices(slices + 1); // Base + top
    std::vector<std::vector<uint>> indices(slices + 1); // Side triangles + base n-gon

    for (uint i = 0; i < slices; ++i) { // Base
        const float angle = 2 * M_PI * float(i) / float(slices);
        vertices[i] = {radius * cos(angle), -height / 2, radius * sin(angle)};
    }
    vertices[slices] = {0, height / 2, 0}; // Top

    // Side triangles
    for (uint i = 0; i < slices; ++i) indices[i] = {slices, (i + 1) % slices, i};
    // Base n-gon
    indices[slices].reserve(slices);
    for (uint i = 0; i < slices; ++i) indices[slices].emplace_back(i);

    return {std::move(vertices), std::move(indices)};
}

inline std::string to_string(Primitive primitive) {
    switch (primitive) {
        case Primitive::Rect: return "Rect";
        case Primitive::Circle: return "Circle";
        case Primitive::Cube: return "Cube";
        case Primitive::IcoSphere: return "IcoSphere";
        case Primitive::UVSphere: return "UVSphere";
        case Primitive::Torus: return "Torus";
        case Primitive::Cylinder: return "Cylinder";
        case Primitive::Cone: return "Cone";
    }
}

inline Mesh CreateDefaultPrimitive(Primitive primitive) {
    switch (primitive) {
        case Primitive::Rect: return Rect();
        case Primitive::Cube: return Cuboid();
        case Primitive::IcoSphere: return IcoSphere();
        case Primitive::Circle: return Circle();
        case Primitive::UVSphere: return UVSphere();
        case Primitive::Torus: return Torus();
        case Primitive::Cylinder: return Cylinder();
        case Primitive::Cone: return Cone();
    }
}

inline const std::vector<Primitive> AllPrimitives{
    Primitive::Rect,
    Primitive::Circle,
    Primitive::Cube,
    Primitive::IcoSphere,
    Primitive::UVSphere,
    Primitive::Torus,
    Primitive::Cylinder,
    Primitive::Cone,
};
