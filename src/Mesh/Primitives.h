#pragma once

#include "numeric/vec2.h"

#include "mesh/Mesh.h"

inline Mesh Rect(vec2 half_extents) {
    const auto x = half_extents.x, y = half_extents.y;
    return {
        {{-x, -y, 0}, {x, -y, 0}, {x, y, 0}, {-x, y, 0}},
        {{0, 1, 2, 3}}
    };
}

inline Mesh Circle(float radius = 1, uint segments = 32) {
    std::vector<vec3> vertices;
    vertices.reserve(segments + 1);
    std::vector<std::vector<uint>> indices;
    indices.reserve(segments);
    for (uint i = 0; i < segments; ++i) {
        const float theta = 2 * M_PI * i / segments;
        vertices.emplace_back(radius * vec3{cos(theta), sin(theta), 0});
        indices.push_back({i, (i + 1) % segments, segments});
    }
    vertices.emplace_back(0, 0, 0);
    return {std::move(vertices), std::move(indices)};
}

inline Mesh Cuboid(vec3 half_extents) {
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

inline Mesh IcoSphere(float radius = 1, int recursion_level = 1) {
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

    for (int r = 0; r < recursion_level; ++r) {
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

enum class Primitive {
    Rect,
    Circle,
    Cube,
    IcoSphere,
};

inline std::string to_string(Primitive primitive) {
    switch (primitive) {
        case Primitive::Rect: return "Rect";
        case Primitive::Circle: return "Circle";
        case Primitive::Cube: return "Cube";
        case Primitive::IcoSphere: return "IcoSphere";
    }
}

inline Mesh CreateDefaultPrimitive(Primitive primitive) {
    switch (primitive) {
        case Primitive::Rect: return Rect({0.5, 0.5});
        case Primitive::Cube: return Cuboid({0.5, 0.5, 0.5});
        case Primitive::IcoSphere: return IcoSphere(0.5, 3);
        case Primitive::Circle: return Circle(0.5, 32);
    }
}

inline const std::vector<Primitive> AllPrimitives{Primitive::Rect, Primitive::Circle, Primitive::Cube, Primitive::IcoSphere};
