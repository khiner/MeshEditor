#pragma once

#include "mesh/MeshData.h"

#include <algorithm>
#include <vector>

inline MeshData Arrow(float base_radius = 0.15, float tip_radius = 0.3, float base_length = 0.75, float tip_length = 0.25, uint slices = 32) {
    std::vector<vec3> vertices;
    vertices.reserve(slices * 3 + 1);
    for (uint i = 0; i < slices; ++i) {
        const float __angle = 2.0f * float(i) / float(slices);
        const float x = __cospif(__angle), z = __sinpif(__angle);
        vertices.emplace_back(x * tip_radius, tip_length, z * tip_radius);
        vertices.emplace_back(x * base_radius, tip_length, z * base_radius);
        vertices.emplace_back(x * base_radius, tip_length + base_length, z * base_radius);
    }
    vertices.emplace_back(0, 0, 0);

    std::vector<uint> base_bottom_face, base_top_face, tip_face;
    base_bottom_face.reserve(slices);
    base_top_face.reserve(slices);
    tip_face.reserve(slices);
    for (uint i = 0; i < slices; ++i) {
        tip_face.emplace_back(i * 3);
        base_bottom_face.emplace_back(i * 3 + 1);
        base_top_face.emplace_back(i * 3 + 2);
    }

    std::vector<std::vector<uint>> faces;
    faces.reserve(slices * 2 + 3); // slices quads/triangles + 3 n-gon caps
    // Quads for the sides of the cylinder.
    for (uint i = 0; i < slices; ++i) {
        faces.push_back({
            base_top_face[(i + 1) % slices],
            base_bottom_face[(i + 1) % slices],
            base_bottom_face[i],
            base_top_face[i],
        });
    }

    // N-gons for the caps
    std::reverse(base_top_face.begin(), base_top_face.end()); // For CCW order
    std::reverse(tip_face.begin(), tip_face.end()); // For CCW order
    faces.emplace_back(base_top_face);
    faces.emplace_back(base_bottom_face);
    faces.emplace_back(tip_face);

    // Triangles for the tip cone.
    const uint tip_index = vertices.size() - 1;
    for (uint i = 0; i < slices; ++i) faces.push_back({tip_face[i], tip_index, tip_face[(i + 1) % slices]});

    return {std::move(vertices), std::move(faces)};
}
