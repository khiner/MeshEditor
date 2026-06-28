#pragma once

#include "numeric/vec3.h"

struct MeshData {
    std::vector<vec3> Positions;
    std::vector<std::vector<uint32_t>> Faces{}; // Per-face vertex index loops (triangles/polygons)
    std::vector<std::array<uint32_t, 2>> Edges{}; // Line segment vertex index pairs

    // Flatten edge pairs into a contiguous index array for GPU line rendering.
    std::vector<uint32_t> CreateEdgeIndices() const {
        std::vector<uint32_t> indices;
        indices.reserve(Edges.size() * 2);
        for (const auto &[a, b] : Edges) {
            indices.emplace_back(a);
            indices.emplace_back(b);
        }
        return indices;
    }
};
