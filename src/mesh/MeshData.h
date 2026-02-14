#pragma once

#include "numeric/vec3.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

struct MeshData {
    std::vector<vec3> Positions; // Vertex positions (required for all topologies)
    std::vector<std::vector<uint32_t>> Faces{}; // Per-face vertex index loops (triangles/polygons)
    std::vector<std::array<uint32_t, 2>> Edges{}; // Line segment vertex index pairs
    std::optional<std::vector<vec3>> Normals{}; // Per-vertex authored normals (triangle meshes only, computed from faces if absent)
};
