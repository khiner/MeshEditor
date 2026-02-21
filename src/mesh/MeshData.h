#pragma once

#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

struct MeshData {
    std::vector<vec3> Positions; // Vertex positions (required for all topologies)
    std::vector<std::vector<uint32_t>> Faces{}; // Per-face vertex index loops (triangles/polygons)
    std::vector<std::array<uint32_t, 2>> Edges{}; // Line segment vertex index pairs
    std::optional<std::vector<vec3>> Normals{}; // Per-vertex authored normals (triangle meshes only, computed from faces if absent)
    std::optional<std::vector<vec4>> Tangents{}; // Per-vertex tangent + handedness (w), from glTF TANGENT
    std::optional<std::vector<vec4>> Colors0{}; // Per-vertex color (rgba), from glTF COLOR_0
    std::optional<std::vector<vec2>> TexCoords0{}, TexCoords1{}, TexCoords2{}, TexCoords3{}; // Per-vertex UV sets used by glTF material textures
    std::optional<std::vector<uint32_t>> FacePrimitiveIndices{}; // Per-face source primitive index (triangle meshes)
    std::optional<std::vector<uint32_t>> PrimitiveMaterialIndices{}; // Primitive index -> scene material index (triangle meshes)
};
