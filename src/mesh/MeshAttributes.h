#pragma once

#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"
#include <optional>
#include <vector>

// Per-vertex GPU vertex attributes (absent channels use GPU defaults)
struct MeshVertexAttributes {
    std::optional<std::vector<vec3>> Normals{};
    std::optional<std::vector<vec4>> Tangents{}, Colors0{};
    std::optional<std::vector<vec2>> TexCoords0{}, TexCoords1{}, TexCoords2{}, TexCoords3{};
};

// Per-face -> per-primitive -> material GPU index mapping
struct MeshPrimitives {
    std::vector<uint32_t> FacePrimitiveIndices{}; // per-face source primitive index
    std::vector<uint32_t> MaterialIndices{}; // primitive index -> scene material index
};
