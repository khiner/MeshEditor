#pragma once

#include "gpu/MeshAttributeBit.h"

#include <numeric/vec2.h>
#include <numeric/vec3.h>
#include <numeric/vec4.h>
#include <optional>
#include <vector>

// Per-vertex attributes.
// Absent channels use GPU defaults.
using numeric::vec2, numeric::vec3, numeric::vec4;

struct MeshVertexAttributes {
    std::optional<std::vector<vec3>> Normals{};
    std::optional<std::vector<vec4>> Tangents{}, Colors0{};
    std::optional<std::vector<vec2>> TexCoords0{}, TexCoords1{}, TexCoords2{}, TexCoords3{};
    uint8_t Colors0ComponentCount{}; // 3 or 4 (0 = COLOR_0 absent); CPU storage is always vec4
};
