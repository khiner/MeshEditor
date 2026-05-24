#pragma once

#include <numeric/vec2.h>
#include <numeric/vec3.h>
#include <numeric/vec4.h>

#include <cstdint>
#include <optional>
#include <vector>

// Per-vertex attributes. Absent channels use GPU defaults.
struct MeshVertexAttributes {
    std::optional<std::vector<vec3>> Normals{};
    std::optional<std::vector<vec4>> Tangents{}, Colors0{};
    std::optional<std::vector<vec2>> TexCoords0{}, TexCoords1{}, TexCoords2{}, TexCoords3{};
    uint8_t Colors0ComponentCount{}; // 3 or 4 (0 = COLOR_0 absent); CPU storage is always vec4
};

enum MeshAttributeBit : uint32_t {
    MeshAttributeBit_Normal = 1u << 0,
    MeshAttributeBit_Tangent = 1u << 1,
    MeshAttributeBit_Color0 = 1u << 2,
    MeshAttributeBit_TexCoord0 = 1u << 3,
    MeshAttributeBit_TexCoord1 = 1u << 4,
    MeshAttributeBit_TexCoord2 = 1u << 5,
    MeshAttributeBit_TexCoord3 = 1u << 6,
};
