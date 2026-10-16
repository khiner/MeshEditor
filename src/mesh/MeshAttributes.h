#pragma once

#include "gpu/MeshAttributeBit.h"
#include "mesh/NamedAttribute.h"

#include <optional>

using numeric::vec2, numeric::vec3, numeric::vec4;

// A source attribute with its type's component words per vertex, which becomes a named attribute of the created mesh.
struct CustomAttribute {
    AttributeName Name;
    AttributeType Type;
    std::vector<uint32_t> Words;
};

// Per-vertex attributes.
// Absent channels use GPU defaults.
struct MeshVertexAttributes {
    std::optional<std::vector<vec3>> Normals{};
    std::optional<std::vector<vec4>> Tangents{}, Colors0{};
    std::optional<std::vector<vec2>> TexCoords0{}, TexCoords1{}, TexCoords2{}, TexCoords3{};
    uint8_t Colors0ComponentCount{}; // 3 or 4 (0 = COLOR_0 absent); CPU storage is always vec4
    std::vector<CustomAttribute> Custom{}; // Unique names
};
