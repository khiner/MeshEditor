#pragma once

#include "numeric/vec2.h"
#include "numeric/vec3.h"

#include <array>
#include <string>
#include <variant>

namespace primitive {
struct Rect {
    vec2 HalfExtents{1, 1};
};
struct Circle {
    float Radius{1};
    uint32_t Segments{32};
};
struct Cuboid {
    vec3 HalfExtents{1, 1, 1};
};
struct IcoSphere {
    float Radius{1};
    uint32_t Subdivisions{3};
};
struct UVSphere {
    float Radius{1};
    uint32_t Slices{32}, Stacks{16};
};
struct Torus {
    float MajorRadius{1}, MinorRadius{0.5};
    uint32_t MajorSegments{32}, MinorSegments{16};
};
struct Cylinder {
    float Radius{1}, Height{2};
    uint32_t Slices{32};
};
struct Cone {
    float Radius{1}, Height{2};
    uint32_t Slices{32};
};
} // namespace primitive

using PrimitiveShape = std::variant<
    primitive::Rect,
    primitive::Circle,
    primitive::Cuboid,
    primitive::IcoSphere,
    primitive::UVSphere,
    primitive::Torus,
    primitive::Cylinder,
    primitive::Cone>;

inline std::string ToString(const PrimitiveShape &shape) {
    static constexpr const char *Names[]{"Rect", "Circle", "Cube", "IcoSphere", "UVSphere", "Torus", "Cylinder", "Cone"};
    return Names[shape.index()];
}

// All default primitive shapes, for iteration in UI menus.
inline const std::array AllPrimitiveShapes{
    PrimitiveShape{primitive::Rect{}},
    PrimitiveShape{primitive::Circle{}},
    PrimitiveShape{primitive::Cuboid{}},
    PrimitiveShape{primitive::IcoSphere{}},
    PrimitiveShape{primitive::UVSphere{}},
    PrimitiveShape{primitive::Torus{}},
    PrimitiveShape{primitive::Cylinder{}},
    PrimitiveShape{primitive::Cone{}},
};
