#pragma once

#include "Field.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"

#include <string>
#include <variant>

namespace primitive {
using numeric::vec2, numeric::vec3;

// Bounds for the editable size fields of every primitive (radii, extents, height).
constexpr float MinSize = 0.01f, MaxSize = 100.f;

// Finite plane in the XZ plane (normal +Y), centered at origin.
struct Plane {
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

template<> inline constexpr field::SpecTable<primitive::Plane> Specs<primitive::Plane>{{&primitive::Plane::HalfExtents, {.Min = primitive::MinSize / 2, .Max = primitive::MaxSize / 2}}};
template<> inline constexpr field::SpecTable<primitive::Circle> Specs<primitive::Circle>{
    {&primitive::Circle::Radius, {.Min = primitive::MinSize, .Max = primitive::MaxSize}},
    {&primitive::Circle::Segments, {.Min = 3, .Max = 128}},
};
template<> inline constexpr field::SpecTable<primitive::Cuboid> Specs<primitive::Cuboid>{{&primitive::Cuboid::HalfExtents, {.Min = primitive::MinSize / 2, .Max = primitive::MaxSize / 2}}};
template<> inline constexpr field::SpecTable<primitive::IcoSphere> Specs<primitive::IcoSphere>{
    {&primitive::IcoSphere::Radius, {.Min = primitive::MinSize, .Max = primitive::MaxSize}},
    {&primitive::IcoSphere::Subdivisions, {.Min = 1, .Max = 6}},
};
template<> inline constexpr field::SpecTable<primitive::UVSphere> Specs<primitive::UVSphere>{
    {&primitive::UVSphere::Radius, {.Min = primitive::MinSize, .Max = primitive::MaxSize}},
    {&primitive::UVSphere::Slices, {.Min = 3, .Max = 128}},
    {&primitive::UVSphere::Stacks, {.Min = 2, .Max = 64}},
};
template<> inline constexpr field::SpecTable<primitive::Torus> Specs<primitive::Torus>{
    {&primitive::Torus::MajorRadius, {.Min = primitive::MinSize, .Max = primitive::MaxSize}},
    {&primitive::Torus::MinorRadius, {.Min = primitive::MinSize, .Max = primitive::MaxSize}},
    {&primitive::Torus::MajorSegments, {.Min = 3, .Max = 256}},
    {&primitive::Torus::MinorSegments, {.Min = 3, .Max = 256}},
};
template<> inline constexpr field::SpecTable<primitive::Cylinder> Specs<primitive::Cylinder>{
    {&primitive::Cylinder::Radius, {.Min = primitive::MinSize, .Max = primitive::MaxSize}},
    {&primitive::Cylinder::Height, {.Min = primitive::MinSize, .Max = primitive::MaxSize}},
    {&primitive::Cylinder::Slices, {.Min = 3, .Max = 128}},
};
template<> inline constexpr field::SpecTable<primitive::Cone> Specs<primitive::Cone>{
    {&primitive::Cone::Radius, {.Min = primitive::MinSize, .Max = primitive::MaxSize}},
    {&primitive::Cone::Height, {.Min = primitive::MinSize, .Max = primitive::MaxSize}},
    {&primitive::Cone::Slices, {.Min = 3, .Max = 128}},
};

using PrimitiveShape = std::variant<
    primitive::Plane,
    primitive::Circle,
    primitive::Cuboid,
    primitive::IcoSphere,
    primitive::UVSphere,
    primitive::Torus,
    primitive::Cylinder,
    primitive::Cone>;

inline std::string ToString(const PrimitiveShape &shape) {
    static constexpr const char *Names[]{"Plane", "Circle", "Cube", "IcoSphere", "UVSphere", "Torus", "Cylinder", "Cone"};
    return Names[shape.index()];
}
