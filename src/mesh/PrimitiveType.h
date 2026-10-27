#pragma once

#include "Field.h"
#include "mesh/MeshSchema.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"

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
struct Cube {
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

// The parameters of one primitive generator.
using PrimitiveParameters = std::variant<primitive::Plane, primitive::Circle, primitive::Cube, primitive::IcoSphere, primitive::UVSphere, primitive::Torus, primitive::Cylinder, primitive::Cone>;

namespace primitive {
constexpr FieldSpec SizeSpec(const char *help) { return {.Min = MinSize, .Max = MaxSize, .Speed = 0.01f, .Unit = FieldUnit::Meters, .Widget = FieldWidget::Drag, .Help = help}; }
constexpr FieldSpec HalfExtentsSpec(const char *help) { return {.Min = MinSize / 2, .Max = MaxSize / 2, .Speed = 0.01f, .Unit = FieldUnit::Meters, .Widget = FieldWidget::Drag, .Label = "Half extents", .Help = help}; }
} // namespace primitive

template<> inline constexpr field::SpecTable<primitive::Plane> Specs<primitive::Plane>{
    {&primitive::Plane::HalfExtents, primitive::HalfExtentsSpec("Distance from the center to each edge along X and Z.")},
};
template<> inline constexpr field::SpecTable<primitive::Circle> Specs<primitive::Circle>{
    {&primitive::Circle::Radius, primitive::SizeSpec("Distance from the center to the rim.")},
    {&primitive::Circle::Segments, {.Min = 3, .Max = 128, .Help = "Number of vertices around the rim"}},
};
template<> inline constexpr field::SpecTable<primitive::Cube> Specs<primitive::Cube>{
    {&primitive::Cube::HalfExtents, primitive::HalfExtentsSpec("Distance from the center to each face along X, Y, and Z.")},
};
template<> inline constexpr field::SpecTable<primitive::IcoSphere> Specs<primitive::IcoSphere>{
    {&primitive::IcoSphere::Radius, primitive::SizeSpec("Distance from the center to each vertex.")},
    {&primitive::IcoSphere::Subdivisions, {.Min = 1, .Max = 6, .Help = "Number of times each icosahedron triangle splits into four. Each level quadruples the face count."}},
};
template<> inline constexpr field::SpecTable<primitive::UVSphere> Specs<primitive::UVSphere>{
    {&primitive::UVSphere::Radius, primitive::SizeSpec("Distance from the center to the surface.")},
    {&primitive::UVSphere::Slices, {.Min = 3, .Max = 128, .Help = "Number of segments around the vertical axis"}},
    {&primitive::UVSphere::Stacks, {.Min = 2, .Max = 64, .Help = "Number of rings from pole to pole"}},
};
template<> inline constexpr field::SpecTable<primitive::Torus> Specs<primitive::Torus>{
    {&primitive::Torus::MajorRadius, primitive::SizeSpec("Distance from the center to the middle of the tube.")},
    {&primitive::Torus::MinorRadius, primitive::SizeSpec("Radius of the tube.")},
    {&primitive::Torus::MajorSegments, {.Min = 3, .Max = 256, .Help = "Number of segments around the ring"}},
    {&primitive::Torus::MinorSegments, {.Min = 3, .Max = 256, .Help = "Number of segments around the tube"}},
};
template<> inline constexpr field::SpecTable<primitive::Cylinder> Specs<primitive::Cylinder>{
    {&primitive::Cylinder::Radius, primitive::SizeSpec("Radius of the caps.")},
    {&primitive::Cylinder::Height, primitive::SizeSpec("Distance between the caps.")},
    {&primitive::Cylinder::Slices, {.Min = 3, .Max = 128, .Help = "Number of segments around the side"}},
};
template<> inline constexpr field::SpecTable<primitive::Cone> Specs<primitive::Cone>{
    {&primitive::Cone::Radius, primitive::SizeSpec("Radius of the base.")},
    {&primitive::Cone::Height, primitive::SizeSpec("Distance from the base to the tip.")},
    {&primitive::Cone::Slices, {.Min = 3, .Max = 128, .Help = "Number of segments around the side"}},
};
