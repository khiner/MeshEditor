#pragma once

namespace primitive {
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
