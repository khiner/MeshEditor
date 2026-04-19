#include "PhysicsDebugDraw.h"

#include <cmath>
#include <numbers>

namespace physics_debug {
namespace {
constexpr uint32_t CircleSegments{24};
constexpr float Pi{std::numbers::pi_v<float>};

void AddCircle(WireframeMesh &mesh, vec3 center, vec3 axis_u, vec3 axis_v, float radius, uint32_t segments = CircleSegments) {
    const uint32_t base = uint32_t(mesh.Positions.size());
    for (uint32_t i = 0; i < segments; ++i) {
        const auto angle = 2 * Pi * float(i) / float(segments);
        mesh.Positions.emplace_back(center + axis_u * (std::cos(angle) * radius) + axis_v * (std::sin(angle) * radius));
    }
    for (uint32_t i = 0; i < segments; ++i) {
        mesh.EdgeIndices.emplace_back(base + i);
        mesh.EdgeIndices.emplace_back(base + (i + 1) % segments);
    }
}

void AddArc(WireframeMesh &mesh, vec3 center, vec3 axis_u, vec3 axis_v, float radius, float start, float end, uint32_t segments = CircleSegments / 2) {
    const uint32_t base = uint32_t(mesh.Positions.size());
    for (uint32_t i = 0; i <= segments; ++i) {
        const auto angle = start + (end - start) * float(i) / float(segments);
        mesh.Positions.emplace_back(center + axis_u * (std::cos(angle) * radius) + axis_v * (std::sin(angle) * radius));
    }
    for (uint32_t i = 0; i < segments; ++i) {
        mesh.EdgeIndices.emplace_back(base + i);
        mesh.EdgeIndices.emplace_back(base + i + 1);
    }
}
} // namespace

WireframeMesh UnitBox() {
    WireframeMesh m;
    constexpr float h{0.5f};
    m.Positions = {
        {-h, -h, -h},
        {h, -h, -h},
        {h, h, -h},
        {-h, h, -h},
        {-h, -h, h},
        {h, -h, h},
        {h, h, h},
        {-h, h, h},
    };
    // clang-format off
    m.EdgeIndices = {
        0, 1, 1,
        2, 2, 3,
        3, 0, 4,
        5, 5, 6,
        6, 7, 7,
        4, 0, 4,
        1, 5, 2,
        6, 3, 7,
    };
    // clang-format on
    return m;
}

WireframeMesh UnitSphere() {
    WireframeMesh m;
    vec3 x{1, 0, 0}, y{0, 1, 0}, z{0, 0, 1};
    AddCircle(m, {}, x, y, 0.5f);
    AddCircle(m, {}, x, z, 0.5f);
    AddCircle(m, {}, y, z, 0.5f);
    return m;
}

WireframeMesh UnitCapsuleCap() {
    // Hemisphere wireframe at origin, opening in +Y direction, radius 0.5.
    // Base circle in XZ plane + two quarter-arcs (XY and ZY).
    WireframeMesh m;
    vec3 x{1, 0, 0}, y{0, 1, 0}, z{0, 0, 1};
    AddCircle(m, {}, x, z, 0.5f);
    AddArc(m, {}, x, y, 0.5f, 0, Pi / 2);
    AddArc(m, {}, {-1, 0, 0}, y, 0.5f, 0, Pi / 2);
    AddArc(m, {}, z, y, 0.5f, 0, Pi / 2);
    AddArc(m, {}, {0, 0, -1}, y, 0.5f, 0, Pi / 2);
    return m;
}

WireframeMesh UnitCircle() {
    WireframeMesh m;
    AddCircle(m, {}, {1, 0, 0}, {0, 0, 1}, 0.5f);
    return m;
}

WireframeMesh UnitLine() {
    return {{{0, 0.5f, 0}, {0, -0.5f, 0}}, {0, 1}};
}
} // namespace physics_debug
