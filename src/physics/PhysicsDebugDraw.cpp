#include "PhysicsDebugDraw.h"

#include <cmath>
#include <numbers>

namespace physics_debug {

static constexpr int CircleSegments = 24;
static constexpr float Pi = std::numbers::pi_v<float>;

static void AddCircle(WireframeMesh &mesh, vec3 center, vec3 axis_u, vec3 axis_v, float radius, int segments = CircleSegments) {
    const uint32_t base = uint32_t(mesh.Positions.size());
    for (int i = 0; i < segments; ++i) {
        float angle = 2.0f * Pi * float(i) / float(segments);
        mesh.Positions.push_back(center + axis_u * (std::cos(angle) * radius) + axis_v * (std::sin(angle) * radius));
    }
    for (int i = 0; i < segments; ++i) {
        mesh.EdgeIndices.push_back(base + i);
        mesh.EdgeIndices.push_back(base + (i + 1) % segments);
    }
}

static void AddArc(WireframeMesh &mesh, vec3 center, vec3 axis_u, vec3 axis_v, float radius, float start, float end, int segments = CircleSegments / 2) {
    const uint32_t base = uint32_t(mesh.Positions.size());
    for (int i = 0; i <= segments; ++i) {
        float angle = start + (end - start) * float(i) / float(segments);
        mesh.Positions.push_back(center + axis_u * (std::cos(angle) * radius) + axis_v * (std::sin(angle) * radius));
    }
    for (int i = 0; i < segments; ++i) {
        mesh.EdgeIndices.push_back(base + i);
        mesh.EdgeIndices.push_back(base + i + 1);
    }
}

WireframeMesh UnitBox() {
    WireframeMesh m;
    constexpr float h = 0.5f;
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
    m.EdgeIndices = {
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        0,
        4,
        5,
        5,
        6,
        6,
        7,
        7,
        4,
        0,
        4,
        1,
        5,
        2,
        6,
        3,
        7,
    };
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

WireframeMesh UnitCapsuleBody() {
    // 4 vertical lines along Y-axis from y=-0.5 to y=+0.5.
    // Instance scale.y encodes cylinder height; caps are separate instances.
    WireframeMesh m;
    for (int i = 0; i < 4; ++i) {
        float angle = Pi * 0.5f * float(i);
        float cx = std::cos(angle) * 0.5f, cz = std::sin(angle) * 0.5f;
        uint32_t base = uint32_t(m.Positions.size());
        m.Positions.push_back({cx, 0.5f, cz});
        m.Positions.push_back({cx, -0.5f, cz});
        m.EdgeIndices.push_back(base);
        m.EdgeIndices.push_back(base + 1);
    }
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

WireframeMesh UnitCylinder() {
    // Two circles (top/bottom) + 4 vertical lines. Radius 0.5, height 1 along Y.
    WireframeMesh m;
    vec3 x{1, 0, 0}, z{0, 0, 1};
    AddCircle(m, {0, 0.5f, 0}, x, z, 0.5f);
    AddCircle(m, {0, -0.5f, 0}, x, z, 0.5f);
    for (int i = 0; i < 4; ++i) {
        float angle = Pi * 0.5f * float(i);
        float cx = std::cos(angle) * 0.5f, cz = std::sin(angle) * 0.5f;
        uint32_t base = uint32_t(m.Positions.size());
        m.Positions.push_back({cx, 0.5f, cz});
        m.Positions.push_back({cx, -0.5f, cz});
        m.EdgeIndices.push_back(base);
        m.EdgeIndices.push_back(base + 1);
    }
    return m;
}

} // namespace physics_debug
