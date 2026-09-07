#include "editor/AudioExcitation.h"
#include "render/Instance.h"
#include <cmath>
#include <entt/entity/registry.hpp>

// Strike impact angle relative to the surface.
// Center strikes along surface normal, rim tilts impulse 90 degrees into the tangent plane. UI-only.
vec2 ImpulseAngle{0, 0};

// Unit surface normal at a mesh vertex.
vec3 VertexNormal(const Mesh &mesh, uint32_t vertex) { return numeric::Normalize(mesh.GetNormal(Mesh::VH{vertex})); }

// Tilts a unit normal using a joystick position in the unit disk.
vec3 TiltAlongNormal(vec3 n, vec2 joy) {
    const float r = numeric::Length(joy);
    if (r < 1e-6f) return n;
    // Orthonormal tangent basis from the normal (Duff et al. 2017).
    const float s = n.z >= 0 ? 1.f : -1.f;
    const float a = -1.f / (s + n.z);
    const float b = n.x * n.y * a;
    const vec3 t{1.f + s * n.x * n.x * a, s * b, -s * n.x};
    const vec3 bt{b, s + n.y * n.y * a, -n.y};
    const float theta = std::min(r, 1.f) * 1.57079633f; // radius maps to [0, pi/2]
    return std::cos(theta) * n + std::sin(theta) * (joy.x * t + joy.y * bt) / r;
}

// Strike direction: the excited vertex's normal, tilted by the current impact angle.
vec3 ExciteDirection(const entt::registry &r, entt::entity e, uint32_t vertex) {
    const auto n = VertexNormal(GetMesh(r, r.get<const Instance>(e).Entity), vertex);
    return TiltAlongNormal(n, ImpulseAngle);
}
