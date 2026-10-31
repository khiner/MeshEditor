#include "numeric/VectorMath.h"

#include "audio/ContactModel.h"
#include "editor/AudioExcitation.h"
#include "render/Instance.h"
#include "state/Registry.h"

#include <numbers>

// Unit surface normal at a mesh vertex.
vec3 VertexNormal(const Mesh &mesh, uint32_t vertex) { return Normalize(mesh.GetNormal(mesh.VertexAt(vertex))); }

// Tilts a unit normal using a joystick position in the unit disk.
vec3 TiltAlongNormal(vec3 n, vec2 joy) {
    const float r = Length(joy);
    if (r < 1e-6f) return n;
    // Orthonormal tangent basis from the normal (Duff et al. 2017).
    const float s = n.z >= 0 ? 1.f : -1.f;
    const float a = -1.f / (s + n.z);
    const float b = n.x * n.y * a;
    const vec3 t{1.f + s * n.x * n.x * a, s * b, -s * n.x};
    const vec3 bt{b, s + n.y * n.y * a, -n.y};
    const float theta = std::min(r, 1.f) * std::numbers::pi_v<float> / 2;
    return std::cos(theta) * n + std::sin(theta) * (joy.x * t + joy.y * bt) / r;
}

vec3 ExciteDirection(const state::Registry &r, state::Entity e, uint32_t vertex) {
    const auto n = VertexNormal(GetMesh(r, r.get<const Instance>(e).Entity), vertex);
    return TiltAlongNormal(n, r.get<const Striker>(state::Document).Tilt);
}
