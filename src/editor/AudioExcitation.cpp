
#include "editor/AudioExcitation.h"
#include "audio/ContactModel.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/Neighborhood.h"
#include "render/Instance.h"
#include "state/Registry.h"

#include <numbers>

namespace {
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
} // namespace

vec3 ExciteDirection(const state::Registry &r, state::Entity e, uint32_t vertex) {
    const auto mesh = GetMesh(r, r.get<const Instance>(e).Entity);
    const std::array handle{*mesh.VertexAt(vertex)};
    const auto n = Normalize(VertexNormals(r.Resources.get<const MeshStore>(), GetMeshPipelines(r), mesh.GetStoreId(), handle).front());
    return TiltAlongNormal(n, r.get<const Striker>(state::Document).Tilt);
}
