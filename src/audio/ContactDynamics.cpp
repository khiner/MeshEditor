#include "AudioSystem.h"

#include "ContactModel.h"
#include "ModalModes.h"
#include "mesh/Mesh.h"
#include "render/Instance.h"

#include <entt/entity/registry.hpp>
#include <glm/geometric.hpp>

#include <algorithm>

namespace {
// Discrete mean curvature (1/length) at a mesh vertex, averaged over the one-ring normal curvatures.
// Reduces to 1/R on a sphere of radius R, and is zero on a flat or boundary vertex.
float MeanCurvature(const Mesh &mesh, uint32_t vertex) {
    const Mesh::VH vh{vertex};
    const vec3 xi = mesh.GetPosition(vh);
    const vec3 ni = glm::normalize(mesh.GetNormal(vh));
    double sum = 0;
    int count = 0;
    for (const auto he : mesh.voh_range(vh)) {
        const vec3 d = mesh.GetPosition(mesh.GetToVertex(he)) - xi;
        const double d2 = glm::dot(d, d);
        if (d2 < 1e-20) continue;
        sum += -2.0 * double(glm::dot(d, ni)) / d2;
        ++count;
    }
    return count ? float(sum / count) : 0.f;
}
} // namespace

void UpdateContactDynamics(entt::registry &r, entt::entity e) {
    const auto *mp = r.try_get<const MassProperties>(e);
    const auto *modes = r.try_get<const ModalModes>(e);
    const auto *inst = r.try_get<const Instance>(e);
    const auto mesh = inst ? TryGetMesh(r, inst->Entity) : std::nullopt;
    if (!mp || !modes || !mesh || modes->Vertices.empty() || modes->Positions.size() != modes->Vertices.size()) {
        r.remove<ContactDynamics>(e);
        return;
    }
    const auto size = [](vec3 v) { const auto a = glm::abs(v); return (a.x + a.y + a.z) / 3.f; };
    const float baked_scale = std::max(size(modes->BakedScale), 1e-6f);
    ContactDynamics cd;
    cd.Mass = mp->Mass;
    cd.InverseInertia = InverseInertiaTensor(*mp);
    cd.ContactArm.reserve(modes->Vertices.size());
    cd.Curvature.reserve(modes->Vertices.size());
    for (size_t i = 0; i < modes->Vertices.size(); ++i) {
        cd.ContactArm.push_back((modes->Positions[i] - mp->CenterOfMass) * baked_scale);
        cd.Curvature.push_back(MeanCurvature(*mesh, modes->Vertices[i]) / baked_scale);
    }
    r.emplace_or_replace<ContactDynamics>(e, std::move(cd));
}
