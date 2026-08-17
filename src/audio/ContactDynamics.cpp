#include "AudioSystem.h"

#include "ContactModel.h"
#include "ModalEigenSummary.h"
#include "ModalModes.h"
#include "TransformMath.h"
#include "physics/PhysicsTypes.h"

#include <entt/entity/registry.hpp>

#include <algorithm>

double ModalDensityRatio(const entt::registry &r, entt::entity e) {
    const auto *summary = r.try_get<const ModalEigenSummary>(e);
    const auto *mat = r.try_get<const AcousticMaterial>(e);
    return summary && mat && summary->SolvedMaterial.Density > 0 ? mat->Properties.Density / summary->SolvedMaterial.Density : 1.0;
}

void UpdateContactDynamics(entt::registry &r, entt::entity e) {
    const auto *mp = r.try_get<const MassProperties>(e);
    const auto *modes = r.try_get<const ModalModes>(e);
    if (!mp || !modes || modes->Positions.empty()) {
        r.remove<ContactDynamics>(e);
        return;
    }
    const float baked_scale = std::max(MeanScale(modes->BakedScale), 1e-6f);

    // A dynamic KHR_physics_rigid_bodies body (no density scaling) is authoritative for contact response.
    // Otherwise the modal mass properties apply, scaled by rho_ratio.
    MassProperties resolved = *mp;
    double mass_scale = ModalDensityRatio(r, e);
    if (const auto *motion = r.try_get<const PhysicsMotion>(e); motion && IsAuthoritativeDynamicBody(*motion)) {
        resolved.Mass = motion->Mass.value_or(DefaultMass);
        resolved.CenterOfMass = motion->CenterOfMass.value_or(resolved.CenterOfMass);
        resolved.InertiaDiagonal = motion->InertiaDiagonal.value_or(resolved.InertiaDiagonal);
        resolved.InertiaOrientation = motion->InertiaOrientation.value_or(resolved.InertiaOrientation);
        mass_scale = 1.0;
    }

    ContactDynamics cd;
    cd.Mass = resolved.Mass * mass_scale;
    cd.InverseInertia = InverseInertiaTensor(resolved) * float(1 / mass_scale);
    cd.ContactArm.reserve(modes->Positions.size());
    for (const auto &position : modes->Positions) cd.ContactArm.push_back((position - resolved.CenterOfMass) * baked_scale);
    r.emplace_or_replace<ContactDynamics>(e, std::move(cd));
}
