#include "ContactModel.h"

#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

double StrikerMass(const Striker &s) {
    const double r = s.TipRadius, l = s.Length;
    return s.Material.Properties.Density * std::numbers::pi * (r * r * l + 4.0 / 3.0 * r * r * r); // cylinder + spherical caps
}

Impactor StrikerImpactor(const Striker &s) { return {.Material = s.Material.Properties, .Curvature = 1.0 / s.TipRadius, .InvMass = 1.0 / StrikerMass(s)}; }

glm::mat3 InverseInertiaTensor(const MassProperties &mp) {
    const glm::mat3 r = glm::mat3_cast(mp.InertiaOrientation);
    glm::vec3 inv{0};
    for (int i = 0; i < 3; ++i) inv[i] = mp.InertiaDiagonal[i] > 0 ? 1.f / mp.InertiaDiagonal[i] : 0.f;
    const glm::mat3 inv_diag{glm::vec3{inv.x, 0, 0}, glm::vec3{0, inv.y, 0}, glm::vec3{0, 0, inv.z}};
    return r * inv_diag * glm::transpose(r);
}

double ReducedContactMass(const ContactDynamics &d, uint32_t i, vec3 impact_direction, const Impactor &impactor) {
    if (i >= d.ContactArm.size() || d.Mass <= 0) return 0;

    // The object's translation and the rotational leverage of an off-center impulse, combined with the impactor.
    // A light impactor dominates, so the reduced mass stays small even against a heavy object.
    const vec3 n = glm::normalize(impact_direction);
    const glm::vec3 arm_cross_n = glm::cross(d.ContactArm[i], n);
    const double inv_effective_mass = 1.0 / d.Mass + glm::dot(arm_cross_n, d.InverseInertia * arm_cross_n) + impactor.InvMass;
    return 1.0 / inv_effective_mass;
}

double InvEffectiveModulus(const AcousticMaterialProperties &a, const AcousticMaterialProperties &b) {
    return (1 - a.PoissonRatio * a.PoissonRatio) / a.YoungModulus + (1 - b.PoissonRatio * b.PoissonRatio) / b.YoungModulus;
}

double CombinedCurvature(double curvature_a, double curvature_b) { return std::max(curvature_a + curvature_b, 1e-6); }

double ContactStiffness(double inv_effective_modulus, double combined_curvature) {
    return 4.0 / 3.0 / inv_effective_modulus / std::sqrt(combined_curvature);
}

double ContactPatchRadius(double normal_force, double inv_effective_modulus, double combined_curvature) {
    return std::cbrt(0.75 * std::max(normal_force, 0.0) * inv_effective_modulus / combined_curvature);
}

double StaticPenetration(double normal_force, double stiffness) {
    return stiffness > 0 ? std::pow(std::max(normal_force, 0.0) / stiffness, 2.0 / 3.0) : 0.0;
}

double EstimateContactTime(const ContactDynamics &d, uint32_t i, vec3 impact_direction, double contact_speed, const AcousticMaterialProperties &m, double object_curvature, const Impactor &impactor, double scale_ratio) {
    if (i >= d.ContactArm.size() || d.Mass <= 0) return MinContactTime;

    const double effective_mass = ReducedContactMass(d, i, impact_direction, impactor);

    const double inv_effective_modulus = InvEffectiveModulus(m, impactor.Material);
    const double curvature = CombinedCurvature(object_curvature, impactor.Curvature);
    const double speed = std::max(std::abs(contact_speed), 1e-6);

    const double tau_baked = 2.87 * std::pow(std::pow(effective_mass * inv_effective_modulus, 2) * (curvature / speed), 0.2);
    return std::clamp(tau_baked * scale_ratio, MinContactTime, MaxContactTime);
}
