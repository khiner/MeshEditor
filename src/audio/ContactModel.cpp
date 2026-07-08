#include "ContactModel.h"

#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

double StrikerMass(const Striker &s) {
    const double r = s.TipRadius, l = s.Length;
    return s.Material.Properties.Density * std::numbers::pi * (r * r * l + 4.0 / 3.0 * r * r * r); // cylinder + spherical caps
}

glm::mat3 InverseInertiaTensor(const MassProperties &mp) {
    const glm::mat3 r = glm::mat3_cast(mp.InertiaOrientation);
    glm::vec3 inv{0};
    for (int i = 0; i < 3; ++i) inv[i] = mp.InertiaDiagonal[i] > 0 ? 1.f / mp.InertiaDiagonal[i] : 0.f;
    const glm::mat3 inv_diag{glm::vec3{inv.x, 0, 0}, glm::vec3{0, inv.y, 0}, glm::vec3{0, 0, inv.z}};
    return r * inv_diag * glm::transpose(r);
}

double EstimateContactTime(const ContactDynamics &d, uint32_t i, vec3 impact_direction, double contact_speed, const AcousticMaterialProperties &m, const Striker &striker, double scale_ratio) {
    if (i >= d.Curvature.size() || d.Mass <= 0) return MinContactTime;

    const auto &sm = striker.Material.Properties;

    // Reduced mass at the contact: the object's translation and the rotational leverage of an off-center impulse,
    // combined with the striker. A light striker dominates, so the contact stays short even against a heavy object.
    const vec3 n = glm::normalize(impact_direction);
    const glm::vec3 arm_cross_n = glm::cross(d.ContactArm[i], n);
    const double inv_effective_mass = 1.0 / d.Mass + glm::dot(arm_cross_n, d.InverseInertia * arm_cross_n) + 1.0 / StrikerMass(striker);
    const double effective_mass = 1.0 / inv_effective_mass;

    // Effective compliance and curvature: each combines the object's and the striker's.
    // Clamp curvature positive, since the object's can be flat or concave.
    const double inv_effective_modulus =
        (1 - m.PoissonRatio * m.PoissonRatio) / m.YoungModulus + (1 - sm.PoissonRatio * sm.PoissonRatio) / sm.YoungModulus;
    const double curvature = std::max(double(d.Curvature[i]) + 1.0 / striker.TipRadius, 1e-6);
    const double speed = std::max(std::abs(contact_speed), 1e-6);

    const double tau_baked = 2.87 * std::pow(std::pow(effective_mass * inv_effective_modulus, 2) * (curvature / speed), 0.2);
    return std::clamp(tau_baked * scale_ratio, MinContactTime, MaxContactTime);
}
