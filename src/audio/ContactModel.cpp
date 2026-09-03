#include "ContactModel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

double StrikerMass(const Striker &s) {
    const double r = s.TipRadius, l = s.Length;
    return s.Material.Properties.Density * std::numbers::pi * (r * r * l + 4.0 / 3.0 * r * r * r); // cylinder + spherical caps
}

Impactor StrikerImpactor(const Striker &s) { return {.Material = s.Material.Properties, .Curvature = 1.0 / s.TipRadius, .InvMass = 1.0 / StrikerMass(s)}; }

mat3 InverseInertiaTensor(const MassProperties &mp) {
    const mat3 r = numeric::ToMat3(mp.InertiaOrientation);
    vec3 inv{0};
    for (int i = 0; i < 3; ++i) inv[i] = mp.InertiaDiagonal[i] > 0 ? 1.f / mp.InertiaDiagonal[i] : 0.f;
    const mat3 inv_diag{vec3{inv.x, 0, 0}, vec3{0, inv.y, 0}, vec3{0, 0, inv.z}};
    return r * inv_diag * numeric::Transpose(r);
}

double ReducedContactMass(const ContactDynamics &d, uint32_t i, vec3 impact_direction, const Impactor &impactor) {
    if (i >= d.ContactArm.size() || d.Mass <= 0) return 0;

    // The object's translation and the rotational leverage of an off-center impulse, combined with the impactor.
    // A light impactor dominates, so the reduced mass stays small even against a heavy object.
    const vec3 n = numeric::Normalize(impact_direction);
    const vec3 arm_cross_n = numeric::Cross(d.ContactArm[i], n);
    const double inv_effective_mass = 1.0 / d.Mass + numeric::Dot(arm_cross_n, d.InverseInertia * arm_cross_n) + impactor.InvMass;
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

double SaturationPenetration(double combined_curvature, double nominal_area) {
    return nominal_area > 0 ? nominal_area * combined_curvature / std::numbers::pi : std::numeric_limits<double>::infinity();
}

double PunchStiffness(double inv_effective_modulus, double nominal_area) {
    if (nominal_area <= 0) return std::numeric_limits<double>::infinity();
    return 2 * std::sqrt(nominal_area / std::numbers::pi) / inv_effective_modulus;
}

namespace {
double ContactWork(double penetration, double hertz_stiffness, double sat_penetration, double punch_stiffness) {
    if (penetration <= 0) return 0;
    const auto hertz = [hertz_stiffness](double x) { return 0.4 * hertz_stiffness * x * x * std::sqrt(x); };
    if (penetration <= sat_penetration) return hertz(penetration);
    const double over = penetration - sat_penetration;
    const double sat_force = hertz_stiffness * sat_penetration * std::sqrt(sat_penetration);
    return hertz(sat_penetration) + sat_force * over + 0.5 * punch_stiffness * over * over;
}
} // namespace

double EstimateContactTime(const ContactDynamics &d, uint32_t i, vec3 impact_direction, double contact_speed, const AcousticMaterialProperties &m, double object_curvature, double nominal_area, const Impactor &impactor, double scale_ratio, double combined_roughness) {
    if (i >= d.ContactArm.size() || d.Mass <= 0) return MinContactTime;

    const double effective_mass = ReducedContactMass(d, i, impact_direction, impactor);
    const double inv_effective_modulus = InvEffectiveModulus(m, impactor.Material);
    if (effective_mass <= 0 || inv_effective_modulus <= 0) return MinContactTime;

    const double curvature = CombinedCurvature(object_curvature, impactor.Curvature);
    const double speed = std::max(std::abs(contact_speed), 1e-6);
    const double hertz_stiffness = ContactStiffness(inv_effective_modulus, curvature);
    const double sat_penetration = SaturationPenetration(curvature, nominal_area);
    const double punch_stiffness = PunchStiffness(inv_effective_modulus, nominal_area);
    const double energy = 0.5 * effective_mass * speed * speed;

    // How deep the contact presses, where the approach energy has all gone into it.
    // A contact that fills its patch before then spends the rest of that energy against the constant stiffness, which solves a quadratic.
    const double sat_work = std::isfinite(sat_penetration) ? ContactWork(sat_penetration, hertz_stiffness, sat_penetration, punch_stiffness) : std::numeric_limits<double>::infinity();
    const double max_penetration = [&] {
        if (energy <= sat_work) return std::pow(energy / (0.4 * hertz_stiffness), 0.4);
        const double sat_force = hertz_stiffness * sat_penetration * std::sqrt(sat_penetration);
        return sat_penetration + (std::sqrt(sat_force * sat_force + 2 * punch_stiffness * (energy - sat_work)) - sat_force) / punch_stiffness;
    }();

    constexpr int Steps = 64;
    double sum = 0;
    for (int n = 0; n < Steps; ++n) {
        const double s = (double(n) + 0.5) / Steps;
        const double left = 1 - ContactWork(max_penetration * (1 - s * s), hertz_stiffness, sat_penetration, punch_stiffness) / energy;
        if (left > 0) sum += 2 * s / std::sqrt(left);
    }
    const double bulk_time = 2 * max_penetration / speed * sum / Steps * scale_ratio;
    const double u0 = 0.4 * combined_roughness;
    const double bed_time = std::numbers::sqrt2 * std::numbers::pi * u0 / speed;
    return std::clamp(std::sqrt(bulk_time * bulk_time + bed_time * bed_time), MinContactTime, MaxContactTime);
}
