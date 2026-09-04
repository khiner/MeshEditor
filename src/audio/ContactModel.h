#pragma once

#include "AcousticMaterial.h"
#include "numeric/mat3.h"
#include "numeric/quat.h"
#include "numeric/vec3.h"

#include <cstdint>
#include <vector>

using MassProperties = fastfem::MassProperties;

// Precomputed SI contact dynamics aligned with ModalModes::Vertices and Positions.
struct ContactDynamics {
    double Mass{0}; // kg
    mat3 InverseInertia{1}; // kg⁻¹·m⁻², about the center of mass
    std::vector<vec3> ContactArm; // per excitable vertex: contact point minus center of mass, meters
};

// Defines a capsule-shaped virtual striker with density-derived mass.
struct Striker {
    AcousticMaterial Material{materials::acoustic::Steel};
    float TipRadius{0.01f}; // Cap radius, also the cylinder cross-section, m
    float Length{0.19f}; // Cylinder length, m (~0.5 kg of steel at the default radius)
};

// Defines the compliance, curvature, and inverse mass of one side of a Hertz contact.
struct Impactor {
    AcousticMaterialProperties Material{};
    double Curvature{0}; // Its contribution to the combined contact curvature 1/R*, 1/m
    double InvMass{0}; // kg⁻¹
};

// Returns striker mass in kilograms.
double StrikerMass(const Striker &);

// Returns the striker's Hertz-contact parameters.
Impactor StrikerImpactor(const Striker &);

// Returns the inverse inertia tensor in inverse kilogram square meters.
mat3 InverseInertiaTensor(const MassProperties &);

// Returns the contact's reduced mass in kilograms, including off-center rotational response.
double ReducedContactMass(const ContactDynamics &, uint32_t excitable_index, vec3 impact_direction, const Impactor &);

/***** Hertz contact constants (Johnson 1985) *****/

// Effective compliance 1/E* = (1 - v1^2)/E1 + (1 - v2^2)/E2, in Pa^-1.
double InvEffectiveModulus(const AcousticMaterialProperties &, const AcousticMaterialProperties &);

// Returns combined curvature 1/R* = k1 + k2 in inverse meters.
// Zero curvature uses R* = 1e6 m to keep point and edge contacts finite.
double CombinedCurvature(double curvature_a, double curvature_b);

// Contact stiffness k = (4/3) E* sqrt(R*), in N/m^(3/2). Relates load to penetration by N = k*delta^(3/2).
double ContactStiffness(double inv_effective_modulus, double combined_curvature);

// Hertz contact patch radius a = (3 N R* / (4 E*))^(1/3), in meters. Sets the contact filter's scale.
double ContactPatchRadius(double normal_force, double inv_effective_modulus, double combined_curvature);

// Equilibrium penetration under load N: delta0 = (N/k)^(2/3), in meters.
double StaticPenetration(double normal_force, double stiffness);

// Penetration at which the growing Hertz patch fills the shared polygon, delta_sat = A0/(pi R*), in meters.
// Infinite where the contact has no shared polygon, so its patch never stops growing.
double SaturationPenetration(double combined_curvature, double nominal_area);

// Stiffness of the filled patch pressing as a flat punch, k = 2 a E* with a the radius of equal area, in N/m.
// This equals Hertz incremental stiffness at the patch radius.
double PunchStiffness(double inv_effective_modulus, double nominal_area);

double EstimateContactTime(const ContactDynamics &, uint32_t excitable_index, vec3 impact_direction, double contact_speed, const AcousticMaterialProperties &object_material, double object_curvature, double nominal_area, const Impactor &, double scale_ratio, double combined_roughness = 0);

// Bounds on the derived contact time (seconds), guarding degenerate curvature, speed, and scale.
inline constexpr double MinContactTime = 2e-5, MaxContactTime = 5e-2;
