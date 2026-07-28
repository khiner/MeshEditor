#pragma once

#include "AcousticMaterial.h"
#include "numeric/vec3.h"

#include <glm/gtc/quaternion.hpp>
#include <glm/mat3x3.hpp>

#include <cstdint>
#include <vector>

// Rigid-body mass properties in SI units, at the object's baked size and the solved material's
// density (ModalEigenSummary::SolvedMaterial). Mass and inertia are linear in density.
// Mirrors KHR_audio_modal `massProperties`: principal moments plus the rotation of the principal
// axes into node-local space.
struct MassProperties {
    double Mass{0}; // kg
    vec3 CenterOfMass{0}; // node-local units
    vec3 InertiaDiagonal{0}; // principal moments, kg·m²
    glm::quat InertiaOrientation{1, 0, 0, 0}; // principal inertia axes -> node-local

    bool operator==(const MassProperties &) const = default;
};

// Per-object contact dynamics, precomputed at the baked size. Node-local lengths are converted to SI.
// Drives the Hertz contact time of each strike. Aligned with ModalModes::Vertices/Positions.
struct ContactDynamics {
    double Mass{0}; // kg
    glm::mat3 InverseInertia{1}; // kg⁻¹·m⁻², about the center of mass
    std::vector<vec3> ContactArm; // per excitable vertex: contact point minus center of mass, meters
    std::vector<float> Curvature; // per excitable vertex: mean surface curvature, 1/m
};

// The virtual mallet a strike models: a capsule (rounded-tip cylinder) of some material, striking on its cap.
// Its mass is the material density times the capsule volume. A harder material or a lighter (shorter) capsule
// brightens the strike; the tip radius sets the contact curvature.
struct Striker {
    AcousticMaterial Material{materials::acoustic::Steel};
    float TipRadius{0.01f}; // m; cap radius, also the cylinder cross-section
    float Length{0.19f}; // m; cylinder length (~0.5 kg of steel at the default radius)
};

// The impactor side of a Hertz contact: elastic compliance, tip curvature, and inverse mass. A strike's virtual
// mallet and a colliding rigid body both reduce to this. InvMass 0 models an immovable impactor.
struct Impactor {
    AcousticMaterialProperties Material{};
    double Curvature{0}; // mean surface curvature at the contact, 1/m
    double InvMass{0}; // kg⁻¹
};

// Striker mass in kg: material density times capsule volume.
double StrikerMass(const Striker &);

// The mallet reduced to its impactor contribution.
Impactor StrikerImpactor(const Striker &);

// Inverse inertia tensor (kg⁻¹·m⁻²) reconstructed from the principal moments and orientation.
glm::mat3 InverseInertiaTensor(const MassProperties &);

// Reduced mass (kg) at the contact: the object's translational and rotational response to an off-center
// impulse, combined with the impactor. Drives the Hertz contact time and the impulse magnitude.
double ReducedContactMass(const ContactDynamics &, uint32_t excitable_index, vec3 impact_direction, const Impactor &);

/***** Hertz contact constants (Johnson 1985) *****/

// Effective compliance 1/E* = (1 - v1^2)/E1 + (1 - v2^2)/E2, in Pa^-1.
double InvEffectiveModulus(const AcousticMaterialProperties &, const AcousticMaterialProperties &);

// Combined curvature 1/R* = k1 + k2, in 1/m, held positive so a flat or concave surface reads as flat at R* = 1e6 m.
double CombinedCurvature(double curvature_a, double curvature_b);

// Contact stiffness k = (4/3) E* sqrt(R*), in N/m^(3/2). Relates load to penetration by N = k*delta^(3/2).
double ContactStiffness(double inv_effective_modulus, double combined_curvature);

// Hertz contact patch radius a = (3 N R* / (4 E*))^(1/3), in meters. Sets the contact filter's scale.
double ContactPatchRadius(double normal_force, double inv_effective_modulus, double combined_curvature);

// Equilibrium penetration under load N: delta0 = (N/k)^(2/3), in meters.
double StaticPenetration(double normal_force, double stiffness);

// Hertz contact time in seconds for a strike at `excitable_index`, from the object and impactor at `contact_speed`,
// scaled by `scale_ratio` for the object's current size.
double EstimateContactTime(const ContactDynamics &, uint32_t excitable_index, vec3 impact_direction, double contact_speed, const AcousticMaterialProperties &object_material, const Impactor &, double scale_ratio);

// Bounds on the derived contact time (seconds), guarding degenerate curvature, speed, and scale.
inline constexpr double MinContactTime = 2e-5, MaxContactTime = 5e-2;

// Approach speed a physics material's restitution is taken to be quoted at, m/s.
// Restitution varies with approach speed while the Hunt-Crossley dissipation constant stays fixed, so recovering that constant through e ~ 1 - alpha*v needs a reference speed.
inline constexpr float RestitutionReferenceSpeed = 1.f;
