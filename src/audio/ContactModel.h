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

// Striker mass in kg: material density times capsule volume.
double StrikerMass(const Striker &);

// Inverse inertia tensor (kg⁻¹·m⁻²) reconstructed from the principal moments and orientation.
glm::mat3 InverseInertiaTensor(const MassProperties &);

// Reduced mass (kg) at the contact: the object's translational and rotational response to an off-center
// impulse, combined with the striker. Drives the Hertz contact time and the impulse magnitude.
double ReducedContactMass(const ContactDynamics &, uint32_t excitable_index, vec3 impact_direction, const Striker &);

// Hertz contact time in seconds for a strike at `excitable_index`, from the object and striker at `contact_speed`,
// scaled by `scale_ratio` for the object's current size.
double EstimateContactTime(const ContactDynamics &, uint32_t excitable_index, vec3 impact_direction, double contact_speed, const AcousticMaterialProperties &object_material, const Striker &, double scale_ratio);

// Bounds on the derived contact time (seconds), guarding degenerate curvature, speed, and scale.
inline constexpr double MinContactTime = 2e-5, MaxContactTime = 5e-2;
