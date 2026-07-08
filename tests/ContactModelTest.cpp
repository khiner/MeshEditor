#include "audio/ContactModel.h"
#include "audio/AcousticMaterialProperties.h"

#include <boost/ut.hpp>

#include <cmath>

using namespace boost::ut;

namespace {
bool Near(double a, double b, double rel = 1e-6) { return std::abs(a - b) <= rel * std::max(1.0, std::abs(b)); }

// A striker whose mass, stiffness, and tip curvature all vanish from the harmonic sums, recovering the pure
// object-side Hertz contact (an infinitely heavy, hard, flat striker). Lets the formula tests stay object-only.
constexpr Striker NullStriker{
    .Material = {.Name = "null", .Properties = {.Density = 1e6, .YoungModulus = 1e30, .PoissonRatio = 0, .Alpha = 0, .Beta = 0}},
    .TipRadius = 1e6f,
    .Length = 1e6f,
};
} // namespace

int main() {
    "inverse inertia round-trips a principal decomposition"_test = [] {
        MassProperties mp;
        mp.Mass = 1.0;
        mp.InertiaDiagonal = {2.f, 5.f, 9.f};
        mp.InertiaOrientation = glm::normalize(glm::quat(0.3f, 0.1f, -0.5f, 0.8f)); // arbitrary orientation
        const auto rot = glm::mat3_cast(mp.InertiaOrientation);
        const glm::mat3 diag{glm::vec3{2, 0, 0}, glm::vec3{0, 5, 0}, glm::vec3{0, 0, 9}};
        const glm::mat3 inertia = rot * diag * glm::transpose(rot);
        const glm::mat3 product = inertia * InverseInertiaTensor(mp);
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r) expect(Near(product[c][r], c == r ? 1.0 : 0.0, 1e-4));
    };

    "contact time matches the Hertz formula"_test = [] {
        ContactDynamics d;
        d.Mass = 1.0;
        d.InverseInertia = glm::mat3{1.0f};
        d.ContactArm = {vec3{0}}; // strike through the center of mass: effective mass is the total mass
        d.Curvature = {100.f}; // 1/m
        const AcousticMaterialProperties mat{.Density = 1000, .YoungModulus = 1e9, .PoissonRatio = 0.3, .Alpha = 0, .Beta = 0};

        const double tau = EstimateContactTime(d, 0, vec3{0, 0, 1}, 1.0, mat, NullStriker, 1.0);
        // Hand-computed: 2.87 * ((1*(1-0.09)/1e9)^2 * 100)^0.2 ~= 1.744e-3 s.
        expect(Near(tau, 1.744e-3, 2e-2));
    };

    "effective mass drops with an off-center strike"_test = [] {
        ContactDynamics base;
        base.Mass = 1.0;
        base.InverseInertia = glm::mat3{1.0f};
        base.Curvature = {100.f};
        const AcousticMaterialProperties mat{.Density = 1000, .YoungModulus = 1e9, .PoissonRatio = 0.3, .Alpha = 0, .Beta = 0};

        auto centered = base;
        centered.ContactArm = {vec3{0}};
        auto offset = base;
        offset.ContactArm = {vec3{0.2f, 0, 0}}; // lever arm perpendicular to a z-strike

        const double tau_center = EstimateContactTime(centered, 0, vec3{0, 0, 1}, 1.0, mat, NullStriker, 1.0);
        const double tau_offset = EstimateContactTime(offset, 0, vec3{0, 0, 1}, 1.0, mat, NullStriker, 1.0);
        // Lower effective mass shortens the contact.
        expect(tau_offset < tau_center);
    };

    "scale ratio and clamping"_test = [] {
        ContactDynamics d;
        d.Mass = 1.0;
        d.InverseInertia = glm::mat3{1.0f};
        d.ContactArm = {vec3{0}};
        d.Curvature = {100.f};
        const AcousticMaterialProperties mat{.Density = 1000, .YoungModulus = 1e9, .PoissonRatio = 0.3, .Alpha = 0, .Beta = 0};

        const double tau1 = EstimateContactTime(d, 0, vec3{0, 0, 1}, 1.0, mat, NullStriker, 1.0);
        const double tau2 = EstimateContactTime(d, 0, vec3{0, 0, 1}, 1.0, mat, NullStriker, 2.0);
        expect(Near(tau2, 2 * tau1, 1e-6)); // tau scales linearly with size, in range

        // Large scale saturates at the max, tiny scale at the min.
        expect(Near(EstimateContactTime(d, 0, vec3{0, 0, 1}, 1.0, mat, NullStriker, 100.0), MaxContactTime));
        expect(Near(EstimateContactTime(d, 0, vec3{0, 0, 1}, 1.0, mat, NullStriker, 1e-6), MinContactTime));
    };

    "a lighter striker shortens the contact against a heavy object"_test = [] {
        ContactDynamics d;
        d.Mass = 1000.0; // heavy object, so the striker's mass dominates the reduced mass
        d.InverseInertia = glm::mat3{0.f};
        d.ContactArm = {vec3{0}};
        d.Curvature = {5.f};
        const AcousticMaterialProperties mat{.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19, .Alpha = 0, .Beta = 0};

        Striker light; // same material and tip, a longer capsule is only heavier
        light.Length = 0.05f;
        Striker heavy = light;
        heavy.Length = 5.f;

        const double tau_light = EstimateContactTime(d, 0, vec3{0, 0, 1}, 1.0, mat, light, 1.0);
        const double tau_heavy = EstimateContactTime(d, 0, vec3{0, 0, 1}, 1.0, mat, heavy, 1.0);
        expect(tau_light < tau_heavy);
    };
}
