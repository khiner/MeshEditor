#include "audio/ContactModel.h"

#include "Near.h"
#include "RunSuites.h"

#include <boost/ut.hpp>

#include <cmath>
#include <numbers>

using namespace boost::ut;

namespace {
// A striker whose mass, stiffness, and tip curvature all vanish from the harmonic sums, recovering the pure
// object-side Hertz contact (an infinitely heavy, hard, flat striker). Lets the formula tests stay object-only.
constexpr Striker NullStriker{
    .Material = {.Name = "null", .Properties = {.Density = 1e6, .YoungModulus = 1e30, .PoissonRatio = 0, .Alpha = 0, .Beta = 0}},
    .TipRadius = 1e6f,
    .Length = 1e6f,
};

constexpr AcousticMaterialProperties Polymer{.Density = 1000, .YoungModulus = 1e9, .PoissonRatio = 0.3, .Alpha = 0, .Beta = 0};
constexpr AcousticMaterialProperties Ceramic{.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19, .Alpha = 0, .Beta = 0};

// A body struck at one point, with `arm` from its center of mass to that point.
// A unit inverse inertia lets the arm shorten the contact, a zero one holds the effective mass at the body's own.
ContactDynamics Body(double mass, mat3 inverse_inertia, vec3 arm = vec3{0}) {
    ContactDynamics d;
    d.Mass = mass;
    d.InverseInertia = inverse_inertia;
    d.ContactArm = {arm};
    return d;
}

// The contact time of a strike along +z, with the striker's own compliance out of the way.
double ContactTime(const ContactDynamics &d, const AcousticMaterialProperties &material, double curvature, double area = 0, double speed = 1, double scale = 1, const Striker &striker = NullStriker) {
    return EstimateContactTime(d, 0, vec3{0, 0, 1}, speed, material, curvature, area, StrikerImpactor(striker), scale);
}
} // namespace

int main() {
    "inverse inertia round-trips a principal decomposition"_test = [] {
        MassProperties mp;
        mp.Mass = 1.0;
        mp.InertiaDiagonal = {2.f, 5.f, 9.f};
        mp.InertiaOrientation = numeric::Normalize(quat(0.3f, 0.1f, -0.5f, 0.8f)); // arbitrary orientation
        const auto rot = numeric::ToMat3(mp.InertiaOrientation);
        const mat3 diag{vec3{2, 0, 0}, vec3{0, 5, 0}, vec3{0, 0, 9}};
        const mat3 inertia = rot * diag * numeric::Transpose(rot);
        const mat3 product = inertia * InverseInertiaTensor(mp);
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r) expect(Near(product[c][r], c == r ? 1.0 : 0.0, 1e-4));
    };

    "contact time matches the Hertz formula"_test = [] {
        // Strike through the center of mass: effective mass is the total mass.
        const double tau = ContactTime(Body(1.0, mat3{1.f}), Polymer, 100);
        // Hand-computed: 2.87 * ((1*(1-0.09)/1e9)^2 * 100)^0.2 ~= 1.744e-3 s.
        expect(Near(tau, 1.744e-3, 2e-2));
    };

    "effective mass drops with an off-center strike"_test = [] {
        const double tau_center = ContactTime(Body(1.0, mat3{1.f}), Polymer, 100);
        const double tau_offset = ContactTime(Body(1.0, mat3{1.f}, vec3{0.2f, 0, 0}), Polymer, 100); // lever arm perpendicular to a z-strike
        // Lower effective mass shortens the contact.
        expect(tau_offset < tau_center);
    };

    "scale ratio and clamping"_test = [] {
        const auto d = Body(1.0, mat3{1.f});
        const auto tau = [&d](double scale) { return ContactTime(d, Polymer, 100, 0, 1, scale); };
        expect(Near(tau(2.0), 2 * tau(1.0), 1e-6)); // tau scales linearly with size, in range

        // Large scale saturates at the max, tiny scale at the min.
        expect(Near(tau(100.0), MaxContactTime));
        expect(Near(tau(1e-6), MinContactTime));
    };

    // Both closed forms are limits of the one collision integral, so it has to land on each of them where that limit applies.
    // A patch that never fills its polygon is Hertz, and one that fills it at once is the punch.
    "the contact time reaches both of its limits"_test = [] {
        const auto d = Body(1.0, mat3{0.f});
        constexpr double InvModulus = 0.91 / 1e9; // NullStriker's own compliance is 1e-30 of this
        const auto tau = [&d](double curvature, double area, double speed) { return ContactTime(d, Polymer, curvature, area, speed); };

        // No shared polygon, so the patch grows for the whole collision: 2.868 (m*^2/(E*^2 R* v))^(1/5).
        constexpr double Curvature = 100; // 1/m
        const double hertz = 2.868 * std::pow(std::pow(InvModulus, 2) * Curvature, 0.2);
        expect(Near(tau(Curvature, 0.0, 1.0), hertz, 1e-3));
        // A flat face fills its polygon at once, so the punch stands alone: pi sqrt(m*/k), k = 2 sqrt(A/pi) E*.
        constexpr double Area = 1e-4; // 1 cm^2
        const double punch = std::numbers::pi * std::sqrt(InvModulus / (2 * std::sqrt(Area / std::numbers::pi)));
        expect(Near(tau(0.0, Area, 1.0), punch, 1e-3));

        // Speed dependence is the regime's, not the model's: Hertz's v^(-1/5) against the punch's none.
        expect(Near(tau(Curvature, 0.0, 32.0) / tau(Curvature, 0.0, 1.0), std::pow(32.0, -0.2), 1e-3));
        expect(Near(tau(0.0, Area, 32.0) / tau(0.0, Area, 1.0), 1.0, 1e-3));
    };

    // A patch that fills its polygon partway through the collision is in neither limit.
    // It stops stiffening where Hertz would go on stiffening, so it outlasts Hertz, and it spent its first part softer than the punch, so it outlasts that too.
    "filling the patch stops the contact stiffening"_test = [] {
        const auto d = Body(0.5, mat3{0.f});
        constexpr double Curvature = 10; // 1/m, a 10 cm radius
        constexpr double Area = 1e-5; // a polygon the patch reaches only at the higher speeds
        const auto tau = [&d](double area, double speed) { return ContactTime(d, Ceramic, Curvature, area, speed); };

        expect(Near(SaturationPenetration(Curvature, Area), 3.183e-5, 1e-3));
        // Slowly enough, the patch never fills and the collision is exactly Hertz's.
        expect(Near(tau(Area, 0.1), tau(0.0, 0.1), 1e-6));
        // A polygon far wider than the patch ever grows is the same statement from the other side.
        expect(Near(tau(1.0, 3.0), tau(0.0, 3.0), 1e-6));
        // Fast enough, it fills and outlasts both limits.
        expect(tau(Area, 3.0) > tau(0.0, 3.0));
        expect(tau(Area, 3.0) > std::numbers::pi * std::sqrt(0.5 / PunchStiffness(0.91 / 7.2e10, Area)));

        // Nothing steps as the polygon crosses the depth the patch reaches, which is what one law buys.
        expect(Near(tau(1.7e-5, 1.0), tau(1.5e-5, 1.0), 1e-3));

        // Speed dependence is what saturating costs: over a 30x ladder Hertz alone falls by 30^(-1/5) and the punch alone not at all, and a contact that saturates partway falls by less than Hertz does.
        const double hertz_ratio = tau(0.0, 3.0) / tau(0.0, 0.1), saturating_ratio = tau(Area, 3.0) / tau(Area, 0.1);
        expect(Near(hertz_ratio, std::pow(30.0, -0.2), 1e-3));
        expect(saturating_ratio > hertz_ratio);
        expect(saturating_ratio < 1.0_d);
    };

    "a lighter striker shortens the contact against a heavy object"_test = [] {
        const auto d = Body(1000.0, mat3{0.f}); // heavy object, so the striker's mass dominates the reduced mass
        Striker light; // same material and tip, a longer capsule is only heavier
        light.Length = 0.05f;
        Striker heavy = light;
        heavy.Length = 5.f;

        expect(ContactTime(d, Ceramic, 5, 0, 1, 1, light) < ContactTime(d, Ceramic, 5, 0, 1, 1, heavy));
    };

    return RunSuites();
}
