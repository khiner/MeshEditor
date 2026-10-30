#include "audio/ContactModel.h"

#include "numeric/MatrixMath.h"

#include "Near.h"
#include <boost/ut.hpp>

#include <random>

using numeric::quat;

using namespace boost::ut;

namespace {
// Models an infinitely heavy, rigid, flat striker to isolate the object's Hertz contribution.
constexpr Striker NullStriker{
    .Material = {.Name = "null", .Properties = {.Density = 1e6, .YoungModulus = 1e30, .PoissonRatio = 0, .Alpha = 0, .Beta = 0}},
    .TipRadius = 1e6f,
    .Length = 1e6f,
};

constexpr AcousticMaterialProperties Polymer{.Density = 1000, .YoungModulus = 1e9, .PoissonRatio = 0.3, .Alpha = 0, .Beta = 0};
constexpr AcousticMaterialProperties Ceramic{.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19, .Alpha = 0, .Beta = 0};
// The polymer's (1 - nu^2) / E, which the null striker's own compliance is 1e-30 of.
constexpr double InvModulus = 0.91 / 1e9;

ContactDynamics Body(double mass, mat3 inverse_inertia = mat3{0.f}, vec3 arm = {0}) { return {.Mass = mass, .InverseInertia = inverse_inertia, .ContactArm = {arm}}; }

// Measure contact duration along +z with striker compliance excluded.
double ContactTime(const ContactDynamics &d, const AcousticMaterialProperties &material, double curvature, double area = 0, double speed = 1, double scale = 1, const Striker &striker = NullStriker) {
    return EstimateContactTime(d, 0, vec3{0, 0, 1}, speed, material, curvature, area, StrikerImpactor(striker), scale);
}
// Hertz's contact time for a patch that grows for the whole collision: 2.868 (m^2 / (E*^2 R v))^(1/5).
double Hertz(double mass, double curvature, double speed) { return 2.868 * std::pow(mass * mass * InvModulus * InvModulus * curvature / speed, 0.2); }
// The flat punch's contact time for a patch that fills its polygon at once: pi sqrt(m / (2 E* a)).
double Punch(double mass, double area) { return std::numbers::pi * std::sqrt(mass * InvModulus / (2 * std::sqrt(area / std::numbers::pi))); }
} // namespace
const boost::ut::suite ContactModel = [] {
    "the inverse inertia tensor inverts every principal decomposition"_test = [] {
        std::mt19937 rng{3};
        std::uniform_real_distribution<float> unit{-1.f, 1.f}, moment{.5f, 9.f};
        for (int trial = 0; trial < 16; ++trial) {
            const MassProperties mp{.Mass = 1.0, .InertiaDiagonal = {moment(rng), moment(rng), moment(rng)}, .InertiaOrientation = numeric::Normalize(quat(unit(rng), unit(rng), unit(rng), unit(rng)))};
            const auto rot = numeric::ToMat3(mp.InertiaOrientation);
            const mat3 diag{vec3{mp.InertiaDiagonal.x, 0, 0}, vec3{0, mp.InertiaDiagonal.y, 0}, vec3{0, 0, mp.InertiaDiagonal.z}};
            const mat3 product = rot * diag * numeric::Transpose(rot) * InverseInertiaTensor(mp);
            for (int c = 0; c < 3; ++c)
                for (int r = 0; r < 3; ++r) expect(Near(product[c][r], c == r ? 1.0 : 0.0, 1e-4));
        }
    };

    // The collision integral reaches Hertz's closed form wherever the patch never fills a polygon, scaled linearly by size and clamped to the contact time range.
    "the contact time is Hertz's over mass, curvature, speed and scale"_test = [] {
        for (const double mass : {0.1, 1.0, 10.0})
            for (const double curvature : {1.0, 10.0, 100.0})
                for (const double speed : {0.1, 1.0, 32.0})
                    for (const double scale : {1e-6, 0.5, 1.0, 2.0, 100.0}) {
                        const auto tau = ContactTime(Body(mass), Polymer, curvature, 0, speed, scale);
                        const auto expected = std::clamp(scale * Hertz(mass, curvature, speed), MinContactTime, MaxContactTime);
                        expect(Near(tau, expected, 1e-3)) << mass << curvature << speed << scale << tau << expected;
                    }
    };
    // A patch that fills its polygon at once is the flat punch, whose contact time does not depend on speed.
    "the contact time is the punch's over mass, area and speed"_test = [] {
        for (const double mass : {0.1, 1.0, 10.0})
            for (const double area : {1e-5, 1e-4, 1e-3})
                for (const double speed : {0.1, 1.0, 32.0}) {
                    const auto tau = ContactTime(Body(mass), Polymer, 0.0, area, speed);
                    const auto expected = std::clamp(Punch(mass, area), MinContactTime, MaxContactTime);
                    expect(Near(tau, expected, 1e-3)) << mass << area << speed << tau << expected;
                }
    };

    // A patch that fills its polygon partway through the collision is in neither limit.
    // Partial saturation produces a contact duration above both limiting laws.
    "filling the patch stops the contact stiffening"_test = [] {
        const auto d = Body(0.5);
        constexpr double Curvature = 10; // 1/m, a 10 cm radius
        constexpr double Area = 1e-5; // a polygon the patch reaches only at the higher speeds
        const auto tau = [&d](double area, double speed) { return ContactTime(d, Ceramic, Curvature, area, speed); };

        expect(Near(SaturationPenetration(Curvature, Area), 3.183e-5, 1e-3));
        // Slowly enough, the patch never fills and the collision is exactly Hertz's.
        expect(Near(tau(Area, 0.1), tau(0.0, 0.1), 1e-6));
        expect(Near(tau(1.0, 3.0), tau(0.0, 3.0), 1e-6));
        expect(tau(Area, 3.0) > tau(0.0, 3.0));
        expect(tau(Area, 3.0) > std::numbers::pi * std::sqrt(0.5 / PunchStiffness(0.91 / 7.2e10, Area)));

        expect(Near(tau(1.7e-5, 1.0), tau(1.5e-5, 1.0), 1e-3));

        const double hertz_ratio = tau(0.0, 3.0) / tau(0.0, 0.1), saturating_ratio = tau(Area, 3.0) / tau(Area, 0.1);
        expect(Near(hertz_ratio, std::pow(30.0, -0.2), 1e-3));
        expect(saturating_ratio > hertz_ratio);
        expect(saturating_ratio < 1.0_d);
    };

    // The reduced mass falls with a lever arm perpendicular to the strike and with a lighter striker against a heavy object, and the contact shortens with it.
    "a smaller reduced mass shortens the contact"_test = [] {
        expect(ContactTime(Body(1.0, mat3{1.f}, vec3{0.2f, 0, 0}), Polymer, 100) < ContactTime(Body(1.0, mat3{1.f}), Polymer, 100));
        Striker light; // same material and tip, a longer capsule is only heavier
        light.Length = 0.05f;
        Striker heavy = light;
        heavy.Length = 5.f;
        expect(ContactTime(Body(1000.0), Ceramic, 5, 0, 1, 1, light) < ContactTime(Body(1000.0), Ceramic, 5, 0, 1, 1, heavy));
    };
};
