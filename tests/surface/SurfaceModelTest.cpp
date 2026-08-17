#include "audio/surface/SurfaceModel.h"

#include "audio/ContactSurface.h"

#include "Near.h"
#include "RunSuites.h"

#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

using namespace boost::ut;

namespace {
// The machined finish of the sample scenes, and ceramic on ceramic, so the bed cases below are the contact the corpus actually renders rather than an invented one.
constexpr float MachinedCorrelation = 5e-5f, MachinedSlope = -2.4f, MachinedRoughness = 2e-6f;
const double CeramicInvModulus = InvEffectiveModulus(materials::acoustic::Ceramic.Properties, materials::acoustic::Ceramic.Properties);

// Two machined faces sharing `bound_area` under `load`, assembled exactly as ResolveContact assembles it.
struct BedCase {
    AsperityBed Bed;
    double GapRoughness{0}; // Both surfaces' heights in the one gap
    double RealArea{0};
};

// Two identical machined faces sharing `bound_area` under `load`, assembled exactly as ResolveContact does.
// A machined face is flat to a couple of microns over a couple of millimetres (ISO 4287 separates roughness from waviness for exactly this reason), and that waviness is what holds most of the face apart, so the asperities stand over what it leaves rather than over the whole shared polygon.
BedCase MachinedBed(double load, double bound_area) {
    constexpr double Roughness = MachinedRoughness, Waviness = 2e-6, WavinessLength = 2e-3;
    const double inv_modulus = CeramicInvModulus;
    // The preset finish's band, and the spacing that resolves its short-wavelength end, exactly as a surface and SynthesizedFinishSpacing give them to ResolveContact.
    // The band is fixed, so the track it synthesizes to is built once and every case reads the same one.
    constexpr double cutoff = MachinedCorrelation / PresetBandRatio;
    const double spacing = double(FinishTrackSpacing(float(cutoff)));
    static const auto track = SynthesizeRoughness(MachinedCorrelation, MachinedSlope, float(cutoff), float(spacing), TrackSamples);
    const double gradient = Roughness * track.SlopeRms / spacing;
    const double curvature = Roughness * track.CurvatureRms / (spacing * spacing);
    // Two identical surfaces, so every quantity of the gap is the single surface's times sqrt(2).
    const double gap_gradient = std::numbers::sqrt2 * gradient;
    const double real_area = RealContactArea(load, bound_area, inv_modulus, gap_gradient);
    // The same area law one scale up: waviness holds most of a nominally flat face apart, and the asperities are available over what it leaves.
    const double meso_gradient = std::numbers::sqrt2 * Waviness / WavinessLength;
    const double confined = RealContactArea(load, bound_area, inv_modulus, meso_gradient);
    const double patch_width = 2 * std::sqrt(confined / std::numbers::pi);
    const double region_width = std::max(2 * std::sqrt(bound_area / std::numbers::pi), patch_width);
    const double peak_density = (curvature / gradient) / (2 * std::numbers::pi);
    // The shipping cell window: the bearing share, floored at the half-variance width, capped at the cell's own population span.
    const double cell_window = BedCellWindow(patch_width, region_width, peak_density, double(TrackHalfVarianceWidth(track)) * spacing);
    const auto bed = ResolveAsperityBed(
        load, patch_width, region_width, inv_modulus, peak_density, cell_window,
        [&](double) { return std::numbers::sqrt2 * curvature; },
        [&](double width) { return std::numbers::sqrt2 * Roughness * TrackWindowRms(track, float(width / spacing)); }
    );
    return {bed, std::numbers::sqrt2 * Roughness, real_area};
}

// The separation Persson's law decays the conformal pressure over, which is the mean the bed must reproduce.
double MachinedConformalLength(double gap_roughness) {
    return gap_roughness / PerssonSeparationCoefficients(HurstExponent(MachinedSlope), RoughnessBandRatio).Alpha;
}
// A cell's force with the sub-population fluctuation its mean leaves out, given the cell's separation in units of its own spread and a unit-variance draw.
// The spots a cell stands for bear independently, so the sum's variance is their count times one spot's, and a contact never pulls.
double CellForce(const AsperityBed &bed, double lambda, double draw) {
    const double one_spot = bed.SpotStiffness * bed.CellSpread * std::sqrt(bed.CellSpread);
    const double mean = bed.SpotWeight * one_spot * BedLoadFactor(float(lambda));
    const double deviation = std::sqrt(bed.SpotWeight * BedVarianceFactor(float(lambda))) * one_spot;
    return std::max(mean + draw * deviation, 0.0);
}

// A synthesized patch scaled to a real roughness, which is how a finish reaches the springs.
std::vector<float> RoughHeights(float correlation, float slope, float cutoff, float spacing, uint32_t columns, uint32_t rows, uint32_t realization, float roughness) {
    const auto patch = SynthesizeRoughnessPatch(correlation, slope, cutoff, spacing, columns, rows, realization);
    std::vector<float> heights(patch.Heights.size());
    for (size_t i = 0; i < heights.size(); ++i) heights[i] = patch.Heights[i] * roughness;
    return heights;
}

// The transverse gap a curved body holds open, as a parabola of `curvature` about the middle row.
std::vector<float> TransverseGap(uint32_t rows, double spacing, double curvature) {
    std::vector<float> gap(rows);
    for (uint32_t r = 0; r < rows; ++r) {
        const double y = (double(r) - 0.5 * (rows - 1)) * spacing;
        gap[r] = float(0.5 * curvature * y * y);
    }
    return gap;
}

// The field a contact bears on: the surface less the body's own gap.
std::vector<float> FoldGap(const std::vector<float> &heights, uint32_t columns, uint32_t rows, const std::vector<float> &gap) {
    std::vector<float> folded(size_t(columns) * rows);
    for (uint32_t c = 0; c < columns; ++c) {
        for (uint32_t r = 0; r < rows; ++r) folded[size_t(c) * rows + r] = heights[size_t(c) * rows + r] - gap[r];
    }
    return folded;
}

// Rms height step between neighbouring samples of a patch, along its columns or across its rows.
// A spectrum with no preferred direction reads the same either way, which is what lets one patch serve a sweep in any direction over it.
double PatchStepRms(const RoughnessField &patch, bool along) {
    double energy = 0;
    size_t count = 0;
    for (uint32_t i = 0; i + 1 < (along ? patch.Columns : patch.Rows); ++i) {
        for (uint32_t j = 0; j < (along ? patch.Rows : patch.Columns); ++j) {
            const size_t at = along ? size_t(i) * patch.Rows + j : size_t(j) * patch.Rows + i;
            const size_t next = along ? size_t(i + 1) * patch.Rows + j : size_t(j) * patch.Rows + i + 1;
            const double d = double(patch.Heights[next]) - double(patch.Heights[at]);
            energy += d * d;
            ++count;
        }
    }
    return std::sqrt(energy / double(count));
}

} // namespace

int main() {
    "the Hurst exponent inverts the spectral slope"_test = [] {
        expect(Near(HurstExponent(-2.4), 0.7, 1e-12)); // machined
        expect(Near(HurstExponent(-2.8), 0.9, 1e-12)); // polished
        expect(Near(HurstExponent(-2.0), 0.5, 1e-12)); // cast
    };

    // Persson 2007 Fig. 2 plots both against H for three band ratios. These are that figure at gamma = 1.
    "Persson's separation coefficients match the published figure"_test = [] {
        struct Row {
            double Hurst, BandRatio, Alpha, Beta;
        };
        for (const auto [hurst, ratio, alpha, beta] : std::array{
                 Row{0.5, 1e2, 0.852, 0.565},
                 Row{0.8, 1e2, 0.949, 0.419},
                 Row{0.8, 1e4, 0.940, 0.429},
                 Row{0.9, 1e3, 0.973, 0.396},
             }) {
            const auto c = PerssonSeparationCoefficients(hurst, ratio, 1.0); // his figure is drawn at gamma = 1
            expect(Near(c.Alpha, alpha, 1e-3));
            expect(Near(c.Beta, beta, 1e-3));
        }
    };

    // Tiwari and Persson 2020 give the decay length as u0 = 0.4 * roughness, fitted against numerical solutions.
    // Persson's own gamma = 1 puts it near the roughness itself, which is 2.5 times too soft.
    "the default gamma reproduces the fitted decay length"_test = [] {
        const auto fitted = PerssonSeparationCoefficients(0.8, 1e4);
        expect(Near(1 / fitted.Alpha, 0.4, 0.05)); // u0 / roughness
        const auto simplest = PerssonSeparationCoefficients(0.8, 1e4, 1.0);
        expect(Near(fitted.Alpha / simplest.Alpha, 1 / SeparationGamma, 1e-9)); // Alpha scales as 1/gamma
    };

    // The one expression has to reach both limits on its own, since nothing else selects between them.
    "the real contact area is the load over the asperity pressure while it is small"_test = [] {
        constexpr double inv_modulus = 1 / 7.2e10, slope = 0.2;
        const double p = AsperityPressure(inv_modulus, slope);
        for (const double bound : {1e-4, 1e-2, 1.0}) {
            expect(Near(RealContactArea(5.0, bound, inv_modulus, slope) / (5.0 / p), 1.0, 1e-6)) << bound;
        }
    };

    "the real contact area saturates at the region bounding the contact"_test = [] {
        constexpr double inv_modulus = 1 / 7.2e10, slope = 0.2, bound = 1e-6;
        const double p = AsperityPressure(inv_modulus, slope);
        // Ten times the load that would fill the region at the asperity pressure.
        const double area = RealContactArea(10 * bound * p, bound, inv_modulus, slope);
        expect(area <= bound);
        expect(area > 0.99 * bound);
    };

    "conformal separation inverts conformal pressure"_test = [] {
        const auto c = PerssonSeparationCoefficients(0.7, 1e3);
        constexpr double roughness = 2e-6, inv_modulus = 1 / 7.2e10;
        const double q0 = RollOffWavevector(5e-5);
        for (const double u : {0.0, 1e-6, 5e-6, 2e-5}) {
            const double p = ConformalPressure(u, roughness, q0, inv_modulus, c);
            expect(Near(ConformalSeparation(p, roughness, q0, inv_modulus, c), u, 1e-12));
        }
    };

    "conformal pressure falls off over a length of order the roughness"_test = [] {
        const auto c = PerssonSeparationCoefficients(0.7, 1e3);
        constexpr double roughness = 2e-6, inv_modulus = 1 / 7.2e10;
        const double q0 = RollOffWavevector(5e-5);
        const double near_p = ConformalPressure(0, roughness, q0, inv_modulus, c);
        const double far_p = ConformalPressure(roughness / c.Alpha, roughness, q0, inv_modulus, c);
        expect(Near(far_p / near_p, std::exp(-1.0), 1e-12)); // one separation constant is one e-fold
    };

    "conformal stiffness is linear in load, unlike Hertz"_test = [] {
        const auto c = PerssonSeparationCoefficients(0.7, 1e3);
        constexpr double roughness = 2e-6;
        const double k1 = ConformalStiffness(10.0, roughness, c);
        const double k2 = ConformalStiffness(20.0, roughness, c);
        expect(Near(k2 / k1, 2.0, 1e-12));
        // A face on a face is orders softer than the flat-guard Hertz stiffness it replaces.
        expect(k1 < ContactStiffness(1 / 7.2e10, CombinedCurvature(0, 0)));
    };

    "the roll-off wavevector is angular"_test = [] {
        expect(Near(RollOffWavevector(5e-5), 2 * std::numbers::pi / 5e-5, 1e-6));
    };

    /***** The asperity bed *****/

    "the bed height integral reaches its unclamped limit"_test = [] {
        // Far above the height spread every spot bears, the clamp at zero never binds, and the integral is the mean of a power of a variable that is lambda plus a fluctuation small against it.
        for (const double power : {0.5, 1.0, 1.5}) {
            expect(Near(BedHeightIntegral(40, power) / std::pow(40.0, power), 1.0, 0.01));
        }
        // Far below it nothing bears at all.
        expect(BedHeightIntegral(-8, 1.5) < 1e-12);
        // And it rises monotonically between.
        double previous = 0;
        for (double lambda = -4; lambda <= 4; lambda += 0.25) {
            const double v = BedHeightIntegral(lambda, 1.5);
            expect(v > previous);
            previous = v;
        }
    };

    "the bed's separation is the one that carries its load"_test = [] {
        const auto machined = MachinedBed(4.905, 8.1e-3); // a 500 g body on a 9 cm face
        expect(machined.Bed.SpotCount > 1u);
        expect(Near(BedLoad(machined.Bed) / 4.905, 1.0, 1e-6));
    };

    "the bed reproduces Persson's stiffness"_test = [] {
        // Persson's exponential pressure-separation law is the mean of a bed, so the bed's own stiffness at the separation it carries its load at has to come back as Alpha*N/h_rms.
        // The two are independent theories of one interface, so they are asked to agree in magnitude rather than to a digit.
        for (const double load : {0.5, 4.905, 50.0}) {
            const auto machined = MachinedBed(load, 8.1e-3);
            const auto coeffs = PerssonSeparationCoefficients(HurstExponent(MachinedSlope), RoughnessBandRatio);
            const double persson = ConformalStiffness(load, machined.GapRoughness, coeffs);
            const double ratio = BedStiffness(machined.Bed) / persson;
            expect(ratio > 0.5 && ratio < 2.0) << "bed stiffness over Persson's" << ratio;
        }
    };

    "the bed's mean law is exponential in separation"_test = [] {
        // What makes a bed reproduce Persson is that its load rises by e over one conformal length, and it has to do that over the load range rather than at one point.
        auto machined = MachinedBed(4.905, 8.1e-3);
        const double u0 = MachinedConformalLength(machined.GapRoughness);
        const double step = u0 / machined.Bed.HeightRms;
        for (const double from : {-1.0, 0.0, 1.0}) {
            machined.Bed.Separation = from;
            const double before = BedLoad(machined.Bed);
            machined.Bed.Separation = from + step;
            const double folds = BedLoad(machined.Bed) / before;
            expect(folds > std::numbers::e / 2 && folds < std::numbers::e * 2) << "e-folds per conformal length" << folds;
        }
    };

    "the bed touches over the area the area law gives"_test = [] {
        // The bed implies its own real contact area, and the erf law is what the rest of the model uses.
        // One sums Hertz spots over a height distribution and the other is Pastewka and Robbins' closed form in load and surface gradient, so the two are independent accounts of one area and agree on it.
        std::array<double, 3> ratios{};
        const std::array loads{0.5, 4.905, 50.0};
        for (size_t i = 0; i < loads.size(); ++i) {
            const auto machined = MachinedBed(loads[i], 8.1e-3);
            ratios[i] = BedContactArea(machined.Bed) / machined.RealArea;
        }
        for (const double ratio : ratios) {
            expect(Near(ratio / ratios[0], 1.0, 1e-3)) << "the load dependence has to be the same";
            // 1.031 at the current parameters.
            // The band holds it well inside the factor near two that separates asperity models from Persson theory.
            expect(ratio > 0.7 && ratio < 1.5) << "bed area over the erf law's" << ratio;
        }
    };

    "the asperity radius is the one Pastewka and Robbins state"_test = [] {
        // Pastewka and Robbins 2016 Eq. (6) states it in a surface's own quantities as rho = lambda_s / (2 pi h'_rms) sqrt(2(2-H)/(1-H)), so the radius is proportional to the band's short-wavelength end.
        // A surface authors that end and the track carries nothing below it, so both sides move together and the law is tested rather than one point of it.
        // Their h'_rms is the two-dimensional gradient <|grad h|^2>^(1/2), which for an isotropic surface is sqrt(2) times what one cut through it measures, and a track measures the cut.
        // The comparison holds only where the surface is self-affine over a wide band and the second difference resolves the cutoff: sampled AT the cutoff the difference reads about 40 percent of the true curvature, and a band of one octave is not self-affine at all.
        for (const double slope : {-1.4, -1.8, -2.0, -2.4}) {
            const double hurst = HurstExponent(slope);
            for (const float cutoff : {8.f, 16.f, 32.f}) {
                // Unit roughness and unit spacing, so both radii come out in samples.
                const auto track = SynthesizeRoughness(4096.f, float(slope), cutoff, 1.f, TrackSamples);
                const double ours = 1.0 / track.CurvatureRms;
                const double gradient_rms = std::numbers::sqrt2 * track.SlopeRms;
                const double theirs = double(cutoff) / (2 * std::numbers::pi * gradient_rms) * std::sqrt(2 * (2 - hurst) / (1 - hurst));
                expect(Near(theirs / ours, 1.0, 0.05)) << "asperity radius against Pastewka's, at slope" << slope << "cutoff" << cutoff << theirs / ours;
            }
        }
    };

    "the gradient at a width falls as the self-affine spectrum says"_test = [] {
        // The flank tilt is read at the width of the contact bearing on it, so the track measures its gradient at each boxcar width.
        // The narrowest width is the adjacent difference itself, which is what makes the wider reads a coarsening of the same quantity rather than a second convention.
        for (const double slope : {-1.8, -2.0, -2.4}) {
            const auto track = SynthesizeRoughness(64.f, float(slope), 2.f, 1.f, TrackSamples);
            expect(Near(double(track.SlopeWindowRms[0]) / track.SlopeRms, 1.0, 1e-4))
                << "the width-one gradient is the adjacent difference, at slope" << slope;
            // A self-affine profile's gradient falls with the scale it is read at, monotonically and without a plateau, which is why the scale has to be chosen rather than converged to.
            for (uint32_t i = 1; i < 8; ++i) {
                expect(track.SlopeWindowRms[i] < track.SlopeWindowRms[i - 1])
                    << "the gradient falls with width, at slope" << slope << "octave" << i;
            }
            // Between the measured octaves the read interpolates, so a width of one octave up agrees.
            expect(Near(double(TrackSlopeAtWidth(track, 4.f)) / track.SlopeWindowRms[2], 1.0, 1e-5))
                << "an exact power of two reads its own entry, at slope" << slope;
        }
    };

    "one spot is one Hertz spring"_test = [] {
        // A bed of a single spot pressed well past its height spread is Hertz outright, which is what makes the spot count a property of the surface rather than a setting that changes the law.
        const auto bed = ResolveAsperityBed(1.0, 1e-4, 1e-4, 1 / 7.2e10, 0.0, 1e-4, [](double) { return 1e4; }, [](double) { return 1e-7; });
        expect(bed.SpotCount == 1u);
        expect(Near(bed.TotalSpots, 1.0, 1e-12));
        AsperityBed pressed = bed;
        pressed.Separation = 40;
        const double depth = pressed.Separation * pressed.HeightRms;
        expect(Near(BedLoad(pressed) / (pressed.SpotStiffness * depth * std::sqrt(depth)), 1.0, 0.01));
        expect(Near(BedStiffness(pressed) / (1.5 * pressed.SpotStiffness * std::sqrt(depth)), 1.0, 0.01));
        expect(Near(BedContactArea(pressed) / (std::numbers::pi * pressed.SpotRadius * depth), 1.0, 0.01));
    };

    "a vanishing cell spread is Hertz outright"_test = [] {
        // The bed factors carry their closed-form asymptotes outside the tabulated range, so the statistical cell force collapses continuously onto the bare Hertz spot as its spread vanishes.
        // The two force branches therefore agree across the CellSpread boundary, and a bed sitting on it is not a special case of the law but only of its evaluation.
        AsperityBed bed{.SpotWeight = 0.37, .SpotStiffness = 2.4e9, .CellSpread = 1e-9};
        for (const double engaged : {5e-8, 2e-7, 1e-6, 5e-6}) {
            const double statistical = CellForce(bed, engaged / bed.CellSpread, 0.0);
            const double hertz = bed.SpotWeight * bed.SpotStiffness * engaged * std::sqrt(engaged);
            expect(Near(statistical / hertz, 1.0, 1e-3)) << "at engagement" << engaged;
            // The sub-population deviation falls as the spread over the engagement, so the draw goes silent as the spread vanishes.
            const double deviation = CellForce(bed, engaged / bed.CellSpread, 1.0) - statistical;
            expect(deviation / hertz < 5.0 * bed.CellSpread / engaged) << "deviation share at engagement" << engaged;
        }
        // Separated, both branches carry nothing.
        expect(CellForce(bed, -5e-7 / bed.CellSpread, 0.0) == 0.0);
    };

    "the potential tables integrate the mean law exactly"_test = [] {
        // The energy-quadratised exchange stores each cell's elastic potential.
        // The mean-law part's derivative in the separation must recover the load factor, which is what makes psi = sqrt(2 * Phi) and the force one coherent object rather than a law plus a patch.
        // The sites' own potentials are closed-form Hertz and need no table.
        const float h = 0.05f;
        for (const float lam : {-4.f, -1.f, 0.f, 1.f, 3.f, 6.f}) {
            expect(Near((BedPotentialFactor(lam + h) - BedPotentialFactor(lam - h)) / (2 * h), BedLoadFactor(lam), 2e-2)) << "load at" << lam;
            expect(Near((BedLinearPotentialFactor(lam + h) - BedLinearPotentialFactor(lam - h)) / (2 * h), BedLinearLoadFactor(lam), 2e-2)) << "linear load at" << lam;
        }
        // The closed-form tails continue the tables.
        expect(Near(BedPotentialFactor(8.01f), BedPotentialFactor(7.99f), 1e-2));
        expect(Near(BedLinearPotentialFactor(8.01f), BedLinearPotentialFactor(7.99f), 1e-2));
    };

    "the asperity pressure carries Pastewka and Robbins' own constant"_test = [] {
        // They write the pressure the asperities bear the load at as E*|grad h|_rms/kappa and put 1/kappa near 1/2 in the continuum hard-wall limit.
        // Ours takes the gradient along one profile, which for an isotropic surface is the full gradient over sqrt(2), so the same constant has to come back out.
        constexpr double inv_modulus = 1 / 7.2e10, profile_slope = 0.04;
        const double full_gradient = std::numbers::sqrt2 * profile_slope;
        const double kappa = full_gradient / (inv_modulus * AsperityPressure(inv_modulus, profile_slope));
        expect(Near(kappa, 2.0, 1e-9)) << "kappa" << kappa;
    };

    "the bed's potential integrates its load law"_test = [] {
        // The psi exchange meters energy through the potential while the load drives the modes, so the pair must satisfy F = dU/d(engagement) at every operating point or the meter drifts silently at each advance.
        // Central differences across half a table cell cover the spread branches, the beyond-table asymptotes, and the Hertz-outright limit at zero spread.
        // Any future force law added to the bed extends this gate before it renders.
        for (const float spread : {1e-7f, 1e-6f, 1e-5f}) {
            for (const float lambda : {-6.f, -3.f, -1.f, 0.f, 1.f, 3.f, 6.f, 12.f}) {
                const float e = lambda * spread;
                const float h = spread / 128;
                const float du = (BedPotentialAt(e + h, spread) - BedPotentialAt(e - h, spread)) / (2 * h);
                const float f = BedLoadAt(e, spread);
                expect(std::abs(du - f) <= 0.02f * f) << "dU vs load at lambda" << lambda << "spread" << spread << du << f;
            }
            // Deep below the tail both the load and the potential's slope vanish together.
            const float e = -12 * spread;
            const float h = spread / 128;
            const float du = (BedPotentialAt(e + h, spread) - BedPotentialAt(e - h, spread)) / (2 * h);
            expect(BedLoadAt(e, spread) == 0.f) << "load below the tail";
            expect(std::abs(du) <= 1e-6f * BedLoadAt(spread, spread)) << "potential slope below the tail" << du;
        }
        for (const float e : {1e-7f, 1e-6f, 1e-5f}) {
            const float h = e / 256;
            const float du = (BedPotentialAt(e + h, 0.f) - BedPotentialAt(e - h, 0.f)) / (2 * h);
            const float f = BedLoadAt(e, 0.f);
            expect(std::abs(du - f) <= 2e-3f * f) << "Hertz-outright dU vs load at" << e << du << f;
        }
    };

    "the bed's contact resonance is Persson's and does not depend on mass"_test = [] {
        // ContactPitch asserts a resting body oscillates on its contact at sqrt(g/u0) whatever it weighs.
        // That comes out of a stiffness linear in load, so the bed has to keep it.
        const auto machined = MachinedBed(4.905, 8.1e-3);
        const double u0 = MachinedConformalLength(machined.GapRoughness);
        std::array<double, 3> pitches{};
        const std::array masses{0.05, 0.5, 5.0};
        for (size_t i = 0; i < masses.size(); ++i) {
            pitches[i] = std::sqrt(BedStiffness(MachinedBed(masses[i] * 9.81, 8.1e-3).Bed) / masses[i]);
        }
        for (const double pitch : pitches) {
            expect(Near(pitch / pitches[0], 1.0, 0.15)) << "pitch over the lightest body's" << pitch / pitches[0];
            const double against_persson = pitch / std::sqrt(9.81 / u0);
            expect(against_persson > 0.7 && against_persson < 1.45) << "pitch over sqrt(g/u0)" << against_persson;
        }
    };

    return RunSuites();
}

// Andersson and Kropp's element springs: each element's own bearing curve, counted from its own crest, with the potential the curve's exact integral.
suite<"element springs"> _element_springs = [] {
    // A rough patch with a self-affine flavour: enough structure that elements differ from one another and crests sit well above their own means, which is where a closure would fail.
    struct Field {
        std::vector<float> Heights;
        std::vector<float> Gap;
        uint32_t Columns{0}, Rows{0};
    };
    const auto make_field = [](uint32_t columns, uint32_t rows, double sigma, double curvature) {
        Field f{.Heights = std::vector<float>(size_t(columns) * rows), .Gap = std::vector<float>(rows), .Columns = columns, .Rows = rows};
        std::mt19937 rng{12345};
        std::normal_distribution<float> draw{0.f, 1.f};
        // A short moving average across both axes gives neighbouring points correlation, so an element's population is clustered rather than independent draws.
        std::vector<float> white(f.Heights.size());
        for (auto &v : white) v = draw(rng);
        for (uint32_t c = 0; c < columns; ++c) {
            for (uint32_t r = 0; r < rows; ++r) {
                float sum = 0;
                for (int dc = -2; dc <= 2; ++dc) {
                    for (int dr = -2; dr <= 2; ++dr) {
                        const auto cc = uint32_t((int(c) + dc + int(columns)) % int(columns));
                        const auto rr = uint32_t((int(r) + dr + int(rows)) % int(rows));
                        sum += white[size_t(cc) * rows + rr];
                    }
                }
                f.Heights[size_t(c) * rows + r] = float(sigma) * sum / 5.f;
            }
        }
        // The body's transverse gap, as a curved contact folds it in.
        f.Gap = TransverseGap(rows, 1e-5, curvature);
        return f;
    };

    constexpr double EngagementMax = 3e-5;
    constexpr uint32_t ElementColumns = 16, Knots = 56;
    constexpr float ColumnSpacing = 1e-5f;
    const auto field = make_field(256, 64, 2e-6, 40.0);
    const auto springs = BuildElementSprings(
        field.Heights, field.Columns, field.Rows, field.Gap, ElementColumns, ColumnSpacing, CeramicInvModulus, {}, EngagementMax, Knots
    );

    // The direct per-asperity sum the springs stand in for: the field's own summits, each at the constant its own curvature gives it, flank samples belonging to their neighbouring summit.
    const auto folded = FoldGap(field.Heights, field.Columns, field.Rows, field.Gap);
    const auto summit = MarkFieldSummits(folded, field.Columns, field.Rows);
    // The depth a summit's own contact takes once the roughness inside that contact has taken the rest.
    // A contact 1.5 k sqrt(x) / E* wide holds (3/H) times its own separation open by the time it carries the load, so the two depths add (Pastewka et al. 2013 Eqs. B17 to B21).
    // Bisected here, where the springs invert a tabulated curve, so the two answers are independent.
    const auto hertz_depth = [](double depth, double stiffness, const SubCutoffRoughness &sub) {
        const auto total = [&](double x) {
            return x + (sub.Hurst > 0 ? 3 / sub.Hurst : 0.0) * SubCutoffSeparation(sub, 1.5 * stiffness * std::sqrt(x) * CeramicInvModulus);
        };
        double low = 0, high = depth;
        for (uint32_t i = 0; i < 200; ++i) {
            const double mid = 0.5 * (low + high);
            (total(mid) > depth ? high : low) = mid;
        }
        return 0.5 * (low + high);
    };
    const auto reference = [&](uint32_t element, float engagement, const SubCutoffRoughness &sub) {
        const float floor_height = springs.Crest[element] - engagement;
        double sum = 0;
        for (uint32_t c = 0; c < ElementColumns; ++c) {
            const size_t src = size_t(element * ElementColumns + c) * field.Rows;
            for (uint32_t r = 0; r < field.Rows; ++r) {
                if (!summit[src + r]) continue;
                if (const double d = double(folded[src + r]) - floor_height; d > 0) {
                    const auto curvature = CombinedCurvature(FieldSummitCurvature(folded, field.Columns, field.Rows, ColumnSpacing, element * ElementColumns + c, r), 0);
                    const double stiffness = ContactStiffness(CeramicInvModulus, curvature);
                    const double x = hertz_depth(d, stiffness, sub);
                    sum += stiffness * x * std::sqrt(x);
                }
            }
        }
        return sum;
    };

    "an element out of contact publishes exactly zero"_test = [&] {
        expect(springs.Count() == field.Columns / ElementColumns);
        for (uint32_t e = 0; e < springs.Count(); ++e) {
            expect(ReadElementSpring(springs, e, 0.f).Force == 0._f);
            expect(ReadElementSpring(springs, e, -1e-6f).Force == 0._f);
            expect(ReadElementSpring(springs, e, 0.f).Potential == 0._f);
        }
    };

    "the spring never softens with engagement"_test = [&] {
        for (uint32_t e = 0; e < springs.Count(); ++e) {
            float previous = 0;
            for (uint32_t i = 1; i <= 400; ++i) {
                const float force = ReadElementSpring(springs, e, float(i) * float(EngagementMax) / 400).Force;
                expect(force >= previous) << "element" << e << "step" << i;
                previous = force;
            }
        }
    };

    "force is the exact derivative of potential"_test = [&] {
        constexpr float h = 1e-9f;
        double worst = 0;
        for (uint32_t e = 0; e < springs.Count(); ++e) {
            for (uint32_t i = 1; i <= 60; ++i) {
                const float z = float(i) * float(EngagementMax) / 60;
                const auto read = ReadElementSpring(springs, e, z);
                if (read.Force <= 0) continue;
                const float slope = (ReadElementSpring(springs, e, z + h).Potential - ReadElementSpring(springs, e, z - h).Potential) / (2 * h);
                worst = std::max(worst, std::abs(double(slope) / read.Force - 1));
            }
        }
        expect(worst < 2e-3) << "worst relative deviation of dU/dz from F" << worst;
    };

    "the spring reproduces the bearing sum it was built from"_test = [&] {
        for (const double fraction : {0.05, 0.2, 0.5, 1.0}) {
            const auto z = float(fraction * EngagementMax);
            double worst = 0;
            for (uint32_t e = 0; e < springs.Count(); ++e) {
                const double want = reference(e, z, {});
                if (want <= 0) continue;
                worst = std::max(worst, std::abs(ReadElementSpring(springs, e, z).Force / want - 1));
            }
            expect(worst < 0.02) << "engagement fraction" << fraction << "worst relative error" << worst;
        }
    };

    "the roughness inside a contact takes its own share of every summit's depth"_test = [&] {
        // A band half a micron tall at the widest contact it is read at, which is the scale a machined finish carries below the wavelength it is described over.
        constexpr SubCutoffRoughness SubCutoff{.Amplitude = 6e-4, .Hurst = 0.7, .MaxWidth = 4e-5};
        const auto rough = BuildElementSprings(
            field.Heights, field.Columns, field.Rows, field.Gap, ElementColumns, ColumnSpacing, CeramicInvModulus, SubCutoff, EngagementMax, Knots
        );
        for (const double fraction : {0.05, 0.2, 0.5, 1.0}) {
            const auto z = float(fraction * EngagementMax);
            double worst = 0, softest = 1;
            for (uint32_t e = 0; e < rough.Count(); ++e) {
                const double want = reference(e, z, SubCutoff);
                if (want <= 0) continue;
                const double force = ReadElementSpring(rough, e, z).Force;
                worst = std::max(worst, std::abs(force / want - 1));
                softest = std::min(softest, force / ReadElementSpring(springs, e, z).Force);
            }
            expect(worst < 0.02) << "engagement fraction" << fraction << "worst relative error" << worst;
            // A compliance in series can only soften the stack, never stiffen it.
            expect(softest < 0.98_d) << "engagement fraction" << fraction << "softest ratio to the bare stack" << softest;
        }
    };

    "a crest stands where the element's own highest point does"_test = [&] {
        for (uint32_t e = 0; e < springs.Count(); ++e) {
            float highest = -1e30f;
            for (uint32_t c = 0; c < ElementColumns; ++c) {
                const size_t src = size_t(e * ElementColumns + c) * field.Rows;
                for (uint32_t r = 0; r < field.Rows; ++r) highest = std::max(highest, field.Heights[src + r] - field.Gap[r]);
            }
            expect(Near(double(springs.Crest[e]), double(highest), 1e-9));
        }
    };
};

// The patch the element springs are built from carries the same surface a track does.
suite<"roughness patch"> _roughness_patch = [] {
    constexpr float Correlation = 1e-3f, Slope = -3.f;
    constexpr float Cutoff = Correlation / PresetBandRatio, Spacing = Cutoff / SurfaceSamplesPerCutoff;

    "a patch is zero mean and unit root-mean-square"_test = [&] {
        const auto patch = SynthesizeRoughnessPatch(Correlation, Slope, Cutoff, Spacing, 512, 128, 0);
        expect(patch.Columns == 512_u && patch.Rows == 128_u);
        const double mean = std::accumulate(patch.Heights.begin(), patch.Heights.end(), 0.0) / double(patch.Heights.size());
        const double energy = std::accumulate(patch.Heights.begin(), patch.Heights.end(), 0.0, [](double a, float h) { return a + double(h) * double(h); });
        expect(Near(mean, 0.0, 1e-5)) << "mean" << mean;
        expect(Near(std::sqrt(energy / double(patch.Heights.size())), 1.0, 1e-5));
    };

    "the same arguments give the same patch"_test = [&] {
        const auto a = SynthesizeRoughnessPatch(Correlation, Slope, Cutoff, Spacing, 256, 64, 0);
        const auto b = SynthesizeRoughnessPatch(Correlation, Slope, Cutoff, Spacing, 256, 64, 0);
        expect(a.Heights == b.Heights);
    };

    "the thread count leaves every height byte the same"_test = [&] {
        // The column ranges jump the phase state past every draw before them, so a parallel fill makes the same draws as a serial pass and the transform is loop-split over unchanged rows.
        SetTransformThreads(1);
        const auto serial_patch = SynthesizeRoughnessPatch(Correlation, Slope, Cutoff, Spacing, 2048, 64, 7);
        std::vector<float> trace(serial_patch.Heights.begin(), serial_patch.Heights.begin() + 2048);
        const auto serial_profile = SynthesizeProfileField(trace, Spacing, 64, 3);
        SetTransformThreads(8);
        const auto parallel_patch = SynthesizeRoughnessPatch(Correlation, Slope, Cutoff, Spacing, 2048, 64, 7);
        const auto parallel_profile = SynthesizeProfileField(trace, Spacing, 64, 3);
        expect(serial_patch.Heights == parallel_patch.Heights);
        expect(serial_profile.Heights == parallel_profile.Heights);
    };

    "a cut across the patch carries the same slope as a cut along it"_test = [&] {
        // The spectrum is radial, so the surface has no preferred direction.
        const auto patch = SynthesizeRoughnessPatch(Correlation, Slope, Cutoff, Spacing, 512, 512, 0);
        const double along = PatchStepRms(patch, true), across = PatchStepRms(patch, false);
        expect(Near(across / along, 1.0, 0.05)) << "across over along" << across / along;
    };

    "a cut through the patch carries the track's own slope"_test = [&] {
        // The springs and the datum have to describe one surface, so the patch a spring is built from and the track a contact sweeps must agree on how steep that surface is.
        const auto patch = SynthesizeRoughnessPatch(Correlation, Slope, Cutoff, Spacing, 4096, 128, 0);
        const auto track = SynthesizeRoughness(Correlation, Slope, Cutoff, Spacing, 4096);
        const double patch_slope = PatchStepRms(patch, true);
        expect(Near(patch_slope / double(track.SlopeRms), 1.0, 0.15)) << "patch slope over track slope" << patch_slope / double(track.SlopeRms);
    };
};

// A measured trace reaches the same elements a parametric finish does, carrying its own heights along the sweep and the surface its own spectrum implies across it.
suite<"profile patch"> _profile_patch = [] {
    constexpr float Correlation = 1e-3f, Slope = -3.f;
    constexpr float Cutoff = Correlation / PresetBandRatio, Spacing = Cutoff / SurfaceSamplesPerCutoff;
    // A strip four correlation lengths across, which is what a ribbon samples.
    constexpr uint32_t Columns = 4096, Rows = 256;
    // A synthesized track stands in for a measurement, which is how the corpus authors one, and it enters the way one does: as heights with no parameters beside them.
    const auto trace = MakeProfileTrack(SynthesizeRoughness(Correlation, Slope, Cutoff, Spacing, Columns).Heights, Spacing);

    "a track reads back the band it was drawn from"_test = [&] {
        expect(Near(double(trace.Band.CorrelationLength) / double(Correlation), 1.0, 0.15)) << "correlation" << trace.Band.CorrelationLength;
        expect(Near(double(trace.Band.SpectralSlope), double(Slope), 0.15)) << "slope" << trace.Band.SpectralSlope;
    };

    "the trace is the patch's own first row"_test = [&] {
        const auto patch = SynthesizeProfileField(trace.Heights, Spacing, Rows, 0);
        expect(patch.Columns == Columns && patch.Rows == Rows);
        double worst = 0;
        for (uint32_t i = 0; i < Columns; ++i) worst = std::max(worst, std::abs(double(patch.Heights[size_t(i) * Rows]) - double(trace.Heights[i])));
        expect(Near(worst, 0.0, 1e-4)) << "worst height" << worst;
    };

    "the same arguments give the same patch"_test = [&] {
        const auto a = SynthesizeProfileField(trace.Heights, Spacing, Rows, 0);
        const auto b = SynthesizeProfileField(trace.Heights, Spacing, Rows, 0);
        const auto other = SynthesizeProfileField(trace.Heights, Spacing, Rows, 1);
        expect(a.Heights == b.Heights);
        expect(a.Heights != other.Heights);
    };

    "the patch carries the trace's own height and slope"_test = [&] {
        const auto patch = SynthesizeProfileField(trace.Heights, Spacing, Rows, 0);
        const double mean = std::accumulate(patch.Heights.begin(), patch.Heights.end(), 0.0) / double(patch.Heights.size());
        const double energy = std::accumulate(patch.Heights.begin(), patch.Heights.end(), 0.0, [](double a, float h) { return a + double(h) * double(h); });
        expect(Near(mean, 0.0, 1e-5)) << "mean" << mean;
        expect(Near(std::sqrt(energy / double(patch.Heights.size())), 1.0, 0.1)) << "root-mean-square";
        // A cut in either direction is a trace of the same surface.
        const double along = PatchStepRms(patch, true), across = PatchStepRms(patch, false);
        expect(Near(across / along, 1.0, 0.05)) << "across over along" << across / along;
        expect(Near(along / double(trace.SlopeRms), 1.0, 0.05)) << "patch slope over trace slope" << along / double(trace.SlopeRms);
    };
};

// The whole stack of springs a contact bears on, which is what the render path reads.
suite<"crest envelope"> _crest_envelope = [] {
    // The envelope is the highest crest a body's own gap lets it reach over its window, compared here against a direct walk of that window at every element.
    // The gaps run from flat, where the envelope is a running maximum, to steep, where every element keeps its own crest.
    std::mt19937 rng{20260814};
    std::normal_distribution<float> draw{0.f, 1e-4f};
    std::vector<float> crest(512);
    for (auto &v : crest) v = draw(rng);
    constexpr float Width = 2.5e-6f;
    const auto walked = [&](double curvature, uint32_t reach) {
        const auto n = int64_t(crest.size());
        std::vector<float> out(crest.size());
        for (int64_t e = 0; e < n; ++e) {
            double top = -std::numeric_limits<double>::max();
            for (int64_t o = -int64_t(reach); o <= int64_t(reach); ++o) {
                const double offset = double(o) * double(Width);
                top = std::max(top, double(crest[size_t(((e + o) % n + n) % n)]) - 0.5 * curvature * offset * offset);
            }
            out[size_t(e)] = float(top);
        }
        return out;
    };
    for (const double curvature : {0.0, 1e3, 1e6, 1e9, 1e12}) {
        for (const uint32_t reach : {1u, 7u, 64u, 255u}) {
            const auto want = walked(curvature, reach);
            const auto got = CrestEnvelope(crest, Width, curvature, reach);
            float worst = 0;
            for (size_t e = 0; e < want.size(); ++e) worst = std::max(worst, std::abs(got[e] - want[e]));
            expect(worst == 0.f);
        }
    }
};

suite<"crest block skip"> _crest_block_skip = [] {
    // The walks skip runs of elements whose tallest crest cannot reach the body.
    // That is a way of reaching the same sums faster, never a different answer, so the datum it produces must equal the one a walk of every element produces.
    // Emptying the block table forces that slow walk.
    constexpr float Correlation = 1e-3f, Slope = -3.f;
    constexpr float Cutoff = Correlation / PresetBandRatio, Spacing = Cutoff / SurfaceSamplesPerCutoff;
    constexpr uint32_t Columns = 4096, Rows = 64, ElementColumns = 16, Knots = 56;
    const auto heights = RoughHeights(Correlation, Slope, Cutoff, Spacing, Columns, Rows, 3, 1e-4f);
    const std::vector<float> gap(Rows, 0.f);
    auto springs = BuildElementSprings(heights, Columns, Rows, gap, ElementColumns, Spacing, 9.1e-12, {}, 6e-4, Knots);
    // A flat body and a curved one, since the gap parabola is what bounds a skipped block's reach.
    for (const double curvature : {0.0, 400.0}) {
        for (const uint32_t reach : {8u, 40u}) {
            auto walked = springs;
            walked.BlockCrest.clear();
            for (const double load : {0.5, 20.0}) {
                const auto fast = BearingDatum(springs, curvature, reach, load);
                const auto slow = BearingDatum(walked, curvature, reach, load);
                expect(fast == slow);
            }
            const auto envelope = ElementEnvelope(springs, curvature, reach);
            for (const float engagement : {1e-5f, 1e-4f}) {
                for (const double position : {0.5, 100.25, 4000.5}) {
                    const auto fast = SpringFlankMoments(springs, envelope, position, engagement, curvature, reach);
                    const auto slow = SpringFlankMoments(walked, envelope, position, engagement, curvature, reach);
                    expect(fast.Modulus == slow.Modulus);
                    expect(fast.Stiffness == slow.Stiffness);
                    expect(fast.Bearing == slow.Bearing);
                    expect(fast.Width == slow.Width);
                }
            }
        }
    }

    // A count that is not a multiple of the block size leaves a last block running past the end, so a skip there spans elements from the array's head too and the block's crest must bound them.
    // A towering crest in that head is what such a skip would wrongly drop.
    {
        constexpr uint32_t WrapColumns = 90 * ElementColumns;
        const auto wrap_heights = RoughHeights(Correlation, Slope, Cutoff, Spacing, WrapColumns, Rows, 5, 1e-4f);
        auto tall = BuildElementSprings(wrap_heights, WrapColumns, Rows, gap, ElementColumns, Spacing, 9.1e-12, {}, 6e-4, Knots);
        tall.Crest[5] += 0.01f;
        RefreshCrestBlocks(tall);
        auto walked = tall;
        walked.BlockCrest.clear();
        constexpr uint32_t WrapReach = 40;
        const auto envelope = ElementEnvelope(tall, 0.0, WrapReach);
        for (const double position : {87.0, 87.5, 87.9}) {
            for (const float engagement : {1e-4f, 5e-4f}) {
                const auto fast = SpringFlankMoments(tall, envelope, position, engagement, 0.0, WrapReach);
                const auto slow = SpringFlankMoments(walked, envelope, position, engagement, 0.0, WrapReach);
                expect(slow.Bearing > 0.0);
                expect(fast.Stiffness == slow.Stiffness);
                expect(fast.Bearing == slow.Bearing);
            }
        }
        for (const double load : {0.5, 20.0}) {
            const auto fast = BearingDatum(tall, 0.0, WrapReach, load);
            const auto slow = BearingDatum(walked, 0.0, WrapReach, load);
            expect(fast == slow);
        }
    }
};

suite<"sweep mode drives"> _sweep_mode_drives = [] {
    // The sweep accumulation walks only the bearing forces, which must reach the same table as a walk of every window slot, term for term.
    constexpr uint32_t Count = 256, Modes = 3;
    // Every seventeenth element bears, no element bears, and every element bears.
    for (const uint32_t bearing_stride : {17u, 0u, 1u}) {
        // A reach of half the count spans a window one slot longer than the array itself.
        for (const uint32_t reach : {0u, 5u, 100u, Count / 2}) {
            const uint32_t window = 2 * reach + 1;
            std::vector<float> field(Count, 0.f);
            for (uint32_t e = 0; bearing_stride > 0 && e < Count; e += bearing_stride) field[e] = 1e-3f + float(e % 7) * 2e-4f;
            std::vector<float> phi(size_t(window) * Modes);
            for (size_t i = 0; i < phi.size(); ++i) phi[i] = float(std::sin(0.37 * double(i)));
            std::vector<float> slow(size_t(Modes) * Count, 0.f);
            for (uint32_t x = 0; x < Count; ++x) {
                for (uint32_t w = 0; w < window; ++w) {
                    const float f = field[(x + w + Count - reach) % Count];
                    if (f <= 0) continue;
                    for (uint32_t k = 0; k < Modes; ++k) slow[size_t(k) * Count + x] += f * phi[size_t(w) * Modes + k];
                }
            }
            std::vector<float> fast(size_t(Modes) * Count, 0.f);
            SweepModeDrives(field, phi, Modes, reach, fast);
            expect(fast == slow);
        }
    }
};

suite<"element summit gather"> _element_summit_gather = [] {
    // The gather walks a three-column window of the folded field, which must produce exactly the populations the materialized construction does: fold the whole field, mark its summits, and read them element by element.
    constexpr float Correlation = 1e-3f, Slope = -3.f;
    constexpr float Cutoff = Correlation / PresetBandRatio, Spacing = Cutoff / SurfaceSamplesPerCutoff;
    // A column count that is not a multiple of the element width leaves trailing columns that neighbour the gathered ones without being gathered themselves.
    constexpr uint32_t Columns = 1000, Rows = 48, ElementColumns = 16;
    const auto heights = RoughHeights(Correlation, Slope, Cutoff, Spacing, Columns, Rows, 11, 1e-4f);
    const auto gap = TransverseGap(Rows, Spacing, 400.0);
    std::vector<float> lift(Columns / ElementColumns);
    for (size_t e = 0; e < lift.size(); ++e) lift[e] = float(e % 5) * 1e-6f;
    ElementSummits fused;
    GatherElementSummits(fused, heights, Columns, Rows, gap, lift, ElementColumns, Spacing, CeramicInvModulus);

    const auto folded = FoldGap(heights, Columns, Rows, gap);
    const auto summit = MarkFieldSummits(folded, Columns, Rows);
    ElementSummits walked;
    const uint32_t count = Columns / ElementColumns;
    walked.Crest.resize(count, -std::numeric_limits<float>::max());
    for (uint32_t e = 0; e < count; ++e) {
        const float element_lift = e < lift.size() ? lift[e] : 0.f;
        float crest = -std::numeric_limits<float>::max();
        for (uint32_t c = 0; c < ElementColumns; ++c) {
            const uint32_t col = e * ElementColumns + c;
            const size_t src = size_t(col) * Rows;
            for (uint32_t r = 0; r < Rows; ++r) {
                crest = std::max(crest, folded[src + r]);
                if (summit[src + r]) {
                    const auto curvature = CombinedCurvature(FieldSummitCurvature(folded, Columns, Rows, Spacing, col, r), 0);
                    walked.Element.push_back(e);
                    walked.Height.push_back(folded[src + r] + element_lift);
                    walked.Stiffness.push_back(float(ContactStiffness(CeramicInvModulus, curvature)));
                }
            }
        }
        walked.Crest[e] = std::max(walked.Crest[e], crest + element_lift);
    }
    expect(fused.Element == walked.Element);
    expect(fused.Height == walked.Height);
    expect(fused.Stiffness == walked.Stiffness);
    expect(fused.Crest == walked.Crest);
};

suite<"contact springs"> _contact_springs = [] {
    // The cast finish of the corpus scenes, a half-kilogram sphere's curvature, and steel asperities.
    constexpr float Correlation = 1e-3f, Slope = -3.f;
    constexpr float Cutoff = Correlation / PresetBandRatio, Spacing = Cutoff / SurfaceSamplesPerCutoff;
    constexpr float Roughness = 1e-4f;
    constexpr double Curvature = 1 / 0.0248, Load = 4.905;
    constexpr uint32_t Columns = 8192, Rows = 128, ElementColumns = 16, Knots = 56;

    const auto heights = RoughHeights(Correlation, Slope, Cutoff, Spacing, Columns, Rows, 0, Roughness);
    const auto gap = TransverseGap(Rows, Spacing, Curvature);
    // The transverse extent the patch spans bounds how deep the springs can be trusted: past it the body reaches off the patch, so the table stops where its own geometry does.
    const double half_width = 0.5 * double(Rows) * double(Spacing);
    const double engagement_max = 0.9 * 0.5 * Curvature * half_width * half_width;
    const auto springs = BuildElementSprings(
        heights, Columns, Rows, gap, ElementColumns, Spacing, CeramicInvModulus, {}, engagement_max, Knots
    );
    const uint32_t reach = ElementReach(springs, Curvature);
    const auto envelope = ElementEnvelope(springs, Curvature, reach);

    "the knot table finds exactly the bisected knot"_test = [&] {
        auto bisect = springs;
        bisect.KnotByExponent.clear();
        expect(!springs.KnotByExponent.empty());
        for (uint32_t e = 0; e < springs.Count(); e += 17) {
            for (uint32_t j = 1; j < springs.Knots; ++j) {
                const float knot = springs.Engagement[j];
                for (const float engagement :
                     {knot, std::nextafter(knot, 0.f), std::nextafter(knot, 1.f), 0.5f * (knot + springs.Engagement[j - 1])}) {
                    expect(ReadElementSpring(springs, e, engagement).Force == ReadElementSpring(bisect, e, engagement).Force);
                    expect(ReadElementSpring(springs, e, engagement).Potential == ReadElementSpring(bisect, e, engagement).Potential);
                }
            }
        }
    };

    "the reach spans more than the engagement alone would suggest"_test = [&] {
        // A crest is a maximum, so first touch is routinely many elements off the contact's centre.
        expect(reach > 1_u) << "reach" << reach;
        expect(reach < springs.Count() / 2 + 1);
    };

    "a body above every crest it can reach publishes exactly zero"_test = [&] {
        for (uint32_t e = 0; e < springs.Count(); e += 17) {
            expect(ReadContactSprings(springs, envelope, double(e) + 0.5, 0.f, Curvature, reach).Force == 0._f);
            expect(ReadContactSprings(springs, envelope, double(e) + 0.5, -1e-6f, Curvature, reach).Force == 0._f);
        }
    };

    "the stack carries the load it was anchored at"_test = [&] {
        const float anchor = SolveSpringEngagement(springs, envelope, Load, Curvature, reach);
        expect(anchor > 0._f) << "anchor" << anchor;
        double sum = 0;
        for (uint32_t e = 0; e < springs.Count(); ++e) sum += ReadContactSprings(springs, envelope, double(e) + 0.5, anchor, Curvature, reach).Force;
        expect(Near(sum / double(springs.Count()) / Load, 1.0, 1e-3)) << "mean force over load" << sum / double(springs.Count()) / Load;
    };

    "the anchor rises with the load"_test = [&] {
        float previous = 0;
        for (const double load : {0.5, 5.0, 50.0}) {
            const float anchor = SolveSpringEngagement(springs, envelope, load, Curvature, reach);
            expect(anchor > previous) << "load" << load << "anchor" << anchor;
            previous = anchor;
        }
    };

    "the stack's force is the exact derivative of the stack's potential"_test = [&] {
        // The quadratised exchange differentiates the whole contact's stored energy, not one element's, so the property has to survive the sum over elements and their gap offsets.
        const float anchor = SolveSpringEngagement(springs, envelope, Load, Curvature, reach);
        // The step keeps the finite difference above float cancellation: the noise scales as eps * U / (2 h F), and U / F grows with engagement, so the deep probes set the floor.
        constexpr float h = 1e-8f;
        for (const float scale : {0.5f, 1.f, 2.f, 6.f}) {
            double worst = 0;
            for (uint32_t e = 0; e < springs.Count(); e += 7) {
                const float a = anchor * scale;
                const auto read = ReadContactSprings(springs, envelope, double(e) + 0.5, a, Curvature, reach);
                if (read.Force <= 0) continue;
                const float slope = (ReadContactSprings(springs, envelope, double(e) + 0.5, a + h, Curvature, reach).Potential -
                                     ReadContactSprings(springs, envelope, double(e) + 0.5, a - h, Curvature, reach).Potential) /
                    (2 * h);
                worst = std::max(worst, std::abs(double(slope) / read.Force - 1));
            }
            expect(worst < 5e-3) << "scale" << scale << "worst relative deviation of dU/da from F" << worst;
        }
    };
};
