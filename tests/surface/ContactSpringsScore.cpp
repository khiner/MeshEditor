// Scores the element-spring contact path against the exact per-summit solve: a 0.5 kg steel sphere slides over a 100 um, 1 mm correlation surface at three speeds and lands at three approach rates, and every observable is compared against the direct per-asperity sum over the same synthesized field.
// The springs, envelope, reach, stack read, and field are all the model's own, so a divergence here is a defect of the model rather than of the harness.
// Reference anchors over the same field: engine flight fraction 0.700 against exact 0.747 at 0.7 m/s, band speed exponent +1.43 against +1.45, landing loads and durations within about 2 percent.

#include "audio/surface/SurfaceModel.h"

#include "audio/ContactSurface.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <vector>

namespace {
// In-place iterative radix-2 FFT.
// A direct Goertzel per bin is O(N^2), and the scores only need the power spectrum, so the whole transform is cheaper than the bins read off it.
// `buf.size()` is a power of two.
void Fft(std::vector<std::complex<double>> &buf) {
    const auto n = buf.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(buf[i], buf[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2 * std::numbers::pi / double(len);
        const std::complex<double> wl{std::cos(ang), std::sin(ang)};
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w{1, 0};
            for (size_t k = 0; k < len / 2; ++k, w *= wl) {
                const auto u = buf[i + k], t = w * buf[i + k + len / 2];
                buf[i + k] = u + t;
                buf[i + k + len / 2] = u - t;
            }
        }
    }
}

constexpr double Mass = 0.5, Grav = 9.81, Load = Mass * Grav;
constexpr double Correlation = 1e-3;
constexpr float Cutoff = float(Correlation) / PresetBandRatio;
constexpr float Spacing = Cutoff / SurfaceSamplesPerCutoff;
// The pair's roughness, and the compliance its asperities bear through.
// Each summit takes the Hertz constant its own curvature gives it, so both arms read the field rather than a shared spot.
const double SigmaPair = 100e-6 * std::sqrt(1 + (2e-6 / 100e-6) * (2e-6 / 100e-6));
constexpr double InvModulus = 9.1e-12; // A steel pair's 1/E*
constexpr double CellStiffness = 1.51458e8; // The cell's resting stiffness, which the rig damping is set about
const double SphereRadius = std::cbrt(3 * Mass / (4 * std::numbers::pi * 7850.0));
constexpr uint32_t Nx = 8192, Ny = 512, ElementColumns = 16, Knots = 56;

// The scaled field without the transverse gap, so the exact arm applies both gaps itself.
std::vector<float> Field;
std::vector<float> GapY; // The body's transverse gap at each row, m.
std::vector<uint8_t> Summit; // Summits of the transverse-folded field: the asperities both arms bear on.
std::vector<float> Folded; // That field, which each summit's own curvature is read from.
std::vector<float> ColTop; // Per column: max over rows of height less transverse gap.
double ColTopMax = 0;

ElementSprings Springs;
std::vector<float> Envelope;
uint32_t Reach = 0;
double Curvature = 0;

// The direct per-summit sum at body surface height y under the sphere centred at xc: truth for both arms.
double ExactForce(double xc, double y) {
    const double dmax = ColTopMax - y;
    if (dmax <= 0) return 0;
    const auto half = int64_t(std::sqrt(2 * SphereRadius * dmax) / Spacing) + 2;
    const auto i0 = int64_t(std::llround(xc / Spacing));
    double sum = 0;
    for (int64_t c = i0 - half; c <= i0 + half; ++c) {
        const double dx = double(c) * Spacing - xc;
        const double lift = dx * dx / (2 * SphereRadius);
        const auto col = uint64_t(((c % int64_t(Nx)) + int64_t(Nx)) % int64_t(Nx));
        if (double(ColTop[col]) - y - lift <= 0) continue;
        const auto *h = &Field[col * Ny];
        const auto *s = &Summit[col * Ny];
        for (uint32_t r = 0; r < Ny; ++r) {
            if (!s[r]) continue;
            const double d = double(h[r]) - double(GapY[r]) - y - lift;
            if (d > 0) {
                const auto curvature = CombinedCurvature(FieldSummitCurvature(Folded, Nx, Ny, Spacing, col, r), 0);
                sum += ContactStiffness(InvModulus, curvature) * d * std::sqrt(d);
            }
        }
    }
    return sum;
}

// The engine arm: the stack read at the position under the sphere, engagement measured from the envelope there, exactly as StepVoiceQuad reads it.
double EngineForce(double xc, double y) {
    const double pos = std::fmod(xc / (double(Spacing) * ElementColumns), double(Springs.Count()));
    return ReadContactSprings(Springs, Envelope, pos, float(double(EnvelopeAt(Envelope, pos)) - y), Curvature, Reach).Force;
}

double ForceOf(bool engine, double xc, double y) { return engine ? EngineForce(xc, y) : ExactForce(xc, y); }

// The body height carrying the load at rest under either arm.
double BisectRest(bool engine, double xc) {
    double lo = -2e-4, hi = 8e-4;
    for (int i = 0; i < 90; ++i) {
        const double mid = 0.5 * (lo + hi);
        (ForceOf(engine, xc, mid) > Load ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

struct RideStats {
    double MeanOverLoad, SigmaOverMean, Free, Band;
};

RideStats Ride(bool engine, double v, double seconds, double xc0) {
    constexpr double sr = 384'000, dt = 1 / sr;
    // The rig damping: 5 percent of critical about the cell's resting stiffness.
    const double d_rest = std::pow(Load / CellStiffness, 2.0 / 3);
    const double c_damp = 2 * 0.05 * std::sqrt(1.5 * CellStiffness * std::sqrt(d_rest) * Mass);
    double y = BisectRest(engine, xc0), vel = 0, xc = xc0;
    const auto n = size_t(seconds * sr);
    std::vector<double> f_out(n);
    for (size_t i = 0; i < n; ++i) {
        const double f = ForceOf(engine, xc, y);
        f_out[i] = f;
        vel += ((f - c_damp * vel) / Mass - Grav) * dt;
        y += vel * dt;
        xc += v * dt;
    }
    // Skip the first quarter, then mean, spread, flight fraction, and 200 Hz to 16 kHz band power.
    const auto s = std::vector<double>(f_out.begin() + n / 4, f_out.end());
    double mu = 0;
    for (const double f : s) mu += f;
    mu /= double(s.size());
    double var = 0, free = 0;
    for (const double f : s) {
        var += (f - mu) * (f - mu);
        if (f <= 0) free += 1;
    }
    const auto m = s.size();
    double band = 0;
    {
        // Hann-windowed DFT power over the band, coarse in frequency: the band sum over a full rfft.
        // Radix-2 needs a power of two, so zero-pad up.
        size_t pow2 = 1;
        while (pow2 < m) pow2 <<= 1;
        std::vector<std::complex<double>> buf(pow2);
        for (size_t i = 0; i < m; ++i) {
            const double w = 0.5 - 0.5 * std::cos(2 * std::numbers::pi * double(i) / double(m - 1));
            buf[i] = (s[i] - mu) * w;
        }
        Fft(buf);
        for (size_t k = 0; k <= pow2 / 2; ++k) {
            const double fr = double(k) * sr / double(pow2);
            if (fr > 200 && fr < 16000) band += std::norm(buf[k]);
        }
        band /= double(m);
    }
    return {mu / Load, std::sqrt(var / double(m)) / std::max(mu, 1e-12), free / double(m), band};
}

struct LandStats {
    double PeakOverLoad, DurationMs, CornerHz;
};

LandStats Land(bool engine, double v0, double xc0) {
    constexpr double sr = 6'144'000, dt = 1 / sr;
    // First touch: the envelope under the sphere at xc0, which both arms share by construction.
    double y;
    if (engine) {
        y = EnvelopeAt(Envelope, std::fmod(xc0 / (double(Spacing) * ElementColumns), double(Springs.Count())));
    } else {
        const auto half = int64_t(std::sqrt(2 * SphereRadius * (1e-3 + 4 * SigmaPair)) / Spacing) + 2;
        const auto i0 = int64_t(std::llround(xc0 / Spacing));
        double top = -1e9;
        for (int64_t c = i0 - half; c <= i0 + half; ++c) {
            const double dx = double(c) * Spacing - xc0;
            const double lift = dx * dx / (2 * SphereRadius);
            const auto col = uint64_t(((c % int64_t(Nx)) + int64_t(Nx)) % int64_t(Nx));
            top = std::max(top, double(ColTop[col]) - lift);
        }
        y = top;
    }
    double vel = -v0, xc = xc0;
    std::vector<double> f_out;
    while (true) {
        const double f = ForceOf(engine, xc, y);
        f_out.push_back(f);
        vel += (f / Mass - Grav) * dt;
        y += vel * dt;
        xc += 0.1 * dt;
        if (f <= 0 && vel > 0 && f_out.size() > 8) break;
        if (f_out.size() > 120'000) break;
    }
    double peak = 0;
    for (const double f : f_out) peak = std::max(peak, f);
    size_t first = f_out.size(), last = 0;
    for (size_t i = 0; i < f_out.size(); ++i) {
        if (f_out[i] > 0.01 * peak) {
            first = std::min(first, i);
            last = i;
        }
    }
    const double ms = first <= last ? double(last - first + 1) * dt * 1e3 : 0.0;
    // The 90 percent energy corner over a fixed 128k transform.
    constexpr size_t N = 1 << 17;
    std::vector<double> spec(N / 2 + 1, 0.0);
    {
        std::vector<std::complex<double>> buf(N);
        for (size_t i = 0; i < std::min(f_out.size(), N); ++i) buf[i] = f_out[i];
        Fft(buf);
        for (size_t k = 0; k <= N / 2; ++k) spec[k] = std::norm(buf[k]);
    }
    double total = 0;
    for (const double p : spec) total += p;
    double cum = 0, corner = 0;
    for (size_t k = 0; k <= N / 2; ++k) {
        cum += spec[k];
        if (cum >= 0.9 * total) {
            corner = double(k) * sr / double(N);
            break;
        }
    }
    return {peak / Load, ms, corner};
}
} // namespace

int main() {
    // The engine's own patch, cuts one power steeper than the radial spectrum by construction.
    // The field is radial q^-3 (Hurst 0.5), whose cuts carry 1D slope -2.
    auto patch = SynthesizeRoughnessPatch(float(Correlation), -2.f, Cutoff, Spacing, Nx, Ny, 0);
    Field.resize(patch.Heights.size());
    for (size_t i = 0; i < Field.size(); ++i) Field[i] = float(SigmaPair) * patch.Heights[i];
    GapY.resize(Ny);
    for (uint32_t r = 0; r < Ny; ++r) {
        const double yy = (double(r) - 0.5 * (Ny - 1)) * Spacing;
        GapY[r] = float(yy * yy / (2 * SphereRadius));
    }
    ColTop.assign(Nx, -1e9f);
    for (uint32_t c = 0; c < Nx; ++c) {
        for (uint32_t r = 0; r < Ny; ++r) ColTop[c] = std::max(ColTop[c], Field[size_t(c) * Ny + r] - GapY[r]);
        ColTopMax = std::max(ColTopMax, double(ColTop[c]));
    }

    Curvature = 1 / SphereRadius;
    const double zeta_max = 0.9 * std::pow(0.5 * Ny * Spacing, 2) / (2 * SphereRadius);
    {
        Folded.resize(Field.size());
        for (uint32_t c = 0; c < Nx; ++c) {
            for (uint32_t r = 0; r < Ny; ++r) Folded[size_t(c) * Ny + r] = Field[size_t(c) * Ny + r] - GapY[r];
        }
        Summit = MarkFieldSummits(Folded, Nx, Ny);
    }
    Springs = BuildElementSprings(Field, Nx, Ny, GapY, ElementColumns, Spacing, InvModulus, {}, zeta_max, Knots);
    Reach = ElementReach(Springs, Curvature);
    // The datum the engine measures engagement from: first touch, which is a maximum over the reach, or the height the elements there actually carry the load at.
    // The exact arm below is truth for both, since its own resting height IS that load balance taken summit by summit.
    const bool first_touch = std::getenv("MUTE_BEARING_DATUM") != nullptr;
    Envelope = first_touch ? ElementEnvelope(Springs, Curvature, Reach) : BearingDatum(Springs, Curvature, Reach, Load);
    std::printf(
        "field %ux%u at %.1f um, sigma %.1f um, %u elements of %.0f um, reach +/-%u elements, zeta_max %.1f um, datum %s\n",
        Nx, Ny, Spacing * 1e6, SigmaPair * 1e6, Springs.Count(), double(Springs.Width) * 1e6, Reach, zeta_max * 1e6,
        first_touch ? "first touch" : "bearing"
    );

    // The resting anchor both ways: the exact arm's bisected height against the engine's, plus the spread of the per-position anchors, which is what the pooled anchor table averages over.
    {
        double dsum = 0, dmax = 0;
        for (int i = 0; i < 6; ++i) {
            const double xc = (i + 0.5) * (Nx * Spacing / 6);
            const double ye = BisectRest(false, xc), yb = BisectRest(true, xc);
            dsum += yb - ye;
            dmax = std::max(dmax, std::abs(yb - ye));
        }
        std::printf("resting height, engine less exact over 6 positions: mean %+.3f um, worst %.3f um\n", dsum / 6 * 1e6, dmax * 1e6);
    }

    const double speeds[] = {0.02, 0.1, 0.7};
    const double durations[] = {0.5, 0.3, 0.15};
    std::printf("\nRIDING: arm, v, mean/load, sigma/mean, free, band power\n");
    double band_log[2][3];
    for (const bool engine : {false, true}) {
        for (int i = 0; i < 3; ++i) {
            const auto st = Ride(engine, speeds[i], durations[i], 0.030);
            band_log[engine][i] = std::log(std::max(st.Band, 1e-30));
            std::printf("  %6s v=%4.2f  %6.2f %8.3f %6.3f  %10.3e\n", engine ? "engine" : "exact", speeds[i], st.MeanOverLoad, st.SigmaOverMean, st.Free, st.Band);
        }
    }
    for (const bool engine : {false, true}) {
        // Least-squares slope of log band power on log speed, halved to the amplitude exponent.
        const double lx[] = {std::log(speeds[0]), std::log(speeds[1]), std::log(speeds[2])};
        const double mx = (lx[0] + lx[1] + lx[2]) / 3, my = (band_log[engine][0] + band_log[engine][1] + band_log[engine][2]) / 3;
        double num = 0, den = 0;
        for (int i = 0; i < 3; ++i) {
            num += (lx[i] - mx) * (band_log[engine][i] - my);
            den += (lx[i] - mx) * (lx[i] - mx);
        }
        std::printf("  band speed exponent %s m = %+.2f\n", engine ? "engine" : "exact", num / den / 2);
    }

    std::printf("\nLANDINGS (mean of 6 positions): peak loads, duration ms, 90%% corner Hz\n");
    for (const double v0 : {0.03, 0.1, 0.3}) {
        for (const bool engine : {false, true}) {
            double pk = 0, ms = 0, cn = 0;
            for (int i = 0; i < 6; ++i) {
                const auto st = Land(engine, v0, (i + 0.5) * (Nx * Spacing / 6));
                pk += st.PeakOverLoad;
                ms += st.DurationMs;
                cn += st.CornerHz;
            }
            std::printf("  v0=%3.0f mm/s  %6s loads %7.1f  ms %6.3f  corner %6.0f\n", v0 * 1e3, engine ? "engine" : "exact", pk / 6, ms / 6, cn / 6);
        }
    }
    return 0;
}
