#include "SurfaceModel.h"

#include "audio/Fft.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <numeric>
#include <span>
#include <thread>

/***** Roughness tracks *****/

namespace {
uint64_t SplitMix64(uint64_t &state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// The state `draws` SplitMix64 calls ahead, the state stepping by a constant per draw.
uint64_t SkipDraws(uint64_t state, uint64_t draws) { return state + draws * 0x9e3779b97f4a7c15ull; }
// One draw of the stream, mapped to a uniform in [0, 1).
float UniformDraw(uint64_t &state) { return float(SplitMix64(state) >> 40) * 0x1p-24f; }
float NextPhase(uint64_t &state) { return UniformDraw(state) * 2 * std::numbers::pi_v<float>; }

// Fill the running integral and rescale the heights to zero mean and unit root-mean-square, returning the root-mean-square divided out.
float Finish(RoughnessTrack &t) {
    const auto n = t.Heights.size();
    const auto mean = float(std::accumulate(t.Heights.begin(), t.Heights.end(), 0.0) / double(n));
    for (float &h : t.Heights) h -= mean;
    const auto energy = std::accumulate(t.Heights.begin(), t.Heights.end(), 0.0, [](double a, float h) { return a + double(h) * double(h); });
    const float rms = float(std::sqrt(energy / double(n)));
    if (rms > 0) {
        for (float &h : t.Heights) h /= rms;
    }
    t.HeightMax = n > 0 ? *std::ranges::max_element(t.Heights) : 0.f;
    t.Sum.resize(n + 1);
    t.Sum[0] = 0;
    std::partial_sum(t.Heights.begin(), t.Heights.end(), t.Sum.begin() + 1);
    if (n > 1) {
        // Exclude the cyclic boundary because measured profile endpoints are not adjacent samples.
        double slope_energy = 0;
        for (size_t i = 1; i < n; ++i) {
            const double d = double(t.Heights[i]) - double(t.Heights[i - 1]);
            slope_energy += d * d;
        }
        t.SlopeRms = float(std::sqrt(slope_energy / double(n - 1)));
    }
    if (n > 2) {
        double curvature_energy = 0;
        for (size_t i = 1; i + 1 < n; ++i) {
            const double d = double(t.Heights[i + 1]) - 2 * double(t.Heights[i]) + double(t.Heights[i - 1]);
            curvature_energy += d * d;
        }
        t.CurvatureRms = float(std::sqrt(curvature_energy / double(n - 2)));
    }
    // Compute cyclic boxcar means from the running integral.
    for (uint32_t i = 0; i < TrackWidthCount; ++i) {
        const size_t width = size_t{1} << i;
        if (width >= n) {
            t.WindowRms[i] = 0;
            continue;
        }
        double energy = 0;
        for (size_t j = 0; j < n; ++j) {
            const size_t end = j + width;
            const double sum = end <= n ? double(t.Sum[end]) - double(t.Sum[j]) : double(t.Sum[n]) - double(t.Sum[j]) + double(t.Sum[end - n]);
            const double mean = sum / double(width);
            energy += mean * mean;
        }
        t.WindowRms[i] = float(std::sqrt(energy / double(n)));
    }
    float widest = 0;
    for (uint32_t i = 0; i < TrackWidthCount; ++i) {
        const size_t width = size_t{1} << i;
        if (2 * width >= n) {
            t.SlopeWindowRms[i] = widest;
            continue;
        }
        double energy = 0;
        for (size_t j = 0; j + 2 * width <= n; ++j) {
            const double d = double(t.Sum[j + 2 * width]) - 2 * double(t.Sum[j + width]) + double(t.Sum[j]);
            energy += d * d;
        }
        const double span = double(width) * double(width);
        widest = float(std::sqrt(energy / double(n - 2 * width + 1)) / span);
        t.SlopeWindowRms[i] = widest;
    }
    return rms;
}

// Returns fixed fractional-octave cells with each center bin and mean power.
// One realization scatters exponentially about its expected spectrum bin by bin, so the low end stays as measured and the high end averages down.
struct SpectrumCells {
    std::vector<double> At, Power; // The bin a cell is centred on, and the mean power over it.
};
SpectrumCells CellSpectrum(std::span<const double> power, double per_octave) {
    const auto bins = uint32_t(power.size());
    SpectrumCells cells;
    for (uint32_t lo = 1; lo + 1 < bins;) {
        const auto hi = std::min(bins - 1, std::max(lo + 1, uint32_t(std::llround(double(lo) * std::exp2(1 / per_octave)))));
        double total = 0;
        for (uint32_t i = lo; i < hi; ++i) total += power[i];
        cells.At.push_back(0.5 * double(lo + hi));
        cells.Power.push_back(total / double(hi - lo));
        lo = hi;
    }
    return cells;
}

// The model every surface here is described by, fitted to a measured spectrum: flat below a corner and a power law above it.
// The corner is scanned over the cells and the two remaining terms solve in closed form at each.
// The corner is returned in bins.
struct FittedBand {
    double Corner{0}, Slope{0};
};
// The dynamic range below a spectrum's peak that a band is read over.
// A measurement stops resolving surface below its instrument's noise, so a fit reaching to zero would be fitting the floor.
constexpr double BandRange{1e-8};

FittedBand FitBand(const SpectrumCells &cells) {
    const double floor = BandRange * (cells.Power.empty() ? 0 : *std::ranges::max_element(cells.Power));
    std::vector<double> log_at, log_power;
    for (size_t i = 0; i < cells.At.size(); ++i) {
        if (cells.Power[i] <= floor) continue;
        log_at.push_back(std::log(cells.At[i]));
        log_power.push_back(std::log(cells.Power[i]));
    }
    if (log_at.size() < 4) return {};
    FittedBand best;
    double best_error = 0;
    for (size_t corner = 0; corner + 3 < log_at.size(); ++corner) {
        const double at = log_at[corner];
        double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (size_t i = 0; i < log_at.size(); ++i) {
            const double x = std::max(0.0, log_at[i] - at), y = log_power[i];
            n += 1;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        const double denominator = n * sxx - sx * sx;
        if (denominator <= 0) continue;
        const double slope = (n * sxy - sx * sy) / denominator, intercept = (sy - slope * sx) / n;
        double error = 0;
        for (size_t i = 0; i < log_at.size(); ++i) {
            const double residual = log_power[i] - (intercept + slope * std::max(0.0, log_at[i] - at));
            error += residual * residual;
        }
        if (best.Corner == 0 || error < best_error) {
            best_error = error;
            best = {std::exp(at), slope};
        }
    }
    return best;
}

// A curve tabulated per bin, read at a fractional bin exactly as the field's amplitudes are.
double SpectrumAt(std::span<const double> curve, double at) {
    if (at < 1 || at >= double(curve.size() - 1)) return 0;
    const auto lo = size_t(at);
    return curve[lo] + (at - double(lo)) * (curve[lo + 1] - curve[lo]);
}

constexpr double RadialCellsPerOctave{16}, BandCellsPerOctave{8};
constexpr uint32_t RadialPasses{64};
std::vector<double> RadialSpectrum(std::span<const double> power, uint32_t rows, double rho) {
    const auto bins = power.size();
    std::vector<double> radial(bins, 0.0);
    if (bins < 8 || rho <= 0) return radial;
    const auto cells = CellSpectrum(power, RadialCellsPerOctave);
    const auto count = cells.At.size();
    if (count < 4) return radial;
    const auto corner = FitBand(cells).Corner;
    // Interpolate cell values onto bins.
    std::vector<double> held(count, 0.0);
    std::vector<size_t> cell(bins, 0);
    std::vector<double> share(bins, 0.0);
    for (size_t i = 1; i + 1 < bins; ++i) {
        const auto j = size_t(std::ranges::lower_bound(cells.At, double(i)) - cells.At.begin());
        cell[i] = std::clamp(j, size_t{1}, count - 1);
        share[i] = (double(i) - cells.At[cell[i] - 1]) / (cells.At[cell[i]] - cells.At[cell[i] - 1]);
        share[i] = std::clamp(share[i], 0.0, 1.0);
    }
    const auto fill = [&] {
        for (size_t i = 1; i + 1 < bins; ++i) radial[i] = held[cell[i] - 1] + share[i] * (held[cell[i]] - held[cell[i] - 1]);
    };
    // The exponent over the octave centred on each cell.
    const auto spread = std::max(size_t(RadialCellsPerOctave / 2), size_t{1});
    for (size_t j = 0; j < count; ++j) {
        const size_t lo = j > spread ? j - spread : 0, hi = std::min(count - 1, j + spread);
        const double slope = cells.Power[lo] > 0 && cells.Power[hi] > 0 && hi > lo ? std::log(cells.Power[hi] / cells.Power[lo]) / std::log(cells.At[hi] / cells.At[lo]) : -2.0;
        const double p = std::clamp(slope, -3.9, -1.1);
        const double integral = std::sqrt(std::numbers::pi) * std::exp(std::lgamma(-0.5 * p) - std::lgamma(0.5 * (1 - p)));
        held[j] = cells.Power[j] * rho / (integral * std::max(cells.At[j], corner));
    }
    for (uint32_t pass = 0; pass < RadialPasses; ++pass) {
        fill();
        for (size_t j = 0; j < count; ++j) {
            if (held[j] <= 0) continue;
            double cut = 0;
            for (uint32_t ky = 0; ky < rows; ++ky) {
                const double perpendicular = double(std::min(ky, rows - ky)) * rho;
                cut += SpectrumAt(radial, std::sqrt(cells.At[j] * cells.At[j] + perpendicular * perpendicular));
            }
            held[j] *= cut > 0 ? cells.Power[j] / cut : 0;
        }
    }
    fill();
    return radial;
}

// Re-realizes every synthesized surface, so a result can be checked across independent surface draws.
// Unset, every draw is fixed by its parameters alone.
uint64_t SurfaceSeed() {
    static const uint64_t seed = [] {
        const char *v = std::getenv("SURFACE_SEED");
        return v ? std::strtoull(v, nullptr, 10) : 0ull;
    }();
    return seed;
}

// A synthesis's draw state: its parameters hashed into `base`, then SURFACE_SEED where one is set.
uint64_t SurfaceDraw(uint64_t base, auto... values) {
    const auto state = HashParams(base, values...);
    const auto seed = SurfaceSeed();
    return seed ? HashParams(state, seed) : state;
}

std::atomic<uint32_t> RequestedTransformThreads{0};

// The thread count a synthesis runs at: SURFACE_FFT_THREADS when set, else the last requested count, else the machine's spare cores.
uint32_t TransformThreads() {
    static const int forced = [] {
        const char *v = std::getenv("SURFACE_FFT_THREADS");
        return v ? std::atoi(v) : 0;
    }();
    if (forced > 0) return uint32_t(forced);
    if (const auto set = RequestedTransformThreads.load(std::memory_order_relaxed); set > 0) return set;
    return std::max(1u, std::thread::hardware_concurrency() - 1);
}

// Run `work(begin, end)` over contiguous column ranges, one per synthesis thread.
// Every range derives its own starting state, so the split makes exactly the draws a serial pass would.
void ParallelColumns(uint32_t columns, auto &&work) {
    constexpr uint32_t MinColumnsPerThread = 256;
    const uint32_t threads = std::min(TransformThreads(), std::max(columns / MinColumnsPerThread, 1u));
    if (threads <= 1) return work(0u, columns);
    std::vector<std::jthread> pool;
    pool.reserve(threads);
    for (uint32_t t = 0; t < threads; ++t) {
        const auto begin = uint32_t(uint64_t(columns) * t / threads), end = uint32_t(uint64_t(columns) * (t + 1) / threads);
        pool.emplace_back([&work, begin, end] { work(begin, end); });
    }
}
} // namespace

void SetTransformThreads(uint32_t threads) {
    RequestedTransformThreads.store(std::max(threads, 1u), std::memory_order_relaxed);
}

RoughnessTrack SynthesizeRoughness(float correlation_length, float spectral_slope, float short_wavelength, float spacing, uint32_t count) {
    RoughnessTrack track;
    track.Spacing = spacing;
    track.Heights.assign(count, 0.f);
    if (count < 2 || spacing <= 0) {
        Finish(track);
        return track;
    }

    const uint32_t bins = count / 2 + 1;
    std::vector<std::complex<float>> spectrum(bins);
    const float q0 = 1.f / std::max(correlation_length, 1e-9f);
    track.Cutoff = std::max(short_wavelength, SurfaceSamplesPerCutoff * spacing);
    const float qs = 1.f / track.Cutoff;
    const float dq = 1.f / (float(count) * spacing);
    uint64_t state = SurfaceDraw(0x517cc1b727220a95ull, correlation_length, spectral_slope, short_wavelength, spacing);
    spectrum[0] = {}; // Zero mean.
    for (uint32_t i = 1; i < bins; ++i) {
        const float q = float(i) * dq;
        // The phase is drawn whether or not the bin has amplitude, so a band the surface does not reach leaves the rest of the realization untouched.
        const float phase = NextPhase(state);
        const float amplitude = q > qs ? 0.f : (q > q0 ? std::pow(q / q0, spectral_slope * 0.5f) : 1.f);
        spectrum[i] = {amplitude * std::cos(phase), amplitude * std::sin(phase)};
    }
    fft::ComplexToReal(spectrum, track.Heights);

    Finish(track);
    return track;
}

RoughnessField SynthesizeRoughnessPatch(float correlation_length, float spectral_slope, float short_wavelength, float spacing, uint32_t columns, uint32_t rows, uint32_t realization) {
    RoughnessField field{.Heights = std::vector<float>(size_t(columns) * rows, 0.f), .Columns = columns, .Rows = rows, .Spacing = spacing};
    if (columns < 2 || rows < 2 || spacing <= 0) return field;

    // Integrating a radial power law q^a over the perpendicular wavenumber leaves qx^(a+1), so a cut is one power shallower than the field it came from.
    // The patch therefore takes the track's slope less one.
    const float radial_slope = spectral_slope - 1;
    // Real-to-complex leaves the last axis half-length plus one.
    const uint32_t bins = rows / 2 + 1;
    std::vector<std::complex<float>> spectrum(size_t(columns) * bins);
    const float q0 = 1.f / std::max(correlation_length, 1e-9f);
    // The same cutoff a track takes, on the radial wavenumber.
    const float qs = 1.f / std::max(short_wavelength, 2 * spacing);
    const float dqx = 1.f / (float(columns) * spacing), dqy = 1.f / (float(rows) * spacing);
    const uint64_t state = SurfaceDraw(0x9e6b1f37a41c05d3ull, correlation_length, spectral_slope, short_wavelength, spacing, columns, rows, realization);
    // One phase draw per sample in row-major order, so parallel column ranges make the same draws as a serial pass.
    // The zero-wavenumber origin takes no draw.
    ParallelColumns(columns, [&](uint32_t begin, uint32_t end) {
        uint64_t local = SkipDraws(state, uint64_t(begin) * bins - (begin > 0 ? 1 : 0));
        for (uint32_t i = begin; i < end; ++i) {
            // Negative frequencies mirror, so the wavenumber runs to the Nyquist and back.
            const float qx = float(i <= columns / 2 ? i : columns - i) * dqx;
            for (uint32_t j = 0; j < bins; ++j) {
                const float qy = float(j) * dqy;
                const float q = std::sqrt(qx * qx + qy * qy);
                const size_t at = size_t(i) * bins + j;
                if (q <= 0) {
                    spectrum[at] = {}; // Zero mean.
                    continue;
                }
                const float phase = NextPhase(local);
                const float amplitude = q > qs ? 0.f : (q > q0 ? std::pow(q / q0, radial_slope * 0.5f) : 1.f);
                spectrum[at] = {amplitude * std::cos(phase), amplitude * std::sin(phase)};
            }
        }
    });
    fft::ComplexToReal2d(spectrum, columns, rows, field.Heights);

    const auto n = field.Heights.size();
    const auto mean = float(std::accumulate(field.Heights.begin(), field.Heights.end(), 0.0) / double(n));
    for (float &h : field.Heights) h -= mean;
    const auto energy = std::accumulate(field.Heights.begin(), field.Heights.end(), 0.0, [](double a, float h) { return a + double(h) * double(h); });
    if (const float rms = float(std::sqrt(energy / double(n))); rms > 0) {
        for (float &h : field.Heights) h /= rms;
    }
    return field;
}

LatticeCovariance RoughnessLatticeCovariance(float correlation_length, float spectral_slope, float short_wavelength, float spacing, uint32_t columns, uint32_t rows) {
    LatticeCovariance cov{};
    if (columns < 2 || rows < 2 || spacing <= 0) return cov;
    const double radial_slope = double(spectral_slope) - 1;
    const double q0 = 1 / std::max(double(correlation_length), 1e-9);
    const double qs = 1 / std::max(double(short_wavelength), 2.0 * spacing);
    const double dqx = 1 / (double(columns) * spacing), dqy = 1 / (double(rows) * spacing);
    for (uint32_t i = 0; i < columns; ++i) {
        const double qx = double(std::min(i, columns - i)) * dqx;
        for (uint32_t j = 0; j < rows; ++j) {
            const double qy = double(std::min(j, rows - j)) * dqy;
            const double q = std::sqrt(qx * qx + qy * qy);
            if (q <= 0 || q > qs) continue;
            const double power = q > q0 ? std::pow(q / q0, radial_slope) : 1.0;
            for (int dc = -2; dc <= 2; ++dc) {
                for (int dr = 0; dr <= 2; ++dr) {
                    const double phase = 2 * std::numbers::pi * (double(i) * dc / columns + double(j) * dr / rows);
                    cov[dc + 2][dr + 2] += power * std::cos(phase);
                }
            }
        }
    }
    for (int dc = -2; dc <= 2; ++dc) {
        for (int dr = -2; dr < 0; ++dr) cov[dc + 2][dr + 2] = cov[2 - dc][2 - dr];
    }
    if (const double norm = cov[2][2]; norm > 0) {
        for (auto &row : cov) {
            for (auto &c : row) c /= norm;
        }
    }
    return cov;
}

RoughnessTrack SynthesizeTurnover(float spacing, uint32_t count, uint64_t seed) {
    RoughnessTrack track;
    track.Spacing = spacing;
    track.Cutoff = SurfaceSamplesPerCutoff * spacing; // Sampling determines the white-noise cutoff.
    track.Heights.resize(count);
    uint64_t state = SurfaceDraw(seed);
    for (float &h : track.Heights) h = float(SplitMix64(state) >> 40) / float(1 << 24) - 0.5f;
    Finish(track);
    return track;
}

RoughnessTrack MakeProfileTrack(std::span<const float> heights, float spacing) {
    RoughnessTrack track;
    track.Spacing = spacing;
    track.Cutoff = SurfaceSamplesPerCutoff * spacing; // Sampling determines the measured cutoff.
    track.Heights.assign(heights.begin(), heights.end());
    track.Rms = Finish(track);
    if (heights.size() >= 8 && spacing > 0) {
        const auto spectrum = fft::RealToComplex(track.Heights);
        std::vector<double> power(spectrum.size());
        for (size_t i = 0; i < power.size(); ++i) power[i] = std::norm(spectrum[i]);
        // The fit's corner is in bins, which the track length converts back into a wavelength.
        if (const auto band = FitBand(CellSpectrum(power, BandCellsPerOctave)); band.Corner > 0) {
            track.Band = {float(double(heights.size()) * spacing / band.Corner), float(band.Slope)};
        }
    }
    return track;
}

RoughnessField SynthesizeProfileField(std::span<const float> heights, float spacing, uint32_t rows, uint32_t realization) {
    const auto columns = uint32_t(heights.size());
    RoughnessField field{.Heights = std::vector<float>(size_t(columns) * rows, 0.f), .Columns = columns, .Rows = rows, .Spacing = spacing};
    if (columns < 8 || rows < 2 || spacing <= 0) return field;

    const uint32_t bins_x = columns / 2 + 1, bins_y = rows / 2 + 1;
    const double rho = double(columns) / rows;
    const auto trace = fft::RealToComplex(heights);
    // The transform's inverse includes the sample count, so the cut the field is conditioned on is the trace's spectrum over that count.
    std::vector<double> power(bins_x);
    for (uint32_t i = 0; i < bins_x; ++i) power[i] = std::norm(trace[i]) / (double(columns) * columns);
    const auto radial = RadialSpectrum(power, rows, rho);
    const auto amplitude = [&](uint32_t i, uint32_t j) {
        const double qx = double(std::min(i, columns - i)), qy = double(j) * rho;
        return std::sqrt(std::max(0.0, SpectrumAt(radial, std::sqrt(qx * qx + qy * qy))));
    };

    std::vector<std::complex<float>> spectrum(size_t(columns) * bins_y);
    const auto at = [bins_y](uint32_t i, uint32_t j) { return size_t(i) * bins_y + j; };
    const uint64_t state = [&] {
        uint64_t s = HashParams(0x2545f4914f6cdd1dull, spacing, columns, rows, realization);
        for (const float h : heights) s = HashParams(s, h);
        return SurfaceDraw(s);
    }();
    // One phase draw per sample in row-major order, so parallel column ranges make the same draws as a serial pass.
    ParallelColumns(columns, [&](uint32_t begin, uint32_t end) {
        uint64_t local = SkipDraws(state, uint64_t(begin) * bins_y);
        for (uint32_t i = begin; i < end; ++i) {
            for (uint32_t j = 0; j < bins_y; ++j) {
                const auto a = float(amplitude(i, j));
                const float phase = NextPhase(local);
                spectrum[at(i, j)] = {a * std::cos(phase), a * std::sin(phase)};
            }
        }
    });
    const auto symmetrize = [&](uint32_t j) {
        spectrum[at(0, j)].imag(0.f);
        if (columns % 2 == 0) spectrum[at(columns / 2, j)].imag(0.f);
        for (uint32_t i = columns / 2 + 1; i < columns; ++i) spectrum[at(i, j)] = std::conj(spectrum[at(columns - i, j)]);
    };
    symmetrize(0);
    if (rows % 2 == 0) symmetrize(rows / 2);

    const uint32_t mirrored = rows % 2 == 0 ? bins_y - 2 : bins_y - 1;
    std::vector<std::complex<double>> miss(columns);
    std::vector<double> held(columns);
    for (uint32_t i = 0; i < columns; ++i) {
        const uint32_t opposite = i == 0 ? 0 : columns - i;
        std::complex<double> cut{spectrum[at(i, 0)]};
        for (uint32_t j = 0; j < bins_y; ++j) {
            const double a = amplitude(i, j);
            held[i] += a * a * (j > 0 && j <= mirrored ? 2 : 1);
            if (j == 0) continue;
            cut += std::complex<double>{spectrum[at(i, j)]};
            if (j <= mirrored) cut += std::conj(std::complex<double>{spectrum[at(opposite, j)]});
        }
        const auto measured = i <= columns / 2 ? std::complex<double>(trace[i]) : std::conj(std::complex<double>(trace[columns - i]));
        miss[i] = measured / double(columns) - cut;
    }
    for (uint32_t i = 0; i < columns; ++i) {
        if (held[i] <= 0) continue;
        for (uint32_t j = 0; j < bins_y; ++j) {
            const double a = amplitude(i, j);
            const auto share = miss[i] * (a * a / held[i]);
            spectrum[at(i, j)] += std::complex<float>{float(share.real()), float(share.imag())};
        }
    }

    fft::ComplexToReal2d(spectrum, columns, rows, field.Heights);
    return field;
}

/***** Element springs *****/

namespace {
// Fritsch-Carlson slopes: the tightest slopes that keep a cubic Hermite monotone through monotone data.
// A bearing curve only grows with engagement, and a dip between knots would read as an adhesive force.
void MonotoneSlopes(std::span<const float> x, std::span<const float> y, std::span<float> slope) {
    const size_t n = y.size();
    if (n < 2) return;
    for (size_t i = 0; i + 1 < n; ++i) {
        const float h = x[i + 1] - x[i];
        slope[i] = h > 0 ? (y[i + 1] - y[i]) / h : 0.f; // delta, replaced below for interior knots
    }
    const float first = slope[0], last = slope[n - 2];
    for (size_t i = n - 2; i >= 1; --i) {
        const float dl = slope[i - 1], dr = slope[i];
        if (dl * dr <= 0) {
            slope[i] = 0;
            continue;
        }
        const float hl = x[i] - x[i - 1], hr = x[i + 1] - x[i];
        const float w1 = 2 * hr + hl, w2 = hr + 2 * hl;
        slope[i] = (w1 + w2) / (w1 / dl + w2 / dr);
    }
    slope[0] = first;
    slope[n - 1] = last;
}

// An element index offset by up to `reach` either way, wrapped into the surface it tiles.
uint32_t Wrap(int64_t at, uint32_t count, uint32_t reach) {
    const int64_t n = int64_t(count);
    return uint32_t((at + n * (1 + int64_t(reach) / n)) % n);
}

// The monotone cubic Hermite's force at `t` across a knot interval of width `h`.
float HermiteForce(float f0, float f1, float d0, float d1, float h, float t, float t2, float t3) {
    return f0 * (2 * t3 - 3 * t2 + 1) + h * d0 * (t3 - 2 * t2 + t) +
        f1 * (-2 * t3 + 3 * t2) + h * d1 * (t3 - t2);
}

uint32_t KnotBelow(const ElementSprings &s, float engagement) {
    const auto *z = s.Engagement.data();
    if (s.KnotByExponent.empty()) {
        const auto hi = uint32_t(std::upper_bound(z, z + s.Knots, engagement) - z);
        return hi > 0 ? hi - 1 : 0;
    }
    const auto x = int32_t(std::bit_cast<uint32_t>(engagement) >> 23) - 127;
    uint32_t i = x < s.KnotExponentFloor ? 0u : s.KnotByExponent[size_t(std::min(x - s.KnotExponentFloor, int32_t(s.KnotByExponent.size()) - 1))];
    while (i + 1 < s.Knots && z[i + 1] <= engagement) ++i;
    return i;
}

float ElementTableAt(const ElementSprings &s, std::span<const float> table, uint32_t element, float engagement) {
    const uint32_t k = s.Knots;
    if (k < 2 || engagement <= 0) return 0.f;
    const auto *z = s.Engagement.data();
    const auto *v = table.data() + size_t(s.CurveOf(element)) * k;
    if (engagement >= z[k - 1]) return v[k - 1];
    const uint32_t i = KnotBelow(s, engagement);
    const float h = z[i + 1] - z[i];
    return h > 0 ? v[i] + (v[i + 1] - v[i]) * (engagement - z[i]) / h : v[i];
}

double BlockReachTop(const ElementSprings &s, uint32_t block, int64_t at, double position, double combined_curvature) {
    const double near = (double(at) + 0.5 - position) * double(s.Width);
    const double far = (double(at + ElementSprings::CrestBlock - 1) + 0.5 - position) * double(s.Width);
    const double closest = near <= 0 && far >= 0 ? 0.0 : std::min(std::abs(near), std::abs(far));
    return double(s.BlockCrest[block]) - 0.5 * combined_curvature * closest * closest;
}

void WalkFootprintElements(const ElementSprings &s, double position, double combined_curvature, uint32_t reach, auto &&bound, auto &&visit) {
    constexpr auto Block = int64_t(ElementSprings::CrestBlock);
    const uint32_t count = s.Count();
    const auto e0 = int64_t(std::floor(position));
    const auto last = e0 + int64_t(reach);
    auto at = e0 - int64_t(reach);
    auto e = Wrap(at, count, reach);
    while (at <= last) {
        if (e % ElementSprings::CrestBlock == 0 && at + Block - 1 <= last && !s.BlockCrest.empty() &&
            BlockReachTop(s, e / ElementSprings::CrestBlock, at, position, combined_curvature) <= bound()) {
            at += Block;
            e += ElementSprings::CrestBlock;
            while (e >= count) e -= count;
            continue;
        }
        const double offset = (double(at) + 0.5 - position) * double(s.Width);
        visit(at, e, 0.5 * combined_curvature * offset * offset);
        ++at;
        if (++e == count) e = 0;
    }
}

// A summit's depth as divided between its contact and the roughness inside that contact.
struct SummitDepth {
    double Hertz{0}; // Depth the contact itself takes, m.
    double Share{1}; // Share of a step of approach the contact takes, the roughness taking the rest.
};

class DepthSplit {
public:
    DepthSplit(const SubCutoffRoughness &sub, double inv_modulus)
        : Hurst{sub.Hurst}, InvModulus{inv_modulus}, Splits{sub.Amplitude > 0 && sub.Hurst > 0 && sub.MaxWidth > 0 && inv_modulus > 0} {
        if (!Splits) return;
        ClampWidth = sub.MaxWidth;
        ClampSeparation = SubCutoffSeparation(sub, sub.MaxWidth);
        // The depth the roughness takes at a summit whose Hertz constant is one, which one coefficient scales to the whole population.
        Coefficient = (3 / Hurst) * ClampSeparation * std::pow(1.5 * inv_modulus / sub.MaxWidth, Hurst);
        LogHertz.resize(Count);
        for (uint32_t i = 0; i < Count; ++i) {
            const double t = std::exp(LogMin + LogStep * i);
            // Both asymptotes are exact at their own end, so one of them is a close start for every solve.
            double y = t < 1 ? std::pow(t, 2 / Hurst) : t;
            for (uint32_t step = 0; step < 40; ++step) {
                const double p = std::pow(y, 0.5 * Hurst);
                const double move = (y + p - t) / (1 + 0.5 * Hurst * p / y);
                y -= move;
                if (std::abs(move) <= 1e-15 * y) break;
            }
            LogHertz[i] = std::log(y);
        }
    }

    // The depth scale of a summit's split, which is all it enters the curve through.
    double Scale(double stiffness) const {
        return Splits ? std::pow(Coefficient * std::pow(stiffness, Hurst), 2 / (2 - Hurst)) : 0.0;
    }

    // How a summit of Hertz constant `stiffness` at its `scale` divides a total depth `d`.
    SummitDepth Split(double d, double stiffness, double scale) const {
        if (!Splits || d <= 0) return {d, 1};
        const double t = d / scale;
        const double u = std::log(t);
        double x = scale *
            (u <= LogMin ? std::pow(t, 2 / Hurst) : u >= LogMax ? t - std::pow(t, 0.5 * Hurst) :
                                                                  Interpolate(u));
        if (const double clamp_depth = ClampDepth(stiffness); x > clamp_depth) {
            for (uint32_t step = 0; step < 4; ++step) {
                const double held = ClampSeparation * (3 / Hurst + 1.5 * std::log(x / clamp_depth));
                x -= (x + held - d) / (1 + 1.5 * ClampSeparation / x);
            }
            return {x, 1 / (1 + 1.5 * ClampSeparation / x)};
        }
        return {x, x > 0 && d > x ? 1 / (1 + 0.5 * Hurst * (d - x) / x) : 1.0};
    }

private:
    static constexpr uint32_t Count{4096};
    static constexpr double LogMin{-27.6}, LogMax{27.6}, LogStep{(LogMax - LogMin) / (Count - 1)};

    double Interpolate(double u) const {
        const double at = (u - LogMin) / LogStep;
        const auto i = uint32_t(at);
        return std::exp(LogHertz[i] + (at - double(i)) * (LogHertz[i + 1] - LogHertz[i]));
    }
    // Depth at which this summit's contact reaches the width the band stops widening at.
    double ClampDepth(double stiffness) const {
        const double root = ClampWidth / (1.5 * stiffness * InvModulus);
        return root * root;
    }

    double Hurst{0}, InvModulus{0}, Coefficient{0}, ClampWidth{0}, ClampSeparation{0};
    bool Splits{false};
    std::vector<double> LogHertz;
};

// The engagement knots and their power-of-two index, shared by the realized and ensemble builds.
// Geometric from a sub-asperity toe, since first touch is one summit and the light-load regime falls within microns of it.
void FillEngagementKnots(ElementSprings &out, double engagement_max, uint32_t knots) {
    out.Engagement.resize(knots);
    constexpr double ToeEngagement = 2e-8;
    out.Engagement[0] = 0;
    for (uint32_t j = 1; j < knots; ++j) {
        const double t = double(j - 1) / double(knots - 2);
        out.Engagement[j] = float(ToeEngagement * std::pow(engagement_max / ToeEngagement, t));
    }
    if (out.Engagement[1] > 0 && out.Engagement[knots - 1] >= out.Engagement[1] && knots <= 256) {
        out.KnotExponentFloor = std::ilogb(out.Engagement[1]);
        const int32_t top = std::ilogb(out.Engagement[knots - 1]);
        out.KnotByExponent.assign(size_t(top - out.KnotExponentFloor) + 1, 0);
        for (int32_t x = out.KnotExponentFloor; x <= top; ++x) {
            const auto low = float(std::ldexp(1.0, x));
            uint32_t i = 0;
            while (i + 1 < knots && out.Engagement[i + 1] < low) ++i;
            out.KnotByExponent[size_t(x - out.KnotExponentFloor)] = uint8_t(i);
        }
    }
}

// The tables a build fills: `curves` curve sets of `knots` knots each, plus the population's strength spectrum.
void InitSpringTables(ElementSprings &out, std::span<const float> crests, float width, double engagement_max, uint32_t curves, uint32_t knots) {
    out.Width = width;
    out.EngagementMax = float(engagement_max);
    out.Knots = knots;
    out.Curves = curves;
    const auto [low, high] = std::ranges::minmax_element(crests);
    out.CrestRange = *high - *low;
    FillEngagementKnots(out, engagement_max, knots);
    const size_t n = size_t(curves) * knots;
    out.Force.assign(n, 0.f);
    out.Slope.assign(n, 0.f);
    out.Potential.assign(n, 0.f);
    out.ForceVariance.assign(n, 0.f);
    out.FlankMoment.assign(n, 0.f);
    out.BearingCount.assign(n, 0.f);
    out.BearingWidth.assign(n, 0.f);
    out.FlankSpectrum.assign(size_t(knots) * FlankSpectrumBins, 0.f);
    out.FlankSpectrumForce.assign(size_t(knots) * FlankSpectrumBins, 0.f);
}

// One filled force curve's interpolant slopes and its own integral at the knots.
// The potential at the knots is that integral, so a read at a knot and a read either side of it agree.
void FillCurveTail(ElementSprings &out, uint32_t curve) {
    const uint32_t knots = out.Knots;
    const size_t base = size_t(curve) * knots;
    const auto *f = out.Force.data() + base;
    auto *d = out.Slope.data() + base;
    auto *u = out.Potential.data() + base;
    MonotoneSlopes(out.Engagement, std::span{f, knots}, std::span{d, knots});
    for (uint32_t j = 0; j + 1 < knots; ++j) {
        const float h = out.Engagement[j + 1] - out.Engagement[j];
        u[j + 1] = u[j] + h * (0.5f * (f[j] + f[j + 1]) + h * (d[j] - d[j + 1]) / 12.f);
    }
}
} // namespace

SubCutoffRoughness UnresolvedBand(double roughness, double correlation_length, double short_wavelength, double slope) {
    const double hurst = 0.5 * (-slope - 1);
    if (roughness <= 0 || hurst <= 0 || hurst >= 1 || short_wavelength <= 0 || short_wavelength >= correlation_length) return {};
    const double whole = 0.5 + (1 - std::pow(correlation_length / short_wavelength, -2 * hurst)) / (2 * hurst);
    return {
        .Amplitude = roughness / (std::sqrt(2 * hurst * whole) * std::pow(correlation_length, hurst)),
        .Hurst = hurst,
        .MaxWidth = short_wavelength,
    };
}

SpringRead ReadElementSpring(const ElementSprings &s, uint32_t element, float engagement) {
    const uint32_t k = s.Knots;
    if (k < 2 || element >= s.Count() || engagement <= 0) return {};
    const auto *z = s.Engagement.data();
    const size_t base = size_t(s.CurveOf(element)) * k;
    const auto *f = s.Force.data() + base;
    const auto *d = s.Slope.data() + base;
    const auto *u = s.Potential.data() + base;
    // Past the tabulated reach the curve continues at its final slope, so force and potential stay consistent instead of saturating into a false equilibrium.
    if (engagement >= z[k - 1]) {
        const float e = engagement - z[k - 1];
        return {f[k - 1] + d[k - 1] * e, u[k - 1] + f[k - 1] * e + 0.5f * d[k - 1] * e * e};
    }
    const uint32_t i = KnotBelow(s, engagement);
    const float h = z[i + 1] - z[i];
    if (h <= 0) return {f[i], u[i]};
    const float t = (engagement - z[i]) / h, t2 = t * t, t3 = t2 * t, t4 = t3 * t;
    const float force = HermiteForce(f[i], f[i + 1], d[i], d[i + 1], h, t, t2, t3);
    // The potential is that same cubic's integral, so the force is its exact derivative everywhere.
    const float potential = u[i] +
        h * (f[i] * (0.5f * t4 - t3 + t) + h * d[i] * (0.25f * t4 - (2.f / 3) * t3 + 0.5f * t2) + f[i + 1] * (-0.5f * t4 + t3) + h * d[i + 1] * (0.25f * t4 - (1.f / 3) * t3));
    return {force, potential};
}

float ElementForceVariance(const ElementSprings &s, uint32_t element, float engagement) {
    const uint32_t k = s.Knots;
    if (k < 2 || element >= s.Count() || engagement <= 0 || s.ForceVariance.empty()) return 0.f;
    const auto *z = s.Engagement.data();
    const auto *v = s.ForceVariance.data() + size_t(s.CurveOf(element)) * k;
    if (engagement >= z[k - 1]) return v[k - 1];
    const uint32_t i = KnotBelow(s, engagement);
    const float h = z[i + 1] - z[i];
    if (h <= 0) return v[i];
    const float t = (engagement - z[i]) / h;
    return v[i] + t * (v[i + 1] - v[i]);
}

void RefreshCrestBlocks(ElementSprings &s) {
    const uint32_t count = s.Count();
    const uint32_t blocks = (count + ElementSprings::CrestBlock - 1) / ElementSprings::CrestBlock;
    s.BlockCrest.assign(blocks, -std::numeric_limits<float>::max());
    for (uint32_t i = 0; i < blocks * ElementSprings::CrestBlock; ++i) {
        float &top = s.BlockCrest[i / ElementSprings::CrestBlock];
        top = std::max(top, s.Crest[i % count]);
    }
}

uint32_t ElementReach(const ElementSprings &s, double combined_curvature) {
    const uint32_t count = s.Count();
    if (count == 0 || s.Width <= 0) return 0;
    if (combined_curvature <= 0) return count / 2; // A flat body reaches everywhere its footprint spans.
    const double stand = double(s.CrestRange) + double(s.EngagementMax);
    const double distance = std::sqrt(2 * stand / combined_curvature);
    return std::min(uint32_t(distance / double(s.Width)) + 2, count / 2);
}

std::vector<float> CrestEnvelope(std::span<const float> crest, float width, double combined_curvature, uint32_t reach) {
    const auto count = uint32_t(crest.size());
    std::vector<float> envelope(count);
    for (uint32_t e = 0; e < count; ++e) {
        double top = -std::numeric_limits<double>::max();
        for (int64_t o = -int64_t(reach); o <= int64_t(reach); ++o) {
            const auto at = Wrap(int64_t(e) + o, count, reach);
            const double offset = double(o) * double(width);
            top = std::max(top, double(crest[at]) - 0.5 * combined_curvature * offset * offset);
        }
        envelope[e] = float(top);
    }
    return envelope;
}

std::vector<float> ElementEnvelope(const ElementSprings &s, double combined_curvature, uint32_t reach) {
    return CrestEnvelope(s.Crest, s.Width, combined_curvature, reach);
}

namespace {
// Force and the force curve's slope at one element's engagement, the Hermite cubic of ReadElementSpring differentiated in place.
std::pair<float, float> ElementForceSlope(const ElementSprings &s, uint32_t element, float engagement) {
    const uint32_t k = s.Knots;
    if (k < 2 || element >= s.Count() || engagement <= 0) return {0.f, 0.f};
    const auto *z = s.Engagement.data();
    const size_t base = size_t(s.CurveOf(element)) * k;
    const auto *f = s.Force.data() + base;
    const auto *d = s.Slope.data() + base;
    if (engagement >= z[k - 1]) return {f[k - 1] + d[k - 1] * (engagement - z[k - 1]), d[k - 1]};
    const uint32_t i = KnotBelow(s, engagement);
    const float h = z[i + 1] - z[i];
    if (h <= 0) return {f[i], d[i]};
    const float t = (engagement - z[i]) / h, t2 = t * t, t3 = t2 * t;
    const float slope = (f[i] * (6 * t2 - 6 * t) + h * d[i] * (3 * t2 - 4 * t + 1) +
                         f[i + 1] * (-6 * t2 + 6 * t) + h * d[i + 1] * (3 * t2 - 2 * t)) /
        h;
    return {HermiteForce(f[i], f[i + 1], d[i], d[i + 1], h, t, t2, t3), slope};
}

// Returns first contact, force, and stiffness over the footprint at `position`.
struct FootprintWalk {
    double Touch{-std::numeric_limits<double>::max()}; // Highest crest under the body's gap, m.
    double Force{0}; // N
    double Stiffness{0}; // dForce / d(descent), N/m
};
FootprintWalk WalkFootprint(const ElementSprings &s, double surface, double position, double combined_curvature, uint32_t reach) {
    FootprintWalk walk;
    // The block top bounds a skipped run's touch as well as its force, so the same bound covers both.
    WalkFootprintElements(
        s, position, combined_curvature, reach,
        [&] { return std::min(surface, walk.Touch); },
        [&](int64_t, uint32_t e, double lift) {
            const double stands = double(s.Crest[e]) - lift;
            walk.Touch = std::max(walk.Touch, stands);
            const auto [force, slope] = ElementForceSlope(s, e, float(stands - surface));
            walk.Force += double(force);
            walk.Stiffness += double(slope);
        }
    );
    return walk;
}
// An element's fractional split between the two bins nearest its footprint coordinate, `coord` in elements from the window's leading edge.
// The tent weights slide continuously with the contact's travel, so a bin's share of an element never steps.
struct BinShare {
    size_t B0, B1;
    float W1;
};
BinShare BinSplit(double coord, int64_t bins, double window) {
    const double g = std::clamp(coord * double(bins) / window - 0.5, 0.0, double(bins - 1));
    const auto b0 = size_t(g);
    return {b0, std::min(b0 + 1, size_t(bins - 1)), float(g - double(b0))};
}
} // namespace

std::vector<float> BearingDatum(const ElementSprings &s, double combined_curvature, uint32_t reach, double load) {
    const uint32_t count = s.Count();
    if (count == 0 || !(load > 0)) return ElementEnvelope(s, combined_curvature, reach);
    std::vector<double> seat(count), touch(count);
    double held = 0;
    uint32_t unseated = 0;
    double worst = 0;
    for (uint32_t e = 0; e < count; ++e) {
        const double position = double(e) + 0.5;
        const double top = WalkFootprint(s, std::numeric_limits<double>::max(), position, combined_curvature, reach).Touch;
        double y = e == 0 ? top - 0.5 * double(s.EngagementMax) : held;
        bool seated = false;
        double residual = 0;
        for (uint32_t step = 0; step < 24 && !seated; ++step) {
            const auto walk = WalkFootprint(s, y, position, combined_curvature, reach);
            residual = std::abs(walk.Force - load) / load;
            if (residual <= 1e-6) {
                seated = true;
                break;
            }
            // A surface standing above every crest bears nothing and has no slope to steer by, so it descends from first touch until it finds the population.
            if (walk.Stiffness <= 0) {
                y = std::min(y, top) - 0.25 * double(s.EngagementMax);
                continue;
            }
            y += std::clamp((walk.Force - load) / walk.Stiffness, -double(s.EngagementMax), double(s.EngagementMax));
        }
        seat[e] = y;
        touch[e] = top;
        held = y;
        unseated += seated ? 0u : 1u;
        worst = std::max(worst, residual);
    }
    // A position whose balance never closed keeps whatever height it reached, and the array is read as a datum either way.
    if (unseated > 0) {
        std::fprintf(
            stderr, "SPRINGSEAT %u of %u positions did not close on a load of %g N, worst off by %.3g of it\n",
            unseated, count, load, worst
        );
    }
    // The array keeps the mean height at first touch with the seat's variation on it.
    double gauge = 0;
    for (uint32_t e = 0; e < count; ++e) gauge += touch[e] - seat[e];
    gauge /= double(count);
    std::vector<float> datum(count);
    for (uint32_t e = 0; e < count; ++e) datum[e] = float(seat[e] + gauge);
    return datum;
}

float EnvelopeAt(std::span<const float> envelope, double position) {
    const auto n = int64_t(envelope.size());
    if (n == 0) return 0.f;
    const double p = position - 0.5;
    const double base = std::floor(p);
    const auto i0 = uint64_t(((int64_t(base) % n) + n) % n);
    const auto i1 = i0 + 1 < uint64_t(n) ? i0 + 1 : 0;
    return envelope[i0] + float(p - base) * (envelope[i1] - envelope[i0]);
}

float Reregister(const SlidingRegistration &slide, double at) {
    if (slide.SlowCrestDev.empty()) return 0.f;
    return EnvelopeAt(slide.SlowCrestDev, at + 0.5 - slide.Slip) - EnvelopeAt(slide.SlowCrestDev, at + 0.5);
}

SpringRead ReadContactSpringBins(
    const ElementSprings &s, std::span<const float> envelope, double position, float engagement, double combined_curvature, uint32_t reach,
    std::span<float> bin_force, std::span<const float> bin_engagement, const SlidingRegistration &slide
) {
    const uint32_t count = s.Count();
    if (count == 0 || envelope.size() < count || engagement <= 0) return {};
    // The body's surface, placed by how far it has descended below first touch here.
    const double surface = double(EnvelopeAt(envelope, position)) + double(Reregister(slide, position - 0.5)) - double(engagement);
    const auto bins = int64_t(bin_force.size());
    const bool modulated = bin_engagement.size() >= bin_force.size() && !bin_force.empty();
    const double window = double(2 * reach + 1);
    float deepest = 0;
    for (const float off : bin_engagement) deepest = std::max(deepest, off);
    const double reaches = surface - double(deepest);
    SpringRead total;
    WalkFootprintElements(
        s, position, combined_curvature, reach,
        [reaches] { return reaches; },
        [&](int64_t at, uint32_t e, double lift) {
            const float eng = float(double(s.Crest[e]) - surface - lift);
            const auto read = ReadElementSpring(s, e, eng);
            total.Force += read.Force;
            total.Potential += read.Potential;
            if (bins > 0) {
                const auto [b0, b1, w1] = BinSplit(double(at) + 0.5 - position + double(reach) + 0.5, bins, window);
                const float off = (1.f - w1) * (modulated ? bin_engagement[b0] : 0.f) + w1 * (modulated ? bin_engagement[b1] : 0.f);
                const float f = modulated ? ReadElementSpring(s, e, eng + off).Force : read.Force;
                if (f > 0) {
                    bin_force[b0] += (1.f - w1) * f;
                    bin_force[b1] += w1 * f;
                }
            }
        }
    );
    return total;
}

SpringRead ReadContactSpringChannels(
    const ElementSprings &s, std::span<const float> envelope, double position, float engagement, double combined_curvature, uint32_t reach,
    std::span<float> bin_force, std::span<float> bin_potential, std::span<const float> bin_engagement, const SlidingRegistration &slide
) {
    const uint32_t count = s.Count();
    const auto bins = int64_t(bin_force.size());
    if (count == 0 || envelope.size() < count || bins == 0) return {};
    // A stack whose deepest channel cannot reach the surface bears exactly zero.
    float deepest = bin_engagement.empty() ? 0.f : bin_engagement[0];
    for (const float off : bin_engagement) deepest = std::max(deepest, off);
    if (engagement + deepest <= 0) return {};
    const double surface = double(EnvelopeAt(envelope, position)) + double(Reregister(slide, position - 0.5)) - double(engagement);
    const double window = double(2 * reach + 1);
    const double reaches = surface - double(std::max(deepest, 0.f));
    SpringRead total;
    WalkFootprintElements(
        s, position, combined_curvature, reach,
        [reaches] { return reaches; },
        [&](int64_t at, uint32_t e, double lift) {
            const auto [b0, b1, w1] = BinSplit(double(at) + 0.5 - position + double(reach) + 0.5, bins, window);
            const float off = bin_engagement.empty() ? 0.f : (1.f - w1) * bin_engagement[b0] + w1 * bin_engagement[b1];
            const float eng = float(double(s.Crest[e]) - surface - lift) + off;
            const auto read = ReadElementSpring(s, e, eng);
            total.Force += read.Force;
            total.Potential += read.Potential;
            bin_force[b0] += (1.f - w1) * read.Force;
            bin_force[b1] += w1 * read.Force;
            bin_potential[b0] += (1.f - w1) * read.Potential;
            bin_potential[b1] += w1 * read.Potential;
        }
    );
    return total;
}

SpringRead ReadContactSprings(const ElementSprings &s, std::span<const float> envelope, double position, float engagement, double combined_curvature, uint32_t reach, const SlidingRegistration &slide) {
    return ReadContactSpringBins(s, envelope, position, engagement, combined_curvature, reach, {}, {}, slide);
}

namespace {
// The tabulated knot a read at `engagement` takes its spectrum from, the deepest one at or below it.
uint32_t SpectrumKnot(const ElementSprings &s, float engagement) {
    uint32_t knot = 1;
    while (knot + 1 < s.Knots && s.Engagement[knot + 1] <= engagement) ++knot;
    return knot;
}

// One knot's shares, each array normalised over the bins. Zero where that knot bears nothing.
bool KnotShares(const ElementSprings &s, uint32_t knot, std::span<float> moment_out, std::span<float> force_out) {
    const size_t at = size_t(knot) * FlankSpectrumBins;
    const auto *moment = s.FlankSpectrum.data() + at;
    const auto *force = s.FlankSpectrumForce.data() + at;
    double moment_total = 0, force_total = 0;
    for (uint32_t b = 0; b < FlankSpectrumBins; ++b) {
        moment_total += double(moment[b]);
        force_total += double(force[b]);
    }
    if (!(moment_total > 0) || !(force_total > 0)) return false;
    for (uint32_t b = 0; b < FlankSpectrumBins; ++b) {
        moment_out[b] = float(double(moment[b]) / moment_total);
        force_out[b] = float(double(force[b]) / force_total);
    }
    return true;
}
} // namespace

FlankSpectrumShares FlankSpectrumShareAt(const ElementSprings &s, float engagement) {
    FlankSpectrumShares out;
    if (s.Knots == 0 || s.FlankSpectrum.empty() || s.FlankSpectrumForce.empty() || !(engagement > 0)) return out;
    // Interpolated between the bracketing knots, as every other tabulated curve here is read.
    // A bin is a fixed fraction of its knot's engagement, so bin `b` means the same relative band of strengths at every knot.
    const uint32_t lo = SpectrumKnot(s, engagement);
    if (!KnotShares(s, lo, out.Moment, out.Force)) return {};
    if (lo + 1 >= s.Knots) return out;
    const float span = s.Engagement[lo + 1] - s.Engagement[lo];
    if (!(span > 0)) return out;
    std::array<float, FlankSpectrumBins> moment_hi{}, force_hi{};
    if (!KnotShares(s, lo + 1, moment_hi, force_hi)) return out;
    const float t = std::clamp((engagement - s.Engagement[lo]) / span, 0.f, 1.f);
    for (uint32_t b = 0; b < FlankSpectrumBins; ++b) {
        out.Moment[b] += t * (moment_hi[b] - out.Moment[b]);
        out.Force[b] += t * (force_hi[b] - out.Force[b]);
    }
    return out;
}

SpringFlankRead SpringFlankMoments(const ElementSprings &s, std::span<const float> envelope, double position, float engagement, double combined_curvature, uint32_t reach) {
    const uint32_t count = s.Count();
    if (count == 0 || envelope.size() < count || engagement <= 0) return {};
    const double surface = double(EnvelopeAt(envelope, position)) - double(engagement);
    SpringFlankRead out;
    WalkFootprintElements(
        s, position, combined_curvature, reach,
        [surface] { return surface; },
        [&](int64_t, uint32_t e, double lift) {
            const auto engagement_here = float(double(s.Crest[e]) - surface - lift);
            const auto [force, slope] = ElementForceSlope(s, e, engagement_here);
            if (force > 0 && slope > 0) {
                out.Modulus += double(ElementTableAt(s, s.FlankMoment, e, engagement_here));
                out.Stiffness += slope;
                out.Bearing += double(ElementTableAt(s, s.BearingCount, e, engagement_here));
                out.Width += double(ElementTableAt(s, s.BearingWidth, e, engagement_here));
            }
        }
    );
    return out;
}

float SolveSpringEngagement(const ElementSprings &s, std::span<const float> envelope, double normal_force, double combined_curvature, uint32_t reach) {
    const uint32_t count = s.Count();
    if (count == 0 || normal_force <= 0) return 0;
    // Every element in turn as the contact centre, so the anchor is the surface average.
    const auto mean_force = [&](float engagement) {
        double sum = 0;
        for (uint32_t e = 0; e < count; ++e) sum += ReadContactSprings(s, envelope, double(e) + 0.5, engagement, combined_curvature, reach).Force;
        return sum / double(count);
    };
    float low = 0, high = s.EngagementMax;
    while (mean_force(high) < normal_force && high < 1e3f * s.EngagementMax) high *= 2;
    for (uint32_t i = 0; i < 60; ++i) {
        const float mid = 0.5f * (low + high);
        (mean_force(mid) > normal_force ? high : low) = mid;
    }
    return 0.5f * (low + high);
}

void SweepModeDrives(std::span<const float> field, std::span<const float> phi, uint32_t modes, uint32_t reach, std::span<float> table) {
    const auto count = uint32_t(field.size());
    if (count == 0 || modes == 0) return;
    const uint32_t window = 2 * reach + 1;
    // Sorted positions of the elements bearing force, which contact leaves sparse.
    std::vector<uint32_t> bearing;
    for (uint32_t e = 0; e < count; ++e) {
        if (field[e] > 0) bearing.push_back(e);
    }
    std::vector<float> acc(modes);
    for (uint32_t x = 0; x < count; ++x) {
        std::ranges::fill(acc, 0.f);
        const uint32_t first = (x + count - reach) % count;
        const auto sum_at = [&](uint32_t e, uint32_t w) {
            if (w >= window) return false;
            const float f = field[e];
            const auto *ph = &phi[size_t(w) * modes];
            for (uint32_t k = 0; k < modes; ++k) acc[k] += f * ph[k];
            return true;
        };
        const auto begin = std::ranges::lower_bound(bearing, first);
        bool inside = true;
        for (uint32_t base = 0; inside && base < window; base += count) {
            for (auto it = begin; inside && it != bearing.end(); ++it) inside = sum_at(*it, base + *it - first);
            for (auto it = bearing.begin(); inside && it != begin; ++it) inside = sum_at(*it, base + *it + count - first);
        }
        for (uint32_t k = 0; k < modes; ++k) table[size_t(k) * count + x] = acc[k];
    }
}

namespace {
bool IsSummit(const float *prev, const float *cur, const float *next, uint32_t r, uint32_t rows) {
    const float h = cur[r];
    for (const float *col : {prev, cur, next}) {
        if (r > 0 && col[r - 1] > h) return false;
        if (col != cur && col[r] > h) return false;
        if (r + 1 < rows && col[r + 1] > h) return false;
    }
    return true;
}

// Curvature at that summit, the mean of the second differences along and across the track.
// A summit on the field's edge has no across neighbours, so it takes the along curvature for both.
double SummitCurvature(const float *prev, const float *cur, const float *next, uint32_t r, uint32_t rows, float column_spacing) {
    const double h = cur[r];
    const double along = double(prev[r]) - 2 * h + double(next[r]);
    const double across = r > 0 && r + 1 < rows ? double(cur[r - 1]) - 2 * h + double(cur[r + 1]) : along;
    return -(along + across) / (double(column_spacing) * column_spacing);
}
} // namespace

std::vector<uint8_t> MarkFieldSummits(std::span<const float> folded, uint32_t columns, uint32_t rows) {
    std::vector<uint8_t> summit(folded.size(), 0);
    for (uint32_t c = 0; c < columns; ++c) {
        const uint32_t cl = c > 0 ? c - 1 : columns - 1;
        const uint32_t cr = c + 1 < columns ? c + 1 : 0;
        const auto *cur = &folded[size_t(c) * rows];
        for (uint32_t r = 0; r < rows; ++r) {
            summit[size_t(c) * rows + r] = IsSummit(&folded[size_t(cl) * rows], cur, &folded[size_t(cr) * rows], r, rows) ? 1 : 0;
        }
    }
    return summit;
}

double FieldSummitCurvature(std::span<const float> folded, uint32_t columns, uint32_t rows, float column_spacing, uint32_t column, uint32_t row) {
    const uint32_t cl = column > 0 ? column - 1 : columns - 1, cr = column + 1 < columns ? column + 1 : 0;
    return SummitCurvature(&folded[size_t(cl) * rows], &folded[size_t(column) * rows], &folded[size_t(cr) * rows], row, rows, column_spacing);
}

void GatherElementCrests(
    std::vector<float> &out, std::span<const float> heights, uint32_t columns, uint32_t rows,
    std::span<const float> transverse_gap, std::span<const float> element_lift, uint32_t element_columns
) {
    if (columns == 0 || rows == 0 || element_columns == 0) return;
    const uint32_t count = columns / element_columns;
    if (count == 0) return;
    out.resize(count, -std::numeric_limits<float>::max());
    for (uint32_t e = 0; e < count; ++e) {
        const float lift = e < element_lift.size() ? element_lift[e] : 0.f;
        float crest = -std::numeric_limits<float>::max();
        for (uint32_t c = 0; c < element_columns; ++c) {
            const size_t src = size_t(e * element_columns + c) * rows;
            for (uint32_t r = 0; r < rows; ++r) crest = std::max(crest, heights[src + r] - transverse_gap[r]);
        }
        out[e] = std::max(out[e], crest + lift);
    }
}

void GatherElementSummits(
    ElementSummits &out, std::span<const float> heights, uint32_t columns, uint32_t rows,
    std::span<const float> transverse_gap, std::span<const float> element_lift,
    uint32_t element_columns, float column_spacing, double inv_effective_modulus
) {
    if (columns == 0 || rows == 0 || element_columns == 0) return;
    const uint32_t count = columns / element_columns;
    if (count == 0) return;

    out.Width = float(element_columns) * column_spacing;
    out.InvModulus = inv_effective_modulus;
    out.Crest.resize(count, -std::numeric_limits<float>::max());

    std::vector<float> window(size_t(3) * rows);
    auto *prev = window.data();
    auto *cur = prev + rows;
    auto *next = cur + rows;
    const auto fold_into = [&](float *into, uint32_t c) {
        const size_t src = size_t(c) * rows;
        for (uint32_t r = 0; r < rows; ++r) into[r] = heights[src + r] - transverse_gap[r];
    };
    fold_into(prev, columns - 1);
    fold_into(cur, 0);
    fold_into(next, 1 % columns);
    float crest = -std::numeric_limits<float>::max();
    for (uint32_t c = 0; c < count * element_columns; ++c) {
        const uint32_t e = c / element_columns;
        // A flank sample belongs to a neighbouring summit and has no spring of its own, so an element's onset stiffness is its tallest asperity's.
        const float lift = e < element_lift.size() ? element_lift[e] : 0.f;
        for (uint32_t r = 0; r < rows; ++r) {
            crest = std::max(crest, cur[r]);
            if (!IsSummit(prev, cur, next, r, rows)) continue;
            const auto curvature = CombinedCurvature(SummitCurvature(prev, cur, next, r, rows, column_spacing), 0);
            out.Element.push_back(e);
            out.Height.push_back(cur[r] + lift);
            out.Stiffness.push_back(float(ContactStiffness(inv_effective_modulus, curvature)));
        }
        if (c % element_columns == element_columns - 1) {
            out.Crest[e] = std::max(out.Crest[e], crest + lift);
            crest = -std::numeric_limits<float>::max();
        }
        std::swap(prev, cur);
        std::swap(cur, next);
        fold_into(next, (c + 2) % columns);
    }
}

void MergeElementSummits(ElementSummits &into, const ElementSummits &from) {
    if (from.Crest.empty()) return;
    into.Width = from.Width;
    into.InvModulus = from.InvModulus;
    into.Crest.resize(std::max(into.Crest.size(), from.Crest.size()), -std::numeric_limits<float>::max());
    for (size_t e = 0; e < from.Crest.size(); ++e) into.Crest[e] = std::max(into.Crest[e], from.Crest[e]);
    into.Element.insert(into.Element.end(), from.Element.begin(), from.Element.end());
    into.Height.insert(into.Height.end(), from.Height.begin(), from.Height.end());
    into.Stiffness.insert(into.Stiffness.end(), from.Stiffness.begin(), from.Stiffness.end());
}

ElementSprings BuildElementSprings(const ElementSummits &summits, const SubCutoffRoughness &sub, double engagement_max, uint32_t knots) {
    ElementSprings out;
    const auto count = uint32_t(summits.Crest.size());
    if (count == 0 || knots < 2) return out;
    const DepthSplit split{sub, summits.InvModulus};

    InitSpringTables(out, summits.Crest, summits.Width, engagement_max, count, knots);
    out.Crest.resize(count);

    // The gathered summits bucketed into their elements, so one element's population is contiguous however many realizations it was gathered from.
    std::vector<uint32_t> start(size_t(count) + 1, 0);
    for (const uint32_t e : summits.Element) ++start[size_t(e) + 1];
    for (uint32_t e = 0; e < count; ++e) start[size_t(e) + 1] += start[e];
    // A summit's height, the Hertz constant its curvature gives it, k = (4/3) E* / sqrt(kappa), and the depth scale its contact splits at.
    struct Summit {
        float Height;
        float Stiffness;
        double Scale;
    };
    std::vector<Summit> sorted(summits.Element.size());
    {
        auto fill = start;
        for (size_t i = 0; i < summits.Element.size(); ++i) {
            sorted[fill[summits.Element[i]]++] = {summits.Height[i], summits.Stiffness[i], split.Scale(summits.Stiffness[i])};
        }
    }
    for (uint32_t e = 0; e < count; ++e) {
        // The element's curve is the Hertz sum over its summits at the full asperity constant.
        const std::span<Summit> sample{sorted.data() + start[e], start[e + 1] - start[e]};
        // Descending, so a knot sums only the summits already bearing.
        // The stiffness breaks a tie in height, so the order is a property of the population.
        std::ranges::sort(sample, [](const Summit &a, const Summit &b) {
            return a.Height != b.Height ? a.Height > b.Height : a.Stiffness > b.Stiffness;
        });
        const float crest = summits.Crest[e];
        out.Crest[e] = crest;

        auto *f = out.Force.data() + size_t(e) * knots;
        auto *variance = out.ForceVariance.data() + size_t(e) * knots;
        auto *moment = out.FlankMoment.data() + size_t(e) * knots;
        auto *bearing_count = out.BearingCount.data() + size_t(e) * knots;
        auto *bearing_width = out.BearingWidth.data() + size_t(e) * knots;
        size_t bearing = 0;
        for (uint32_t j = 1; j < knots; ++j) {
            const float floor_height = crest - out.Engagement[j];
            while (bearing < sample.size() && sample[bearing].Height > floor_height) ++bearing;
            double sum = 0, square = 0, flank = 0, width = 0;
            for (size_t i = 0; i < bearing; ++i) {
                const double k = double(sample[i].Stiffness);
                const auto [depth, share] = split.Split(double(sample[i].Height) - floor_height, k, sample[i].Scale);
                const double root = std::sqrt(depth);
                const double force = k * depth * root;
                sum += force;
                square += force * force;
                // One Hertz summit's slope^2 / force, which leaves its stiffness and depth alone: (1.5 k sqrt(d))^2 / (k d sqrt(d)) is 2.25 k / sqrt(d).
                // The roughness inside the contact takes its share of the slope with it.
                const double term = 2.25 * k * share * share / root;
                flank += term;
                const size_t bin = size_t(j) * FlankSpectrumBins + FlankSpectrumBin(depth, out.Engagement[j]);
                out.FlankSpectrum[bin] += float(term);
                out.FlankSpectrumForce[bin] += float(force);
                // The contact's width, dF/dd = 2 a E* read at the depth the contact takes.
                width += 1.5 * k * root * summits.InvModulus;
            }
            f[j] = float(sum);
            variance[j] = float(square);
            moment[j] = float(flank);
            bearing_count[j] = float(bearing);
            bearing_width[j] = float(width);
        }
        FillCurveTail(out, e);
    }
    RefreshCrestBlocks(out);
    return out;
}

ElementSprings BuildEnsembleSprings(
    std::span<const EnsembleBand> bands, float spacing, uint32_t columns, uint32_t rows,
    std::span<const float> transverse_gap, double body_curvature, uint32_t element_columns,
    uint32_t strip_realizations, double inv_effective_modulus,
    const SubCutoffRoughness &sub, double engagement_max, uint32_t knots, std::vector<float> crests
) {
    ElementSprings out;
    const auto count = uint32_t(crests.size());
    if (count == 0 || knots < 2 || rows < 3 || spacing <= 0 || element_columns == 0) return out;

    // The summed bands' lattice covariance: independent draws add in variance, so the field's law is the sigma-squared-weighted mix of the bands'.
    double variance = 0;
    LatticeCovariance cov{};
    for (const auto &band : bands) {
        if (band.Sigma <= 0) continue;
        const auto one = RoughnessLatticeCovariance(band.Correlation, band.Slope, band.Cutoff, spacing, columns, rows);
        const double v = double(band.Sigma) * band.Sigma;
        for (size_t i = 0; i < 5; ++i) {
            for (size_t j = 0; j < 5; ++j) cov[i][j] += v * one[i][j];
        }
        variance += v;
    }
    if (variance <= 0) return out;
    for (auto &row : cov) {
        for (auto &c : row) c /= variance;
    }
    const double sigma = std::sqrt(variance);

    double crest_sum = 0, crest_min = 1e300, crest_max = -1e300;
    for (const float c : crests) {
        crest_sum += c;
        crest_min = std::min(crest_min, double(c));
        crest_max = std::max(crest_max, double(c));
    }
    const double xi_crest = crest_sum / count / sigma;
    // A crest reference at or below the field's mean is not a crest, so the population law has nothing to anchor to and the realized build is used instead.
    if (!(xi_crest > 1)) return out;
    constexpr double XLo{-4};
    constexpr uint32_t CeilingBins{16};
    const double ceiling_lo = std::max(crest_min / sigma, XLo + 0.5);
    const double ceiling_span = std::max(crest_max / sigma - ceiling_lo, 0.0);
    std::array<double, CeilingBins> ceiling_weight{}, ceiling_sum{};
    for (const float c : crests) {
        const double xi = std::clamp(double(c) / sigma, ceiling_lo, ceiling_lo + ceiling_span);
        const auto b = ceiling_span > 0 ? std::min(CeilingBins - 1, uint32_t((xi - ceiling_lo) / ceiling_span * CeilingBins)) : 0u;
        ceiling_weight[b] += 1;
        ceiling_sum[b] += xi;
    }

    // The eight neighbour offsets, the four axial first so the Laplacian reads a prefix.
    // The conditional law given the centre height has neighbours of mean rho_i x with an x-free covariance, Cholesky-factored for the deviate draw.
    constexpr int Off[8][2]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    double rho[8], s[8][8];
    for (uint32_t i = 0; i < 8; ++i) rho[i] = cov[Off[i][0] + 2][Off[i][1] + 2];
    for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
            s[i][j] = cov[Off[i][0] - Off[j][0] + 2][Off[i][1] - Off[j][1] + 2] - rho[i] * rho[j];
        }
    }
    double l[8][8]{};
    for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            double sum = s[i][j];
            for (uint32_t k = 0; k < j; ++k) sum -= l[i][k] * l[j][k];
            if (i == j) l[i][i] = std::sqrt(std::max(sum, 0.0));
            else l[i][j] = l[j][j] > 0 ? sum / l[j][j] : 0;
        }
    }
    const double rho_axial = rho[0] + rho[1] + rho[2] + rho[3];
    const double kappa_scale = sigma / (double(spacing) * spacing);

    const double x_hi = crest_max / sigma;
    const auto nodes = uint32_t(std::ceil((x_hi - XLo) / 0.02));
    const double dx = (x_hi - XLo) / nodes;
    constexpr uint32_t KappaBins{32};
    const double axial_var = [&] {
        double v = 0;
        for (uint32_t i = 0; i < 4; ++i) {
            for (uint32_t j = 0; j < 4; ++j) v += s[i][j];
        }
        return v;
    }();
    const double kappa_max = kappa_scale * ((4 - rho_axial) * x_hi + 4 * std::sqrt(std::max(axial_var, 0.0)));
    if (!(kappa_max > 0)) return out;
    std::vector<double> cell_weight(size_t(nodes) * KappaBins, 0.0), cell_kappa(size_t(nodes) * KappaBins, 0.0);
    constexpr uint32_t Samples{400'000};
    {
        uint64_t state{0x8f1bbcdcbfa53e0bull};
        double held[2];
        uint32_t held_count{0};
        const auto normal = [&] {
            if (held_count == 0) {
                const double u = (double(SplitMix64(state) >> 11) + 0.5) * 0x1p-53;
                const double v = double(SplitMix64(state) >> 11) * 0x1p-53 * 2 * std::numbers::pi;
                const double r = std::sqrt(-2 * std::log(u));
                held[0] = r * std::cos(v);
                held[1] = r * std::sin(v);
                held_count = 2;
            }
            return held[--held_count];
        };
        for (uint32_t sample = 0; sample < Samples; ++sample) {
            double z[8], e[8];
            for (auto &v : z) v = normal();
            double threshold = -1e300;
            for (uint32_t i = 0; i < 8; ++i) {
                double sum = 0;
                for (uint32_t j = 0; j <= i; ++j) sum += l[i][j] * z[j];
                e[i] = sum;
                threshold = std::max(threshold, sum / (1 - rho[i]));
            }
            if (threshold >= x_hi) continue;
            const double axial = e[0] + e[1] + e[2] + e[3];
            const auto first = threshold <= XLo ? 0u : uint32_t(std::ceil((threshold - XLo) / dx - 0.5));
            for (uint32_t n = first; n < nodes; ++n) {
                const double x = XLo + (double(n) + 0.5) * dx;
                const double kappa = kappa_scale * ((4 - rho_axial) * x - axial);
                const auto bin = std::min(KappaBins - 1, uint32_t(std::max(kappa, 0.0) / kappa_max * KappaBins));
                const size_t at = size_t(n) * KappaBins + bin;
                cell_weight[at] += 1;
                cell_kappa[at] += kappa;
            }
        }
    }

    InitSpringTables(out, crests, float(element_columns) * spacing, engagement_max, 1, knots);
    out.Crest = std::move(crests);

    constexpr uint32_t QuarterBins{FlankSpectrumBins * 4};
    std::vector<double> ugrid;
    ugrid.reserve(size_t(knots) * 3);
    ugrid.push_back(0);
    for (uint32_t j = 1; j < knots; ++j) {
        const double e = out.Engagement[j];
        if (const double prev = out.Engagement[j - 1]; prev > 0) {
            const double step = std::cbrt(e / prev);
            ugrid.push_back(prev * step);
            ugrid.push_back(prev * step * step);
        }
        ugrid.push_back(e);
    }
    const auto nu = uint32_t(ugrid.size());
    // One family per ceiling bin, each anchored at and truncated below its own crest height.
    std::vector<double> site_force(size_t(CeilingBins) * nu, 0.0), site_square(size_t(CeilingBins) * nu, 0.0), site_moment(size_t(CeilingBins) * nu, 0.0);
    std::vector<double> site_width(size_t(CeilingBins) * nu, 0.0), site_count(size_t(CeilingBins) * nu, 0.0);
    std::vector<double> site_spec_moment(size_t(CeilingBins) * nu * QuarterBins, 0.0), site_spec_force(size_t(CeilingBins) * nu * QuarterBins, 0.0);
    const DepthSplit split{sub, inv_effective_modulus};
    const double phi_norm = 1 / std::sqrt(2 * std::numbers::pi);
    for (uint32_t n = 0; n < nodes; ++n) {
        const double x = XLo + (double(n) + 0.5) * dx;
        const double site = phi_norm * std::exp(-0.5 * x * x) * dx / Samples;
        for (uint32_t b = 0; b < KappaBins; ++b) {
            const size_t at = size_t(n) * KappaBins + b;
            if (!(cell_weight[at] > 0)) continue;
            const double weight = cell_weight[at] * site;
            const double kappa = cell_kappa[at] / cell_weight[at] + body_curvature;
            const double k = ContactStiffness(inv_effective_modulus, CombinedCurvature(kappa, 0));
            const double scale = split.Scale(k);
            for (uint32_t cb = 0; cb < CeilingBins; ++cb) {
                if (!(ceiling_weight[cb] > 0)) continue;
                const double ceiling = ceiling_sum[cb] / ceiling_weight[cb];
                if (x >= ceiling) continue;
                const double delta = sigma * (ceiling - x);
                const size_t family = size_t(cb) * nu;
                for (uint32_t u = 1; u < nu; ++u) {
                    const double raw = ugrid[u] - delta;
                    if (!(raw > 0)) continue;
                    const auto [depth, share] = split.Split(raw, k, scale);
                    const double root = std::sqrt(depth);
                    const double force = k * depth * root;
                    const double moment = 2.25 * k * share * share / root;
                    site_force[family + u] += weight * force;
                    site_square[family + u] += weight * force * force;
                    site_moment[family + u] += weight * moment;
                    site_width[family + u] += weight * 1.5 * k * root * inv_effective_modulus;
                    site_count[family + u] += weight;
                    const auto sub_bin = std::min(QuarterBins - 1, uint32_t(std::max(0.0, 4 * std::log2(ugrid[u] / depth))));
                    site_spec_moment[(family + u) * QuarterBins + sub_bin] += weight * moment;
                    site_spec_force[(family + u) * QuarterBins + sub_bin] += weight * force;
                }
            }
        }
    }

    const double crest_kappa = [&] {
        const auto node = std::min(nodes - 1, uint32_t(std::max((xi_crest - XLo) / dx - 0.5, 0.0)));
        double w = 0, ks = 0;
        for (uint32_t b = 0; b < KappaBins; ++b) {
            const size_t at = size_t(node) * KappaBins + b;
            w += cell_weight[at];
            ks += cell_kappa[at];
        }
        return (w > 0 ? ks / w : kappa_scale * (4 - rho_axial) * xi_crest) + body_curvature;
    }();
    const double crest_stiffness = ContactStiffness(inv_effective_modulus, CombinedCurvature(crest_kappa, 0));
    const double crest_scale = split.Scale(crest_stiffness);

    const bool flat = body_curvature == 0 && std::ranges::all_of(transverse_gap, [](float g) { return g == 0.f; });
    const double site_scale = double(element_columns) * strip_realizations;
    const auto read_site = [&](uint32_t cb, double u, uint32_t j, double row_scale) {
        if (!(u > 0)) return;
        const auto hi = uint32_t(std::ranges::upper_bound(ugrid, u) - ugrid.begin());
        const auto i1 = std::min(hi, nu - 1), i0 = i1 - 1;
        const double span = ugrid[i1] - ugrid[i0];
        const double t = span > 0 ? std::clamp((u - ugrid[i0]) / span, 0.0, 1.0) : 0.0;
        const size_t f0 = size_t(cb) * nu + i0, f1 = size_t(cb) * nu + i1;
        const double f = site_force[f0] + t * (site_force[f1] - site_force[f0]);
        const double sq = site_square[f0] + t * (site_square[f1] - site_square[f0]);
        const double m = site_moment[f0] + t * (site_moment[f1] - site_moment[f0]);
        const double w = site_width[f0] + t * (site_width[f1] - site_width[f0]);
        const double c = site_count[f0] + t * (site_count[f1] - site_count[f0]);
        out.Force[j] += float(row_scale * f);
        out.ForceVariance[j] += float(row_scale * sq);
        out.FlankMoment[j] += float(row_scale * m);
        out.BearingWidth[j] += float(row_scale * w);
        out.BearingCount[j] += float(row_scale * c);
        const auto slide = uint32_t(std::lround(4 * std::log2(double(out.Engagement[j]) / u)));
        const double spectrum_scale = row_scale * count;
        for (uint32_t q = 0; q < QuarterBins; ++q) {
            const double sm = site_spec_moment[f0 * QuarterBins + q] + t * (site_spec_moment[f1 * QuarterBins + q] - site_spec_moment[f0 * QuarterBins + q]);
            const double sf = site_spec_force[f0 * QuarterBins + q] + t * (site_spec_force[f1 * QuarterBins + q] - site_spec_force[f0 * QuarterBins + q]);
            if (!(sm > 0) && !(sf > 0)) continue;
            const auto bin = std::min(FlankSpectrumBins - 1, (q + slide) / 4);
            out.FlankSpectrum[size_t(j) * FlankSpectrumBins + bin] += float(spectrum_scale * sm);
            out.FlankSpectrumForce[size_t(j) * FlankSpectrumBins + bin] += float(spectrum_scale * sf);
        }
    };
    for (uint32_t j = 1; j < knots; ++j) {
        const double e = out.Engagement[j];
        for (uint32_t cb = 0; cb < CeilingBins; ++cb) {
            if (!(ceiling_weight[cb] > 0)) continue;
            // Elements share the one composed curve, so each ceiling's family takes its own share of the element population.
            const double share = ceiling_weight[cb] / count;
            if (flat) {
                read_site(cb, e, j, site_scale * rows * share);
            } else {
                for (uint32_t r = 0; r < rows; ++r) read_site(cb, e - double(transverse_gap[r]), j, site_scale * share);
            }
        }
        const auto [depth, share] = split.Split(e, crest_stiffness, crest_scale);
        const double root = std::sqrt(depth);
        const double force = crest_stiffness * depth * root;
        const double moment = 2.25 * crest_stiffness * share * share / root;
        out.Force[j] += float(force);
        out.ForceVariance[j] += float(force * force);
        out.FlankMoment[j] += float(moment);
        out.BearingWidth[j] += float(1.5 * crest_stiffness * root * inv_effective_modulus);
        out.BearingCount[j] += 1;
        const size_t bin = size_t(j) * FlankSpectrumBins + FlankSpectrumBin(depth, e);
        out.FlankSpectrum[bin] += float(moment * count);
        out.FlankSpectrumForce[bin] += float(force * count);
    }
    FillCurveTail(out, 0);
    RefreshCrestBlocks(out);
    return out;
}

ElementSprings BuildElementSprings(
    std::span<const float> heights, uint32_t columns, uint32_t rows, std::span<const float> transverse_gap,
    uint32_t element_columns, float column_spacing, double inv_effective_modulus,
    const SubCutoffRoughness &sub, double engagement_max, uint32_t knots
) {
    ElementSummits summits;
    GatherElementSummits(summits, heights, columns, rows, transverse_gap, {}, element_columns, column_spacing, inv_effective_modulus);
    return BuildElementSprings(summits, sub, engagement_max, knots);
}

/***** Interface junctions and the asperity bed *****/

double FlankSineCube(double slope) {
    if (!(slope > 0)) return 0;
    constexpr uint32_t N{257};
    const double hi = 8 * slope, h = 2 * hi / (N - 1);
    double sum = 0;
    for (uint32_t i = 0; i < N; ++i) {
        const double x = -hi + i * h, s = x / std::sqrt(1 + x * x);
        const double w = std::exp(-0.5 * (x / slope) * (x / slope)) / (slope * std::sqrt(2 * std::numbers::pi));
        sum += (i == 0 || i == N - 1 ? 0.5 : 1.0) * std::abs(s * s * s) * w;
    }
    return sum * h;
}

uint32_t FlankJunctionSpread(
    double sine_cube, double normal_force, double flank_modulus,
    std::span<const float> moment_share, std::span<const float> force_share, std::span<FlankJunctionBin> out
) {
    const size_t bins = std::min({moment_share.size(), force_share.size(), out.size()});
    uint32_t count = 0;
    for (size_t b = 0; b < bins; ++b) {
        const double force = normal_force * double(force_share[b]), moment = flank_modulus * double(moment_share[b]);
        const double stiffness = force > 0 && moment > 0 ? FlankJunctionStiffness(sine_cube, force, moment) : 0.0;
        if (stiffness > 0) out[count++] = {float(stiffness), force_share[b]};
    }
    return count;
}

double AsperityPressure(double inv_effective_modulus, double profile_slope_rms) {
    return inv_effective_modulus > 0 ? profile_slope_rms / (std::numbers::sqrt2 * inv_effective_modulus) : 0.0;
}

double RealContactArea(double normal_force, double bound_area, double inv_effective_modulus, double profile_slope_rms) {
    const double pressure = AsperityPressure(inv_effective_modulus, profile_slope_rms);
    const double load = std::max(normal_force, 0.0);
    if (pressure <= 0 || load <= 0) return 0;
    // The error function's argument overflows to full contact long before the exponential in it underflows.
    // A contact with no bounding region is the load over the pressure outright.
    if (!(bound_area > 0) || !std::isfinite(bound_area)) return load / pressure;
    return bound_area * std::erf(std::sqrt(std::numbers::pi) * load / (2 * bound_area * pressure));
}

double HurstExponent(double spectral_slope) { return std::clamp(-(1.0 + spectral_slope) / 2.0, 1e-3, 1 - 1e-3); }

double RollOffWavevector(double correlation_length) { return 2 * std::numbers::pi / std::max(correlation_length, 1e-9); }

namespace {
// Persson's g(x) = x^-H (x^2 - x^2H)^-1/2 times dx/dt, at x = 1 + t^2.
// x^2 - x^2H loses every significant digit near x = 1, so it is factored as x^2H * expm1((2-2H) log x).
double SeparationWeight(double t, double hurst) {
    const double log_x = std::log1p(t * t);
    const double gap = std::expm1((2 - 2 * hurst) * log_x);
    return gap > 0 ? 2 * t * std::exp(-2 * hurst * log_x) / std::sqrt(gap) : 0.0;
}

// The logarithm Eq. 17 weights by, at the same point and factored the same way.
double SeparationLogArg(double t, double hurst) {
    const double arg = hurst / (2 * (1 - hurst)) * std::expm1(2 * (1 - hurst) * std::log1p(t * t));
    return arg > 0 ? std::log(arg) : 0.0;
}

// Pressure at full contact, where Eq. 18's exponential is 1 and the roughness has flattened out.
double FullContactPressure(double roughness, double roll_off_wavevector, double inv_effective_modulus, const SeparationCoefficients &c) {
    return inv_effective_modulus > 0 ? c.Beta * roll_off_wavevector * roughness / inv_effective_modulus : 0.0;
}
} // namespace

SeparationCoefficients PerssonSeparationCoefficients(double hurst, double band_ratio, double gamma) {
    if (band_ratio <= 1 || hurst <= 0 || hurst >= 1 || gamma <= 0) return {};
    const double t_max = std::sqrt(band_ratio - 1);
    constexpr int Half = 48;
    constexpr double HalfWidth = 3.2, Step = HalfWidth / Half;
    double weight_sum = 0, log_sum = 0;
    for (int i = -Half; i <= Half; ++i) {
        const double sinh_term = std::numbers::pi / 2 * std::sinh(double(i) * Step);
        const double t = 0.5 * t_max * (std::tanh(sinh_term) + 1);
        if (t <= 0 || t >= t_max) continue;
        const double cosh_term = std::cosh(sinh_term);
        const double node = std::numbers::pi / 2 * std::cosh(double(i) * Step) / (cosh_term * cosh_term) * SeparationWeight(t, hurst);
        weight_sum += node;
        log_sum += node * SeparationLogArg(t, hurst);
    }
    if (weight_sum <= 0) return {};
    // The quadrature's scale is common to both sums, so it cancels out of Beta and applies only to Alpha.
    const double inv_alpha = std::sqrt(2 * hurst * (1 - hurst) / std::numbers::pi) * gamma * 0.5 * t_max * Step * weight_sum;
    return {.Alpha = inv_alpha > 0 ? 1 / inv_alpha : 1.0, .Beta = std::exp(log_sum / (2 * weight_sum))};
}

double ConformalPressure(double separation, double roughness, double roll_off_wavevector, double inv_effective_modulus, const SeparationCoefficients &c) {
    if (roughness <= 0) return 0;
    return FullContactPressure(roughness, roll_off_wavevector, inv_effective_modulus, c) * std::exp(-c.Alpha * std::max(separation, 0.0) / roughness);
}

double ConformalSeparation(double pressure, double roughness, double roll_off_wavevector, double inv_effective_modulus, const SeparationCoefficients &c) {
    const double full = FullContactPressure(roughness, roll_off_wavevector, inv_effective_modulus, c);
    if (pressure <= 0 || full <= 0 || roughness <= 0 || c.Alpha <= 0) return 0;
    return std::max(roughness / c.Alpha * std::log(full / pressure), 0.0);
}

double ConformalPatchWidth(double combined_curvature, double roughness, const SeparationCoefficients &c) {
    if (combined_curvature <= 0 || roughness <= 0 || c.Alpha <= 0) return 0;
    return std::sqrt(roughness / (c.Alpha * combined_curvature)); // s^2 = R*u0, u0 = roughness/Alpha
}

double ConformalStiffness(double normal_force, double roughness, const SeparationCoefficients &c) {
    return roughness > 0 ? c.Alpha * std::max(normal_force, 0.0) / roughness : 0.0;
}

double BedHeightIntegral(double lambda, double power) {
    // Integrate from -lambda to eight standard deviations with the midpoint rule.
    constexpr int Steps = 256;
    constexpr double Limit = 8;
    const double lo = std::max(-lambda, -Limit);
    if (lo >= Limit) return 0;
    const double step = (Limit - lo) / Steps;
    const double norm = 1 / std::sqrt(2 * std::numbers::pi);
    double sum = 0;
    for (int n = 0; n < Steps; ++n) {
        const double z = lo + (double(n) + 0.5) * step;
        sum += std::pow(lambda + z, power) * std::exp(-0.5 * z * z);
    }
    return sum * step * norm;
}

double SolveBedSeparation(double normal_force, double spot_count, double spot_stiffness, double height_rms, double power) {
    if (spot_count <= 0 || spot_stiffness <= 0 || height_rms <= 0) return 0;
    // Each spot bears k*delta^(3/2), so the bed's mean load is the integral times the load one spot bears at a separation of the height spread.
    const double unit_load = spot_count * spot_stiffness * height_rms * std::sqrt(height_rms);
    const double target = std::max(normal_force, 0.0) / unit_load;
    if (target <= 0) return -8;
    // The integral rises monotonically in lambda, so the separation is bracketed instead of iterated on a derivative that vanishes once every spot has parted.
    double lo = -8, hi = 8;
    for (int i = 0; i < 60; ++i) {
        const double mid = 0.5 * (lo + hi);
        (BedHeightIntegral(mid, power) < target ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

double BedLoad(const AsperityBed &b) {
    return b.TotalSpots * b.SpotStiffness * b.HeightRms * std::sqrt(b.HeightRms) * BedHeightIntegral(b.Separation, 1.5);
}

double BedStiffness(const AsperityBed &b) {
    return 1.5 * b.TotalSpots * b.SpotStiffness * std::sqrt(b.HeightRms) * BedHeightIntegral(b.Separation, 0.5);
}

double BedContactArea(const AsperityBed &b) {
    return std::numbers::pi * b.TotalSpots * b.SpotRadius * b.HeightRms * BedHeightIntegral(b.Separation, 1.0);
}

double BedSurfaceShare(double inv_effective_modulus, double bound_area, const AsperityBed &b) {
    if (!(bound_area > 0) || inv_effective_modulus <= 0) return 1;
    const double bulk = 2 * std::sqrt(bound_area / std::numbers::pi) / inv_effective_modulus;
    const double interfacial = BedStiffness(b);
    return interfacial > 0 ? bulk / (bulk + interfacial) : 1;
}

namespace {
// The two integrals a cell reads every sample, tabulated over the range the clamp at zero makes nonlinear.
// Outside that range they have closed forms and the table goes unread.
constexpr int BedTableSize = 1024;
constexpr float BedTableLimit = 8;

struct BedTable {
    std::array<float, BedTableSize + 1> Load{}, Variance{}, LinearLoad{};
    // Cumulative integrals of the load factors from the low edge, for the stored elastic potential.
    std::array<float, BedTableSize + 1> Potential{}, LinearPotential{};
    BedTable() {
        for (int i = 0; i <= BedTableSize; ++i) {
            const double lambda = -BedTableLimit + 2.0 * BedTableLimit * i / BedTableSize;
            const double load = BedHeightIntegral(lambda, 1.5);
            Load[i] = float(load);
            // The second moment of one spot's load less the square of its first.
            Variance[i] = float(std::max(BedHeightIntegral(lambda, 3.0) - load * load, 0.0));
            LinearLoad[i] = float(BedHeightIntegral(lambda, 1.0));
        }
        constexpr double step = 2.0 * BedTableLimit / BedTableSize;
        double pm = 0, plm = 0;
        for (int i = 1; i <= BedTableSize; ++i) {
            pm += 0.5 * (double(Load[i - 1]) + Load[i]) * step;
            plm += 0.5 * (double(LinearLoad[i - 1]) + LinearLoad[i]) * step;
            Potential[i] = float(pm);
            LinearPotential[i] = float(plm);
        }
    }
};
const BedTable &Table() {
    static const BedTable table;
    return table;
}

// Where a height ratio falls in the tables, in table steps from the low limit.
float BedTableIndex(float lambda) { return (lambda + BedTableLimit) * (BedTableSize / (2 * BedTableLimit)); }

// One table's value at that index, linear between its entries.
float ReadBedInterp(const std::array<float, BedTableSize + 1> &values, float t) {
    const auto i = int(t);
    return values[i] + (t - float(i)) * (values[i + 1] - values[i]);
}

// One table's value at `lambda`, continued past the table's limit at `power`.
float ReadBedTable(const std::array<float, BedTableSize + 1> &values, float lambda, float power) {
    if (lambda <= -BedTableLimit) return 0;
    if (lambda >= BedTableLimit) return std::pow(lambda, power);
    return ReadBedInterp(values, BedTableIndex(lambda));
}
} // namespace

float BedLoadFactor(float lambda) { return ReadBedTable(Table().Load, lambda, 1.5f); }
float BedLinearLoadFactor(float lambda) { return ReadBedTable(Table().LinearLoad, lambda, 1.f); }

float BedVarianceFactor(float lambda) {
    if (lambda <= -BedTableLimit) return 0;
    // Once every spot bears, the load is smooth in the height and its variance is the slope squared.
    if (lambda >= BedTableLimit) return 2.25f * lambda;
    return ReadBedInterp(Table().Variance, BedTableIndex(lambda));
}

float BedPotentialFactor(float lambda) {
    if (lambda <= -BedTableLimit) return 0;
    // Beyond the table the load factor is its argument to the 3/2, so the integral continues in closed form.
    if (lambda >= BedTableLimit)
        return Table().Potential[BedTableSize] + (std::pow(lambda, 2.5f) - std::pow(float(BedTableLimit), 2.5f)) / 2.5f;
    return ReadBedInterp(Table().Potential, BedTableIndex(lambda));
}

float BedLoadAt(float engagement, float spread) {
    // Beyond the table every asperity bears, so the mean is the engagement's 3/2 power.
    // That branch is evaluated in meters, which stays finite however small the spread is.
    if (engagement >= spread * BedTableLimit) return engagement > 0 ? engagement * std::sqrt(engagement) : 0.f;
    if (engagement <= -spread * BedTableLimit) return 0;
    return spread * std::sqrt(spread) * ReadBedTable(Table().Load, engagement / spread, 1.5f);
}

float BedPotentialAt(float engagement, float spread) {
    if (engagement <= -spread * BedTableLimit) return 0;
    if (engagement < spread * BedTableLimit)
        return spread * spread * std::sqrt(spread) * ReadBedInterp(Table().Potential, BedTableIndex(engagement / spread));
    const float above = engagement * engagement * std::sqrt(engagement) / 2.5f;
    return spread * spread * std::sqrt(spread) * (Table().Potential[BedTableSize] - std::pow(float(BedTableLimit), 2.5f) / 2.5f) + above;
}

float BedLinearPotentialFactor(float lambda) {
    if (lambda <= -BedTableLimit) return 0;
    if (lambda >= BedTableLimit)
        return Table().LinearPotential[BedTableSize] + (lambda * lambda - float(BedTableLimit * BedTableLimit)) / 2;
    return ReadBedInterp(Table().LinearPotential, BedTableIndex(lambda));
}

/***** The audio-thread render *****/

namespace {
// A voice ends after this long without a fresh contact report.
constexpr float MaxVoiceIdleSeconds{0.1f};
// Ceiling on a cell's absolute engagement, m.
// A contact whose geometry changes under it can report an approach from far outside any physical regime, where the spot law's powers overflow float.
// Clamping the engagement saturates force, potential and gradient together, and the ceiling sits far beyond every real contact.
constexpr float MaxEngagement{1.f};

std::optional<uint32_t> FindIndex(const auto &v, const auto &value) {
    const auto it = std::ranges::find(v, value);
    return it != v.end() ? std::optional{uint32_t(std::ranges::distance(v.begin(), it))} : std::nullopt;
}

// Drop voice `i` from each parallel column, swapping the last into its place.
void RemoveVoice(SurfaceAudioState &surface, uint32_t i) {
    surface.VoiceId[i] = surface.VoiceId.back();
    surface.VoiceObject[i] = surface.VoiceObject.back();
    surface.VoiceState[i] = surface.VoiceState.back();
    surface.VoiceCarry[i] = surface.VoiceCarry.back();
    surface.VoiceId.pop_back();
    surface.VoiceObject.pop_back();
    surface.VoiceState.pop_back();
    surface.VoiceCarry.pop_back();
}
} // namespace

void SurfaceAdoptVoices(ModalAudio &m, ModalBank &b, uint32_t frame_count) {
    auto &surface = Surface(m);
    const auto *set = surface.PublishedVoices.load(std::memory_order_acquire);
    if (set && set->Frame != surface.AdoptedVoiceFrame) {
        surface.AdoptedVoiceFrame = set->Frame;
        surface.VoiceSetIdleSamples = 0;
    } else {
        surface.VoiceSetIdleSamples += frame_count;
    }
    const bool reporting = set != nullptr && surface.VoiceSetIdleSamples <= uint32_t(b.SampleRate * MaxVoiceIdleSeconds);
    const auto named = [&](uint64_t id) {
        return reporting && std::ranges::contains(set->Voices, id, &VoiceSet::Voice::Id);
    };
    for (uint32_t v = uint32_t(surface.VoiceId.size()); v-- > 0;) {
        if (!named(surface.VoiceId[v])) RemoveVoice(surface, v);
    }
    if (reporting) {
        for (const auto &voice : set->Voices) {
            // A set built against a bank that has since been replaced can name slots this one does not have.
            if (voice.Object >= b.Entities.size()) continue;
            if (const auto v = FindIndex(surface.VoiceId, voice.Id)) {
                surface.VoiceObject[*v] = voice.Object;
                const auto &prev = surface.VoiceState[*v];
                const auto &next = voice.State;
                const float prev_depth = prev.CellSpread > 0 ? prev.CellSpread : std::abs(prev.StaticPenetration);
                const float next_depth = next.CellSpread > 0 ? next.CellSpread : std::abs(next.StaticPenetration);
                bool rebase = next.SpotCount != prev.SpotCount || next.SpringIndex != prev.SpringIndex || next.SpringBins != prev.SpringBins ||
                    std::abs(next.StaticPenetration - prev.StaticPenetration) > std::max(prev_depth, next_depth);
                for (uint32_t i = 0; i < SustainedState::TrackCount && !rebase; ++i) {
                    const auto &pt = prev.Tracks[i];
                    const auto &nt = next.Tracks[i];
                    rebase = nt.Index != pt.Index || nt.Window > 2 * pt.Window || pt.Window > 2 * nt.Window;
                }
                if (rebase) surface.VoiceCarry[*v].Rebase = true;
                surface.VoiceState[*v] = voice.State;
            } else {
                surface.VoiceId.push_back(voice.Id);
                surface.VoiceObject.push_back(voice.Object);
                surface.VoiceState.push_back(voice.State);
                surface.VoiceCarry.emplace_back();
            }
            b.Ringing[voice.Object] = 1;
        }
    }
    // Published so the main thread never repoints a slot a live voice reads.
    // Every slot a voice reads is named, the turnover track included.
    uint64_t mask = 0, spring_mask = 0, sweep_mask = 0;
    for (const auto &st : surface.VoiceState) {
        for (const auto &t : st.Tracks) {
            if (t.Index >= 0) mask |= 1ull << uint32_t(t.Index);
        }
        if (st.Turnover.Index >= 0) mask |= 1ull << uint32_t(st.Turnover.Index);
        if (st.SpringIndex >= 0) spring_mask |= 1ull << uint32_t(st.SpringIndex);
        if (st.SweepIndex >= 0) sweep_mask |= 1ull << uint32_t(st.SweepIndex);
    }
    surface.VoiceTrackMask.store(mask, std::memory_order_release);
    surface.VoiceSpringMask.store(spring_mask, std::memory_order_release);
    surface.VoiceSweepMask.store(sweep_mask, std::memory_order_release);
}

namespace {
struct VoiceForce {
    float Normal; // The fluctuation about the load, which drives the modes and accelerates the body alike.
    std::array<float, 2> Geometric; // Per surface, in the contact's surface order, matching SustainedState::SweepDir.
    float Frictional;
    float Transverse; // The junction's force along normal cross slip, the interface's second in-plane axis.
};

// Shock statistics of the contact the renderer runs, reported under CONTACT_REPORT.
// A cell taking up load again after parting is one shock, and a sample where the whole contact bears nothing is time the body spends clear of the surface.
struct ContactReport {
    std::atomic<uint64_t> Samples{0}, Free{0}, Shocks{0}, CellSamples{0}, CellBearing{0};
    // Junction engagements the turnover shot process spawned, summed over bins and voices.
    std::atomic<uint64_t> TurnoverEvents{0};
    std::atomic<uint64_t> Peaks{0}, UnderLoad{0}, UnderTen{0}, UnderHundred{0};
    std::atomic<uint64_t> TangentSamples{0}, StickSamples{0}, StickTransitions{0};
    float SampleRate{0};

    void AddPeak(float peak, float load) {
        if (!(load > 0)) return;
        const float ratio = peak / load;
        Peaks.fetch_add(1, std::memory_order_relaxed);
        if (ratio < 1) UnderLoad.fetch_add(1, std::memory_order_relaxed);
        if (ratio < 10) UnderTen.fetch_add(1, std::memory_order_relaxed);
        if (ratio < 100) UnderHundred.fetch_add(1, std::memory_order_relaxed);
    }

    ~ContactReport() {
        const auto samples = Samples.load();
        if (samples == 0 || SampleRate <= 0) return;
        const double seconds = double(samples) / SampleRate;
        const double cell_samples = double(CellSamples.load());
        const double peaks = double(Peaks.load());
        std::fprintf(stderr, "CONTACT rate=%.6g shocks/s  free=%.4f  bearing=%.4f  shocks=%llu  turnover=%.6g events/s  seconds=%.3f\n", double(Shocks.load()) / seconds, double(Free.load()) / double(samples), cell_samples > 0 ? double(CellBearing.load()) / cell_samples : 0.0, static_cast<unsigned long long>(Shocks.load()), double(TurnoverEvents.load()) / seconds, seconds);
        if (peaks > 0) {
            std::fprintf(stderr, "CONTACTPEAKS underM=%.4f under10M=%.4f under100M=%.4f peaks=%llu"
                                 "  (Dang 0.300 0.570 1.000)\n",
                         double(UnderLoad.load()) / peaks, double(UnderTen.load()) / peaks, double(UnderHundred.load()) / peaks, static_cast<unsigned long long>(Peaks.load()));
        }
        if (const auto tangent = TangentSamples.load(); tangent > 0) {
            std::fprintf(stderr, "STICK stick=%.4f transitions/s=%.6g tangent_samples=%llu\n", double(StickSamples.load()) / double(tangent), double(StickTransitions.load()) / seconds, static_cast<unsigned long long>(tangent));
        }
    }
};
ContactReport Report;
const bool ReportContacts = std::getenv("CONTACT_REPORT") != nullptr;

// One cell's engagement counted for CONTACT_REPORT, whichever exchange path produced it.
void ReportCell(SustainedCarry &carry, const SustainedState &st, uint32_t k, float load) {
    const uint32_t bit = 1u << k;
    if (load > 0) {
        if ((carry.SpotBearing & bit) == 0) {
            Report.Shocks.fetch_add(1, std::memory_order_relaxed);
            carry.SpotPeak[k] = load;
        } else {
            carry.SpotPeak[k] = std::max(carry.SpotPeak[k], load);
        }
        carry.SpotBearing |= bit;
        Report.CellBearing.fetch_add(1, std::memory_order_relaxed);
    } else if (carry.SpotBearing & bit) {
        carry.SpotBearing &= ~bit;
        Report.AddPeak(carry.SpotPeak[k], st.NormalForce);
    }
    Report.CellSamples.fetch_add(1, std::memory_order_relaxed);
}

constexpr size_t MaxCoupledVoices{16};
constexpr uint32_t SampledLanes{8};

float BedPotentialIntegral(const SustainedState &st, float approach, const float *heights, const float *means) {
    const uint32_t spots = st.SpotCount;
    if (spots == 0) {
        const float a = std::clamp(approach, 0.f, MaxEngagement);
        return 0.4f * st.Stiffness * a * a * std::sqrt(a);
    }
    float sum = 0;
    for (uint32_t k = 0; k < spots; ++k) {
        const float engaged = std::min(approach + st.SurfaceCompliance * (heights[k] - means[k]), MaxEngagement);
        if (engaged <= 0) continue;
        sum += 0.4f * st.SpotWeight * st.SpotStiffness * engaged * engaged * std::sqrt(engaged);
    }
    return sum;
}
struct SiteSums {
    float Force; // mean of [engaged + spread * h]^(3/2) over the footprint's site population, m^(3/2)
    float Potential; // mean of 0.4 [engaged + spread * h]^(5/2), the spot law's integral, m^(5/2)
};
SiteSums SampleCellSites(const RoughnessTrack &t, double centre, float footprint, float taper, float engaged, float spread) {
    // A cell whose engagement cannot reach the track's tallest site bears nothing, the same result the full sum gives.
    if (engaged + spread * t.HeightMax <= 0) return {0.f, 0.f};
    const auto size = int64_t(t.Heights.size());
    const double lane_stride = double(size) / double(SampledLanes);
    float wsum = 0, force = 0, pot = 0;
    for (uint32_t lane = 0; lane < SampledLanes; ++lane) {
        const double lo = centre + double(lane) * lane_stride - 0.5 * double(footprint);
        const double hi = lo + double(footprint);
        for (int64_t j = int64_t(std::ceil(lo - taper)); j <= int64_t(std::floor(hi + taper)); ++j) {
            const float w = std::min(std::clamp(float(double(j) - lo) / taper + 1.f, 0.f, 1.f), std::clamp(float(hi - double(j)) / taper + 1.f, 0.f, 1.f));
            if (w <= 0) continue;
            wsum += w;
            const float x = engaged + spread * t.Heights[uint64_t(((j % size) + size) % size)];
            if (x <= 0) continue;
            const float s = x * std::sqrt(x);
            force += w * s;
            pot += w * 0.4f * s * x;
        }
    }
    if (wsum <= 0) return {0.f, 0.f};
    return {force / wsum, pot / wsum};
}

struct QuadInputs {
    float PsiTilde; // The auxiliary variable after the datum's metered rebase.
    float GNominal; // The analytic gradient dpsi/dw at the current state.
    float Gamma; // The constrained Hunt-Crossley damping's denominator term, h * psi / (2 dt).
    float Hx; // The constrained damping function at the physical relative rate.
    float WNew; // The current approach.
    float SlamSpeed{0}; // The engagement rate at a landing that crosses the whole spread in one sample, zero otherwise.
    bool InContact;
    bool Rebased;
    std::array<float, 2> Slope;
    // The bin channels: a binned spring voice exchanges through one collocated channel per bin, each with the scalar scheme's quantities.
    // Zero channels is every other voice, exchanging as one.
    uint32_t Channels{0};
    uint32_t ChContact{0}; // One bearing bit per channel.
    std::array<float, MaxSpringBins> ChPsi{}, ChGNominal{}, ChGamma{}, ChHx{}, ChW{};
};

// Whether a voice bears on the pool's per-element springs rather than on the statistical bed's cells.
bool BearsOnSprings(const SustainedState &st, const ContactSpringSet *set) {
    return st.SpringIndex >= 0 && set != nullptr && set->Springs.Count() > 0;
}

VoiceBlock BlockConstants(const SustainedState &st, const ContactSpringSet *spring_set, const RoughnessTrack *turnover, float sample_rate) {
    VoiceBlock out;
    // A sub-audio one-pole mean removes static junction-force offsets after sweep motion stops.
    // The 10 Hz corner sits below the channel's band.
    out.Dcn = std::min(2 * std::numbers::pi_v<float> * 10.f / sample_rate, 1.f);
    const uint32_t spots = st.SpotCount;
    const bool spring_read = BearsOnSprings(st, spring_set);
    if (spots > 0 && st.CellSpread > 0 && !spring_read) {
        out.LoadW = st.SpotWeight * st.SpotStiffness;
        const float cutoff = -std::sqrt(2.f * std::log(std::max(st.SpotWeight, 2.f)));
        out.CutoffM = cutoff * st.CellSpread;
        out.PotCutM = st.CellSpread * st.CellSpread * std::sqrt(st.CellSpread) * BedPotentialFactor(cutoff);
        out.CSite = turnover && st.Turnover.Index >= 0 ? std::min(std::sqrt(float(SampledLanes) * st.Turnover.SubWindow / std::max(st.SpotWeight, 1.f)), 1.f) : 0.f;
        out.TaperS = std::max(2.f * st.Turnover.Rate, 1.f);
    }
    if (!spring_read) return out;
    const uint32_t fast = 1 - (st.SlowSide & 1), slow = st.SlowSide & 1;
    const double redraw = std::abs(double(st.Tracks[fast].Step) - double(st.Tracks[slow].Step));
    out.Refresh = spring_set->Springs.Curves == 1 ? std::max(redraw, std::abs(double(st.Tracks[fast].Step))) : redraw;
    out.Shot = st.JunctionSpacing > 0 && st.JunctionTransit > 0 &&
        st.FootprintLength > 4 * st.JunctionTransit && st.NoiseStiffness > 0;
    if (!out.Shot) return out;
    // A stopped sweep freezes both poles while the blocker removes residual offsets.
    out.Moving = out.Refresh > 0;
    out.P = out.Moving ? float(out.Refresh / double(st.JunctionSpacing)) : 0.f;
    double sum_h2 = 1;
    if (out.Moving) {
        out.P1 = std::exp(-out.Refresh / double(st.FootprintLength));
        out.P2 = std::exp(-out.Refresh / double(st.JunctionTransit));
        const double w = (1 - out.P2) / (out.P1 - out.P2);
        sum_h2 = w * w * (out.P1 * out.P1 / (1 - out.P1 * out.P1) + out.P2 * out.P2 / (1 - out.P2 * out.P2) - 2 * out.P1 * out.P2 / (1 - out.P1 * out.P2));
    }
    const float sigma_f = st.NoiseRms * std::sqrt(st.SpringScale);
    out.A = out.Moving ? sigma_f / std::sqrt(out.P * float(sum_h2)) : 0.f;
    return out;
}

QuadInputs StepVoiceQuad(const SustainedState &st, const RoughnessTrack *const *tracks, const RoughnessTrack *turnover, const ContactSpringSet *spring_set, const VoiceBlock &block, SustainedCarry &carry, float deflection, std::span<const float> bin_defl, float sample_rate) {
    const bool priming = !carry.Primed;
    const bool rebase = priming || carry.Rebase;
    carry.Primed = true;
    carry.Rebase = false;
    if (priming) {
        for (uint32_t i = 0; i < SustainedState::TrackCount; ++i) carry.Pos[i] = double(i) * double(TrackSamples) / SustainedState::TrackCount;
        carry.TurnoverPos = 0;
        carry.SpringPos = 0;
        carry.NoisePos = 0;
    }
    const uint32_t spots = st.SpotCount;
    const bool spring_read = BearsOnSprings(st, spring_set);
    float relief = 0, distance = 0;
    std::array<float, 2> slope{};
    std::array<float, MaxBedSpots> spot_height{};
    for (uint32_t i = 0; i < SustainedState::TrackCount; ++i) {
        const auto &t = st.Tracks[i];
        if (!tracks[i]) continue;
        carry.Pos[i] += t.Rate;
        const bool bedded = spots > 0 && i < 2 && !spring_read;
        const auto reading = bedded ? ReadTrackSpots(*tracks[i], carry.Pos[i], t.Window, t.SubWindow, spots, t.Sigma, spot_height.data()) : ReadTrackSloped(*tracks[i], carry.Pos[i], t.Window);
        if (!bedded && !(spring_read && i < 2)) relief += t.Sigma * reading.Height;
        if (t.Step > 0 && t.Spacing > 0) slope[i & 1] += t.Sigma * reading.Slope / t.Spacing;
        distance = std::max(distance, t.Step);
    }
    double spring_pos = 0;
    float env_here = 0;
    // Register slower-surface crests at the relative slip offset.
    SlidingRegistration slide, prev_slide;
    if (spring_read) {
        carry.SpringPos += st.SpringRate;
        carry.SlipPos += st.SlipRate;
        const auto count = spring_set->Springs.Count();
        spring_pos = std::fmod(carry.SpringPos, double(count));
        if (spring_pos < 0) spring_pos += double(count);
        if (const auto &dev = spring_set->SideCrestDev[st.SlowSide & 1]; dev.size() == count && st.SlipRate != 0) {
            slide = {.SlowCrestDev = dev, .Slip = std::fmod(carry.SlipPos, double(count))};
            prev_slide = {.SlowCrestDev = dev, .Slip = std::fmod(carry.PrevSlipPos, double(count))};
        }
        env_here = EnvelopeAt(spring_set->Envelope, spring_pos) + Reregister(slide, spring_pos - 0.5);
    }
    const float compliance_share = std::max(st.SurfaceCompliance, 1e-12f);
    const float body_side = st.SurfaceCompliance * (carry.RigidNormal + deflection);
    const bool sampled = spots > 0 && st.CellSpread > 0 && !spring_read;
    const bool sites_live = sampled && turnover && st.Turnover.Index >= 0;
    const float load_w = block.LoadW, csite = block.CSite, cutoff_m = block.CutoffM, pot_cut_m = block.PotCutM, taper_s = block.TaperS;
    // The stored energy of the given readings at the given body state, and each cell's site force into `force_out`.
    // The psi rebase differences this at two states, so the site share is evaluated fresh both times.
    // Recompute site potential to exclude previous-body state from the difference.
    const auto sampled_potential = [&](float approach_v, const float *heights, double turnover_pos, float *force_out = nullptr) {
        const double stride_t = double(st.Turnover.Window) / double(spots);
        const double base_t = turnover_pos - 0.5 * double(st.Turnover.Window);
        float sum = 0;
        for (uint32_t k = 0; k < spots; ++k) {
            const float d = std::min(approach_v + st.SurfaceCompliance * (heights[k] - carry.SpotMean[k]), MaxEngagement);
            const auto sums = sites_live ? SampleCellSites(*turnover, base_t + (double(k) + 0.5) * stride_t, st.Turnover.SubWindow, taper_s, d, st.CellSpread) : SiteSums{0.f, 0.f};
            if (force_out) force_out[k] = sums.Force;
            float part = csite * sums.Potential;
            if (d > cutoff_m) part += (1 - csite) * (BedPotentialAt(d, st.CellSpread) - pot_cut_m);
            sum += part;
        }
        return sum * load_w;
    };
    std::array<float, MaxSpringBins> bin_engagement{};
    const uint32_t bins = spring_read ? std::min({st.SpringBins, MaxSpringBins, uint32_t(bin_defl.size())}) : 0;
    const bool tilt_live = spring_read && bins > 0 && st.InverseAngularInertia > 0 && st.SpringHalfExtent > 0;
    carry.PrevTilt = carry.Tilt - carry.TiltMean;
    if (rebase) {
        carry.Tilt = carry.TiltRate = carry.TiltMoment = carry.TiltMean = 0;
        carry.PrevTilt = 0;
    } else if (tilt_live) {
        carry.TiltRate += st.InverseAngularInertia * carry.TiltMoment / sample_rate;
        carry.Tilt += carry.TiltRate / sample_rate;
        carry.TiltMean += (carry.Tilt - carry.TiltMean) * std::min(2 * std::numbers::pi_v<float> * 10.f / sample_rate, 1.f);
    }
    const float tilt_now = carry.Tilt - carry.TiltMean;
    // Convert rigid tilt at a bin lever arm into interface engagement.
    const auto bin_tilt = [&](uint32_t bb, float tilt) {
        return tilt_live ? st.SurfaceCompliance * tilt * SpringBinOffset(bb, st.SpringBins, st.SpringHalfExtent) : 0.f;
    };
    // Include prior turnover injection and tilt in the old-state conformity term.
    for (uint32_t bb = 0; bb < bins; ++bb) {
        bin_engagement[bb] = -st.SurfaceCompliance * (bin_defl[bb] - deflection) + carry.NoiseEngageBin[bb] + bin_tilt(bb, carry.PrevTilt);
    }
    float potential_old = 0;
    std::array<float, MaxSpringBins> bin_pot_old{};
    if (!rebase) {
        const float approach_old = st.StaticPenetration + st.SurfaceCompliance * (carry.PrevRelief - carry.ReliefMean) - body_side + carry.NoiseEngagePrev;
        if (spring_read) {
            const float eng_old = approach_old +
                st.SurfaceCompliance * (EnvelopeAt(spring_set->Envelope, carry.PrevSpringPos) + Reregister(prev_slide, carry.PrevSpringPos - 0.5) - carry.SpringDatum);
            if (bins > 0) {
                std::array<float, MaxSpringBins> bin_f_old{};
                potential_old = st.SpringScale *
                    ReadContactSpringChannels(
                        spring_set->Springs, spring_set->Envelope, carry.PrevSpringPos, eng_old, spring_set->Curvature, spring_set->Reach,
                        std::span{bin_f_old.data(), bins}, std::span{bin_pot_old.data(), bins}, std::span{bin_engagement.data(), bins}, prev_slide
                    )
                        .Potential;
            } else {
                potential_old = st.SpringScale *
                    ReadContactSprings(spring_set->Springs, spring_set->Envelope, carry.PrevSpringPos, eng_old, spring_set->Curvature, spring_set->Reach, prev_slide).Potential;
            }
        } else {
            potential_old = sampled ? sampled_potential(approach_old, carry.PrevSpotHeight.data(), carry.TurnoverPos) : BedPotentialIntegral(st, approach_old, carry.PrevSpotHeight.data(), carry.SpotMean.data());
        }
    }
    const float dc = std::min(distance / ReliefDcLength, 1.f);
    if (rebase) {
        carry.ReliefMean = relief;
        for (uint32_t k = 0; k < spots; ++k) carry.SpotMean[k] = spot_height[k];
        if (spring_read) carry.SpringDatum = env_here;
        ResetRelaxation(carry); // The moved datum steps both positions the channel's history is measured in.
    }
    carry.ReliefMean += (relief - carry.ReliefMean) * dc;
    for (uint32_t k = 0; k < spots; ++k) carry.SpotMean[k] += (spot_height[k] - carry.SpotMean[k]) * dc;
    if (spring_read) carry.SpringDatum += (env_here - carry.SpringDatum) * dc;
    // Asperity sites remain surface-fixed while footprints advance.
    if (sampled) carry.TurnoverPos += st.Turnover.Rate;
    float approach = st.StaticPenetration + st.SurfaceCompliance * (relief - carry.ReliefMean) - body_side +
        (spring_read ? st.SurfaceCompliance * (env_here - carry.SpringDatum) : 0.f) + carry.NoiseEngage;
    if (rebase && !priming && st.SurfaceCompliance > 0) {
        carry.RigidNormal += (approach - carry.RawApproach) / st.SurfaceCompliance;
        approach = carry.RawApproach;
    }
    std::array<float, MaxSpringBins> turnover_bin{};
    float turnover_total = 0;
    carry.NoiseEngagePrev = carry.NoiseEngage;
    carry.NoiseEngage = 0;
    bool bin_injected = false;
    if (spring_read && st.NoiseRms > 0 && st.SpringScale > 0) {
        const uint32_t fast = 1 - (st.SlowSide & 1);
        const auto &ft = st.Tracks[fast];
        if (const auto *track = tracks[fast]; track && track->Heights.size() >= 4 && ft.Spacing > 0) {
            const uint32_t nb = std::max(bins, 1u);
            const float dcn = block.Dcn;
            if (block.Shot) {
                if (carry.NoiseRng == 0) {
                    carry.NoiseRng = 0x51f2c3a496b8d7e5ull ^ (uint64_t(uint32_t(st.SpringIndex + 1)) << 32) ^ std::bit_cast<uint32_t>(st.JunctionSpacing);
                }
                const bool moving = block.Moving;
                const float p = block.P, a = block.A;
                const double p1 = block.P1, p2 = block.P2;
                uint32_t events = 0;
                for (uint32_t bb = 0; bb < nb; ++bb) {
                    const float share = bins > 0 ? std::max(carry.SpringShareMean[bb], 0.f) : 1.f;
                    float &pool = carry.NoisePool[bb];
                    float &smooth = carry.NoiseSmooth[bb];
                    float &expected_pool = carry.NoiseExpectedPool[bb];
                    float &expected_smooth = carry.NoiseExpectedSmooth[bb];
                    if (rebase) pool = smooth = expected_pool = expected_smooth = carry.NoiseMean[bb] = 0;
                    pool *= float(p1);
                    expected_pool = expected_pool * float(p1) + p * share * a;
                    if (moving && share > 0) {
                        const float bound = std::exp(-p * share);
                        uint32_t k = 0;
                        for (float prod = UniformDraw(carry.NoiseRng); prod > bound && k < 64; prod *= UniformDraw(carry.NoiseRng), ++k) pool += a;
                        events += k;
                    }
                    smooth += (pool - smooth) * float(1 - p2);
                    expected_smooth += (expected_pool - expected_smooth) * float(1 - p2);
                    carry.NoiseMean[bb] += (smooth - expected_smooth - carry.NoiseMean[bb]) * dcn;
                    const float fluct = smooth - expected_smooth - carry.NoiseMean[bb];
                    if (bins > 0) {
                        const float dd = fluct * float(nb) / st.NoiseStiffness;
                        carry.NoiseEngageBin[bb] = dd;
                        bin_engagement[bb] = -st.SurfaceCompliance * (bin_defl[bb] - deflection) + dd + bin_tilt(bb, tilt_now);
                    } else {
                        carry.NoiseEngage = fluct / st.NoiseStiffness;
                    }
                }
                bin_injected = bins > 0;
                if (ReportContacts && events > 0) Report.TurnoverEvents.fetch_add(events, std::memory_order_relaxed);
            } else {
                carry.NoisePos += block.Refresh / double(ft.Spacing);
                const double stride_n = double(track->Heights.size()) / double(nb);
                for (uint32_t bb = 0; bb < nb; ++bb) {
                    const float xi = ReadTrackSmooth(*track, carry.NoisePos + double(bb) * stride_n).Height;
                    if (rebase) carry.NoiseMean[bb] = xi;
                    carry.NoiseMean[bb] += (xi - carry.NoiseMean[bb]) * dcn;
                    const float share = bins > 0 ? std::max(carry.SpringShareMean[bb], 0.f) : 1.f;
                    turnover_bin[bb] = st.NoiseRms * std::sqrt(st.SpringScale * share) * (xi - carry.NoiseMean[bb]);
                    turnover_total += turnover_bin[bb];
                }
            }
        }
    }
    if (!bin_injected) carry.NoiseEngageBin.fill(0.f);
    if (tilt_live && !bin_injected) {
        for (uint32_t bb = 0; bb < bins; ++bb) bin_engagement[bb] += bin_tilt(bb, tilt_now) - bin_tilt(bb, carry.PrevTilt);
    }
    std::array<float, MaxBedSpots> site_force{};
    float potential_new;
    SpringRead stack{};
    std::array<float, MaxSpringBins> bin_force{};
    std::array<float, MaxSpringBins> bin_potential{};
    if (spring_read) {
        // With bins each element is read at its own bin's engagement, and the totals are the bins' exact sums.
        stack = bins > 0 ?
            ReadContactSpringChannels(
                spring_set->Springs, spring_set->Envelope, spring_pos, approach, spring_set->Curvature, spring_set->Reach,
                std::span{bin_force.data(), bins}, std::span{bin_potential.data(), bins}, std::span{bin_engagement.data(), bins}, slide
            ) :
            ReadContactSpringBins(
                spring_set->Springs, spring_set->Envelope, spring_pos, approach, spring_set->Curvature, spring_set->Reach,
                std::span{bin_force.data(), bins}, std::span{bin_engagement.data(), bins}, slide
            );
        stack.Force *= st.SpringScale;
        stack.Potential *= st.SpringScale;
        potential_new = stack.Potential;
    } else if (sampled) {
        potential_new = sampled_potential(approach, spot_height.data(), carry.TurnoverPos, site_force.data());
    } else {
        potential_new = BedPotentialIntegral(st, approach, spot_height.data(), carry.SpotMean.data());
    }
    const float psi_new = std::sqrt(2 * potential_new / compliance_share);
    const float psi = rebase ? psi_new : std::max(carry.Psi + psi_new - std::sqrt(2 * potential_old / compliance_share), 0.f);
    // The elastic force at the current state, whose ratio to psi is the analytic gradient.
    // Damping enters through the scheme rather than as a force factor.
    // Cell loads are reported after the loops, keeping the peak and engagement instruments on the whole force.
    std::array<float, MaxBedSpots> report_load{};
    float raw_elastic = 0;
    if (spring_read) {
        raw_elastic += stack.Force > 0 ? std::max(stack.Force + turnover_total, 0.f) : 0.f;
    } else if (sampled) {
        for (uint32_t k = 0; k < spots; ++k) {
            const float d = std::min(approach + st.SurfaceCompliance * (spot_height[k] - carry.SpotMean[k]), MaxEngagement);
            float cell = csite * site_force[k];
            if (d > cutoff_m) cell += (1 - csite) * BedLoadAt(d, st.CellSpread);
            cell *= load_w;
            if (ReportContacts) report_load[k] = cell;
            raw_elastic += std::max(cell, 0.f);
        }
    } else if (spots > 0) {
        for (uint32_t k = 0; k < spots; ++k) {
            const float engaged = std::min(approach + st.SurfaceCompliance * (spot_height[k] - carry.SpotMean[k]), MaxEngagement);
            if (ReportContacts) report_load[k] = engaged > 0 ? st.SpotWeight * st.SpotStiffness * engaged * std::sqrt(engaged) : 0.f;
            if (engaged <= 0) continue;
            raw_elastic += st.SpotWeight * st.SpotStiffness * engaged * std::sqrt(engaged);
        }
    } else {
        const float a = std::clamp(approach, 0.f, MaxEngagement);
        raw_elastic += st.Stiffness * a * std::sqrt(a);
    }
    if (bins > 0) {
        float bin_sum = 0;
        for (uint32_t bb = 0; bb < bins; ++bb) bin_sum += bin_force[bb];
        if (bin_sum > 0) {
            for (uint32_t bb = 0; bb < bins; ++bb) {
                const float share = bin_force[bb] / bin_sum;
                if (rebase) carry.SpringShareMean[bb] = share;
                carry.SpringShareMean[bb] += (share - carry.SpringShareMean[bb]) * dc;
            }
        }
    }
    if (ReportContacts && spring_read) {
        // The stack reads as one force, so the engagement instruments count the whole contact as one cell.
        ReportCell(carry, st, 0, raw_elastic);
    } else if (ReportContacts && spots > 0) {
        for (uint32_t k = 0; k < spots; ++k) ReportCell(carry, st, k, report_load[k]);
    }
    const float v_rel = rebase ? 0.f : (approach - carry.RawApproach) * sample_rate;
    static const bool landing_sized = std::getenv("ETA_LANDING") != nullptr;
    if (!carry.PrevContact) {
        carry.EngagementDamping = landing_sized && !rebase ? st.DampingFactor / std::max(v_rel, ImpactVelocityChi / (sample_rate * sample_rate)) : st.DampingFactor;
    }
    const float eta = carry.EngagementDamping;
    const float hx = eta > 0 ? (v_rel >= -1.f / eta ? eta : -1.f / v_rel) : 0.f;
    // The scheme's damping term scales with psi, so a contact made and broken within one sample returns near-elastically at any damping factor.
    // A landing whose approach crosses the whole engagement spread in one sample therefore resolves impulsively in the sample loop.
    const float engagement_spread = st.CellSpread > 0 ? st.CellSpread : std::abs(st.StaticPenetration);
    const bool slam = spring_read && !rebase && !carry.PrevContact && raw_elastic > 0 && engagement_spread > 0 && v_rel >= engagement_spread * sample_rate;
    carry.RawApproach = approach;
    carry.PrevRelief = relief;
    if (spring_read) {
        carry.PrevSpringPos = spring_pos;
        carry.PrevSlipPos = carry.SlipPos;
    }
    for (uint32_t k = 0; k < spots; ++k) carry.PrevSpotHeight[k] = spot_height[k];
    QuadInputs out{
        .PsiTilde = psi,
        .GNominal = potential_new > 0 ? raw_elastic / (compliance_share * psi_new) : 0.f,
        .Gamma = 0.5f * sample_rate * hx * psi,
        .Hx = hx,
        .WNew = approach,
        .SlamSpeed = slam ? v_rel : 0.f,
        .InContact = raw_elastic > 0,
        .Rebased = rebase,
        .Slope = slope,
    };
    if (spring_read && bins > 0) {
        out.Channels = bins;
        const bool ch_rebase = rebase || !carry.BinsPrimed;
        carry.BinsPrimed = true;
        for (uint32_t bb = 0; bb < bins; ++bb) {
            const float u_new = st.SpringScale * bin_potential[bb];
            const float psi_new_b = std::sqrt(2 * u_new / compliance_share);
            const float psi_old_b = std::sqrt(2 * st.SpringScale * bin_pot_old[bb] / compliance_share);
            out.ChPsi[bb] = ch_rebase ? psi_new_b : std::max(carry.PsiBins[bb] + psi_new_b - psi_old_b, 0.f);
            const float f_b = bin_force[bb] > 0 ? std::max(st.SpringScale * bin_force[bb] + turnover_bin[bb], 0.f) : 0.f;
            out.ChGNominal[bb] = u_new > 0 ? f_b / (compliance_share * psi_new_b) : 0.f;
            out.ChW[bb] = approach + bin_engagement[bb];
            const float v_rel_b = ch_rebase ? 0.f : (out.ChW[bb] - carry.RawApproachBins[bb]) * sample_rate;
            out.ChHx[bb] = eta > 0 ? (v_rel_b >= -1.f / eta ? eta : -1.f / v_rel_b) : 0.f;
            out.ChGamma[bb] = 0.5f * sample_rate * out.ChHx[bb] * out.ChPsi[bb];
            if (f_b > 0) out.ChContact |= 1u << bb;
            carry.RawApproachBins[bb] = out.ChW[bb];
        }
    } else {
        carry.BinsPrimed = false;
    }
    return out;
}

struct TangentInputs {
    float Rhs; // The contact point's free step relative to the surface, before any friction force.
    float Ct; // Tangential displacement response per newton of friction force, modal plus body.
    float Cnt; // Tangential displacement per newton of normal reaction, through modes carrying both shapes.
    float Defl; // This sample's tangential deflection read.
    float RhsTr; // The transverse free step. No surface travel enters: the surface slides along the slip direction alone.
    float CtTr; // Transverse displacement response per newton of transverse force.
    float CntTr; // Transverse displacement per newton of normal reaction.
    float DeflTr; // This sample's transverse deflection read.
    bool Active; // A frictional voice with a tangent direction and a resolvable tangential compliance.
    bool ActiveTr; // A spring voice's transverse channel, resolvable the same way.
};

// One direction's solved junction force with the state to commit for it.
struct DahlStep {
    float Force;
    float Stretch;
};

// The exact Dahl relaxation factor over a junction step of `x` cone widths, r = (1 - exp(-x)) / x, and its derivative.
// Both are series-expanded near zero, where the quotient cancels.
struct DahlRelax {
    float R, dR;
};
DahlRelax DahlRelaxation(float x) {
    if (x < 1e-4f) return {1.f - 0.5f * x + x * x / 6.f, -0.5f + x / 3.f};
    const float e = std::exp(-x);
    return {(1.f - e) / x, (e * (x + 1.f) - 1.f) / (x * x)};
}

DahlStep SolveDahlDirection(float z, float kt, float zmax, float rhs, float compliance) {
    // The linear stick solution starts the iteration and is exact in the small-step limit.
    float ft = -kt * (z + 0.25f * rhs) / (1 + 0.25f * kt * compliance);
    float s = 1.f;
    for (uint32_t it = 0; it < 3; ++it) {
        const float du = 0.5f * (rhs + compliance * ft);
        s = du >= 0.f ? 1.f : -1.f;
        const auto [r, dr] = DahlRelaxation(std::abs(du) / zmax);
        const float zbar = s * zmax + (z - s * zmax) * r;
        const float g = ft + kt * zbar;
        const float dg = 1.f + 0.25f * kt * compliance * (z - s * zmax) * dr * s / zmax;
        ft -= g / dg;
    }
    // Commit the exact map at the solved step, so the stored stretch is the path's endpoint.
    const float du = 0.5f * (rhs + compliance * ft);
    const float sf = du >= 0.f ? 1.f : -1.f;
    const float zn = sf * zmax + (z - sf * zmax) * std::exp(-std::abs(du) / zmax);
    return {ft, std::clamp(zn, -zmax, zmax)};
}

float SolveDahlSpread(std::span<const FlankJunctionBin> bins, std::span<float> z, float cone, float rhs, float compliance) {
    std::array<float, MaxFlankBins> zmax;
    float stiffness = 0, stretch = 0;
    for (size_t b = 0; b < bins.size(); ++b) {
        zmax[b] = cone * bins[b].ConeShare / bins[b].Stiffness;
        stiffness += bins[b].Stiffness;
        stretch += bins[b].Stiffness * z[b];
    }
    float ft = -(stretch + 0.25f * rhs * stiffness) / (1 + 0.25f * stiffness * compliance);
    for (uint32_t it = 0; it < 3; ++it) {
        const float du = 0.5f * (rhs + compliance * ft);
        const float s = du >= 0.f ? 1.f : -1.f;
        float g = ft, dg = 1.f;
        for (size_t b = 0; b < bins.size(); ++b) {
            const auto [r, dr] = DahlRelaxation(std::abs(du) / zmax[b]);
            const float edge = z[b] - s * zmax[b];
            g += bins[b].Stiffness * (s * zmax[b] + edge * r);
            dg += 0.25f * bins[b].Stiffness * compliance * edge * dr * s / zmax[b];
        }
        ft -= g / dg;
    }
    const float du = 0.5f * (rhs + compliance * ft);
    const float sf = du >= 0.f ? 1.f : -1.f;
    for (size_t b = 0; b < bins.size(); ++b) {
        const float zn = sf * zmax[b] + (z[b] - sf * zmax[b]) * std::exp(-std::abs(du) / zmax[b]);
        z[b] = std::clamp(zn, -zmax[b], zmax[b]);
    }
    return ft;
}

} // namespace

float FlankJunctionStep(const SustainedState &st, SustainedCarry &carry, float load, float step, float cn) {
    const float kt = st.FlankStiffness;
    const float fmax = st.Friction * load;
    if (kt <= 0 || fmax <= 0 || cn <= 0 || st.SpringIndex < 0) {
        carry.PatchFlank = 0.f;
        carry.PatchFlankBin = {};
        return 0.f;
    }
    const float compliance = st.SurfaceCompliance * cn;
    // The spread replaces the single element with the population's own strengths, and its stick stiffness is the sum over its bins.
    const std::span bins{st.FlankBin.data(), st.FlankBins};
    const std::span stretches{carry.PatchFlankBin.data(), st.FlankBins};
    float force = 0;
    if (st.FlankBins > 0) {
        force = SolveDahlSpread(bins, stretches, fmax, step, compliance);
    } else {
        const auto slip = SolveDahlDirection(carry.PatchFlank, kt, fmax / kt, step, compliance);
        force = slip.Force;
        carry.PatchFlank = slip.Stretch;
    }
    return -force;
}

void ResetRelaxation(SustainedCarry &carry) {
    carry.RelaxEdges = 0;
    carry.RelaxDepth = 0;
    carry.RelaxTangentPos = 0;
}

float RelaxationRelease(const SustainedState &st, SustainedCarry &carry, float approach_step, float tangent_step) {
    if (st.RelaxScale <= 0) return 0.f;
    const float d_prev = carry.RelaxDepth, x_prev = carry.RelaxTangentPos;
    const float d = d_prev + approach_step, x = x_prev + tangent_step;
    carry.RelaxDepth = d;
    carry.RelaxTangentPos = x;
    if (carry.RelaxEdges > 0 && d == carry.RelaxApproach[carry.RelaxEdges - 1]) return 0.f;
    if (carry.RelaxEdges == 0 || d > carry.RelaxApproach[carry.RelaxEdges - 1]) {
        // Growing: the depth this step covers engages along the ramp the step's motion draws, which the point it ends at records.
        // A full history keeps every other point and the deepest one, halving the resolution evenly over the whole span.
        if (carry.RelaxEdges == MaxRelaxEdges) {
            uint32_t kept = 0;
            for (uint32_t i = 0; i < MaxRelaxEdges - 1; i += 2, ++kept) {
                carry.RelaxApproach[kept] = carry.RelaxApproach[i];
                carry.RelaxTangent[kept] = carry.RelaxTangent[i];
            }
            carry.RelaxApproach[kept] = carry.RelaxApproach[MaxRelaxEdges - 1];
            carry.RelaxTangent[kept] = carry.RelaxTangent[MaxRelaxEdges - 1];
            carry.RelaxEdges = kept + 1;
        }
        carry.RelaxApproach[carry.RelaxEdges] = d;
        carry.RelaxTangent[carry.RelaxEdges] = x;
        ++carry.RelaxEdges;
        return 0.f;
    }
    const float release_slope = d != d_prev ? (x - x_prev) / (d - d_prev) : 0.f;
    double energy = 0;
    while (carry.RelaxEdges > 1 && d < carry.RelaxApproach[carry.RelaxEdges - 1]) {
        const uint32_t top = carry.RelaxEdges - 1;
        const float hi = carry.RelaxApproach[top], bottom = carry.RelaxApproach[top - 1];
        const float lo = std::max(d, bottom), span = hi - bottom;
        const float engaged_lo = span > 0 ? carry.RelaxTangent[top - 1] + (carry.RelaxTangent[top] - carry.RelaxTangent[top - 1]) * (lo - bottom) / span : carry.RelaxTangent[top];
        const float s_hi = x + release_slope * (hi - d) - carry.RelaxTangent[top];
        const float s_lo = x + release_slope * (lo - d) - engaged_lo;
        energy += double(st.RelaxScale) * (double(hi) - lo) * (double(s_hi) * s_hi + double(s_hi) * s_lo + double(s_lo) * s_lo) / 3.0;
        if (lo > d) {
            --carry.RelaxEdges; // spent, and the annulus under it is now the edge
        } else {
            carry.RelaxApproach[top] = d;
            carry.RelaxTangent[top] = engaged_lo;
            return float(energy);
        }
    }
    if (d < carry.RelaxApproach[0]) {
        carry.RelaxApproach[0] = 0;
        carry.RelaxTangent[0] = 0;
        carry.RelaxDepth = 0;
        carry.RelaxTangentPos = 0;
    }
    return float(energy);
}

namespace {

// The junction's forces on the two in-plane axes: the frictional fluctuation along the slip direction and the transverse force.
struct FrictionForces {
    float Frictional;
    float Transverse;
};

FrictionForces SolveVoiceFriction(const SustainedState &st, SustainedCarry &carry, float reaction, float fy, const TangentInputs &tan, float approach_step) {
    float frictional = st.Friction * reaction;
    float transverse = 0.f;
    if (tan.Active) {
        const float fmax = st.Friction * fy;
        const float rhs = tan.Rhs + tan.Cnt * reaction;
        const float kt = st.ShearStiffness;
        float ft;
        bool stick = false;
        if (const float zmax = fmax / kt; kt > 0 && fmax > 0 && zmax >= std::numeric_limits<float>::min() && st.SpringIndex >= 0) {
            // Both stretches live on half-steps like psi.
            // The shared cone couples the axes, the vector force projecting back to the cone, which only releases stretch.
            const auto along = SolveDahlDirection(carry.PatchShear, kt, zmax, rhs, tan.Ct);
            ft = along.Force;
            carry.PatchShear = along.Stretch;
            if (tan.ActiveTr) {
                const auto across = SolveDahlDirection(carry.PatchShearTransverse, kt, zmax, tan.RhsTr + tan.CntTr * reaction, tan.CtTr);
                transverse = across.Force;
                carry.PatchShearTransverse = across.Stretch;
            } else {
                carry.PatchShearTransverse = 0.f;
            }
            const float norm = std::sqrt(ft * ft + transverse * transverse);
            if (norm > fmax) {
                const float scale = fmax / norm;
                ft *= scale;
                transverse *= scale;
            }
            // The stretch vector projects onto the cone ball, and scaling stored stretch only releases energy.
            const float stretch_norm = std::sqrt(carry.PatchShear * carry.PatchShear + carry.PatchShearTransverse * carry.PatchShearTransverse);
            if (stretch_norm > zmax) {
                const float stretch_scale = zmax / stretch_norm;
                carry.PatchShear *= stretch_scale;
                carry.PatchShearTransverse *= stretch_scale;
            }
            stick = norm < fmax;
        } else if (kt > 0) {
            const float z = carry.PatchShear;
            const float f_stick = -kt * (z + 0.25f * rhs) / (1 + 0.25f * kt * tan.Ct);
            if (fmax <= 0) {
                ft = 0.f;
                carry.PatchShear = 0.f;
                carry.PatchShearTransverse = 0.f;
            } else if (std::abs(f_stick) <= fmax) {
                ft = f_stick;
                stick = true;
                carry.PatchShear = z + 0.5f * (rhs + tan.Ct * ft);
            } else {
                ft = std::copysign(fmax, f_stick);
                carry.PatchShear = -ft / kt;
            }
        } else {
            // Clamp the force that cancels relative contact-point motion to the Coulomb cone.
            const float hold = -rhs / tan.Ct;
            float hold_tr = tan.ActiveTr ? -(tan.RhsTr + tan.CntTr * reaction) / tan.CtTr : 0.f;
            const float norm = std::sqrt(hold * hold + hold_tr * hold_tr);
            stick = norm < fmax;
            const float scale = norm > fmax && norm > 0 ? fmax / norm : 1.f;
            ft = hold * scale;
            transverse = hold_tr * scale;
        }
        // Relaxation removes energy only from stored junction shear.
        // Remove shear energy from annuli released by a receding contact.
        if (kt > 0 && fmax <= 0) {
            ResetRelaxation(carry); // A parted contact has no edge, so its history no longer applies.
        } else if (kt > 0) {
            const float tangent_step = rhs + tan.Ct * ft;
            const float released = RelaxationRelease(st, carry, approach_step, tangent_step);
            const float stored = 0.5f * kt * (carry.PatchShear * carry.PatchShear + carry.PatchShearTransverse * carry.PatchShearTransverse);
            if (released > 0 && stored > 0) {
                const float scale = std::sqrt(std::max(1.f - released / stored, 0.f));
                carry.PatchShear *= scale;
                carry.PatchShearTransverse *= scale;
            }
        }
        frictional = ft - st.SolverFriction;
        if (ReportContacts) {
            Report.TangentSamples.fetch_add(1, std::memory_order_relaxed);
            if (stick) Report.StickSamples.fetch_add(1, std::memory_order_relaxed);
            if (stick != carry.PrevStick) Report.StickTransitions.fetch_add(1, std::memory_order_relaxed);
            carry.PrevStick = stick;
        }
    }
    carry.PrevRigidTangent = carry.RigidTangent;
    carry.PrevTangentDefl = tan.Defl;
    carry.PrevRigidTransverse = carry.RigidTransverse;
    carry.PrevTransverseDefl = tan.DeflTr;
    return {frictional, transverse};
}

struct ChannelAdvance {
    float Psi, Force, Fy;
};
ChannelAdvance AdvanceChannel(float psi, float g, float step, float hx, bool bearing, float compliance, float eta, float sample_rate, bool tangent) {
    const float psi_step = psi + 0.5f * g * step;
    float psi_next = std::max(bearing ? psi_step : std::min(psi_step, psi), 0.f);
    if (!bearing && psi_next < std::numeric_limits<float>::min()) psi_next = 0.f;
    const float dpsi = (psi_next - psi) * sample_rate;
    const float force = compliance * (0.5f * (psi_next + psi) * g + (bearing ? hx * dpsi * psi : 0.f));
    float fy = 0.f;
    if (tangent) {
        const float vxc = 0.5f * step * sample_rate;
        const float hy = eta > 0 ? (vxc >= -1.f / eta ? eta : -1.f / vxc) : 0.f;
        fy = compliance * 0.5f * (psi_next + psi) * g * (bearing ? 1 + vxc * hy : 1.f);
    }
    return {psi_next, force, fy};
}

// The whole contact's sample counted for CONTACT_REPORT, whichever exchange path solved it.
void ReportSample(float force, float sample_rate) {
    Report.SampleRate = sample_rate;
    Report.Samples.fetch_add(1, std::memory_order_relaxed);
    if (force <= 0) Report.Free.fetch_add(1, std::memory_order_relaxed);
}

// The junction forces on a solved reaction: friction, the transverse axis, and the oblique-flank element on the normal channel.
// The flank force is returned apart so a binned voice can spread it over its rows.
// The geometric rows keep the exchange's load, the flank force being a fluctuation about it.
struct VoiceJunctions {
    VoiceForce Force;
    float Flank;
};
VoiceJunctions SolveVoiceJunctions(const SustainedState &st, SustainedCarry &carry, float reaction, float fy, float step, float cn, float slope0, float slope1, const TangentInputs &tan) {
    const auto junction = SolveVoiceFriction(st, carry, reaction, fy, tan, step);
    const float load = st.NormalForce + reaction;
    const float flank = FlankJunctionStep(st, carry, load, step, cn);
    return {{reaction + flank, {load * slope0, load * slope1}, junction.Frictional, junction.Transverse}, flank};
}

VoiceForce FinishVoiceQuad(const SustainedState &st, SustainedCarry &carry, float psi_tilde, float hx, bool in_contact, float g, float step, float cn, float slope0, float slope1, float sample_rate, const TangentInputs &tan) {
    const auto advance = AdvanceChannel(psi_tilde, g, step, hx, in_contact, st.SurfaceCompliance, carry.EngagementDamping, sample_rate, tan.Active);
    carry.Psi = advance.Psi;
    carry.PrevContact = in_contact;
    carry.PrevRigidNormal = carry.RigidNormal;
    if (ReportContacts) ReportSample(advance.Force, sample_rate);
    return SolveVoiceJunctions(st, carry, advance.Force - st.NormalForce, advance.Fy, step, cn, slope0, slope1, tan).Force;
}

VoiceForce FinishBinnedVoiceQuad(
    const SustainedState &st, SustainedCarry &carry, uint32_t channels, uint32_t contact_mask,
    const float *psi, const float *hx, const float *g, const float *step,
    const float *support, float *bin_reaction,
    float cn, float slope0, float slope1, float sample_rate, const TangentInputs &tan
) {
    float force = 0.f, fy = 0.f, psi_sq = 0.f;
    const float eta = carry.EngagementDamping;
    for (uint32_t bb = 0; bb < channels; ++bb) {
        const bool bearing_b = ((contact_mask >> bb) & 1u) != 0;
        const auto advance = AdvanceChannel(psi[bb], g[bb], step[bb], hx[bb], bearing_b, st.SurfaceCompliance, eta, sample_rate, tan.Active);
        carry.PsiBins[bb] = advance.Psi;
        psi_sq += advance.Psi * advance.Psi;
        force += advance.Force;
        bin_reaction[bb] = advance.Force - support[bb];
        fy += advance.Fy;
    }
    carry.Psi = std::sqrt(psi_sq);
    carry.PrevContact = contact_mask != 0;
    carry.PrevRigidNormal = carry.RigidNormal;
    if (ReportContacts) ReportSample(force, sample_rate);
    std::array<float, MaxSpringBins> share{};
    float support_total = 0.f;
    for (uint32_t bb = 0; bb < channels; ++bb) support_total += support[bb];
    float mean_step = 0.f;
    for (uint32_t bb = 0; bb < channels; ++bb) {
        share[bb] = support_total > 0 ? support[bb] / support_total : 1.f / float(channels);
        mean_step += share[bb] * step[bb];
    }
    // The bins share one flank junction on the voice's channel total, as the friction does, the interface being one population under one footprint.
    // Its force is distributed over the same bin rows the reactions are, so the modes and the body read one contact.
    const auto solved = SolveVoiceJunctions(st, carry, force - st.NormalForce, fy, mean_step, cn, slope0, slope1, tan);
    for (uint32_t bb = 0; bb < channels; ++bb) bin_reaction[bb] += share[bb] * solved.Flank;
    return solved.Force;
}

// One channel's step against its own diagonal, which is the one-channel solve and the whole solve past the coupling cap.
float SolveChannelStep(float cross, float free_step, float psi, float gamma, float g) {
    return (free_step - cross * psi * g) / (1 + cross * (gamma * g + 0.25f * g * g));
}

float ReadDeflection(const float *__restrict gains_im, const float *__restrict gains_re, const float *__restrict state_im, const float *__restrict state_re, uint32_t count) {
#pragma clang fp reassociate(on)
    float sum = 0.f;
    for (uint32_t k = 0; k < count; ++k) sum += gains_im[k] * state_im[k] + gains_re[k] * state_re[k];
    return sum;
}

// The displacement an already-known excitation adds through one read row.
void AccumulateDot(float &acc, const float *__restrict gains, const float *__restrict excite, uint32_t count) {
    for (uint32_t k = 0; k < count; ++k) acc += gains[k] * excite[k];
}

// Returns current displacement, two-step free displacement, and known-arrival displacement for one row.
struct RowRead {
    float Now, Next, Arrival;
};
RowRead ReadRow(
    const ModeReadGains &g, size_t at, const float *state_im, const float *state_re, uint32_t count,
    const float *impacts, const float *sweep
) {
    RowRead out{
        ReadDeflection(&g.Im[at], &g.Re[at], state_im, state_re, count),
        ReadDeflection(&g.Im2[at], &g.Re2[at], state_im, state_re, count),
        0.f,
    };
    if (impacts) AccumulateDot(out.Arrival, &g.Re[at], impacts, count);
    if (sweep) AccumulateDot(out.Arrival, &g.Re[at], sweep, count);
    return out;
}

// A voice's drive rows: the normal, each surface's geometric tangential, and the frictional.
// The transverse frictional row is appended per candidate voice like the bin rows.
constexpr uint32_t VoiceDrives{4};
constexpr uint32_t NoTransverseRow{~0u};

// This sample's excitation of every mode from the drives past the first voice's, summed over those bearing force.
void GatherExcitation(const float *__restrict gains, const float *__restrict forces, uint32_t voice_drives, uint32_t drives, float *__restrict excite, uint32_t count) {
    bool seeded = false;
    const auto row = [&](uint32_t d) { return gains + size_t(d) * count; };
    for (uint32_t d = VoiceDrives; d < voice_drives; d += VoiceDrives) {
        const float f0 = forces[d], f1 = forces[d + 1], f2 = forces[d + 2], f3 = forces[d + 3];
        const float *__restrict g0 = row(d), *__restrict g1 = row(d + 1), *__restrict g2 = row(d + 2), *__restrict g3 = row(d + 3);
        if (seeded) {
            for (uint32_t k = 0; k < count; ++k) excite[k] += g0[k] * f0 + g1[k] * f1 + g2[k] * f2 + g3[k] * f3;
        } else {
            for (uint32_t k = 0; k < count; ++k) excite[k] = g0[k] * f0 + g1[k] * f1 + g2[k] * f2 + g3[k] * f3;
            seeded = true;
        }
    }
    for (uint32_t d = voice_drives; d < drives; ++d) {
        const float force = forces[d];
        if (force == 0.f) continue;
        const float *__restrict g = row(d);
        if (seeded) {
            for (uint32_t k = 0; k < count; ++k) excite[k] += g[k] * force;
        } else {
            for (uint32_t k = 0; k < count; ++k) excite[k] = g[k] * force;
            seeded = true;
        }
    }
    if (!seeded) std::fill_n(excite, count, 0.f);
}

// Advances every mode by one sample and returns the radiated pressure.
// The first voice's drive rows enter here directly, and the sum runs the length of the mode count, so it reassociates into independent partial sums.
float AdvanceModes(
    float *__restrict state_re, float *__restrict state_im, const float *__restrict coeff_re, const float *__restrict coeff_im,
    const float *__restrict gains, const float *__restrict forces, const float *__restrict excite,
    const float *__restrict phase_im, const float *__restrict phase_re, uint32_t count
) {
#pragma clang fp reassociate(on)
    const float f0 = forces[0], f1 = forces[1], f2 = forces[2], f3 = forces[3];
    const float *__restrict g0 = gains, *__restrict g1 = gains + count, *__restrict g2 = gains + 2 * count, *__restrict g3 = gains + 3 * count;
    float acc = 0.f;
    for (uint32_t k = 0; k < count; ++k) {
        const auto excitation = g0[k] * f0 + g1[k] * f1 + g2[k] * f2 + g3[k] * f3 + excite[k];
        const auto re = state_re[k] * coeff_re[k] - state_im[k] * coeff_im[k] + excitation;
        state_im[k] = state_re[k] * coeff_im[k] + state_im[k] * coeff_re[k];
        state_re[k] = re;
        acc += phase_im[k] * state_im[k] + phase_re[k] * re;
    }
    return acc;
}

// One block's fixed shape: the object's mode window, its drive levels, its row and channel layout, and its contacts' support.
struct RenderBlock {
    uint32_t K0{0}, Count{0}; // The object's first mode, and how many of them still sound.
    float OutGain{0}, ListenerGain{0}, SustainLevel{0};
    uint32_t ExtraBase{0}; // The first drive row past the voices' own and the impacts'.
    uint32_t Drives{0}, Channels{0};
    bool Coupled{false}; // Whether the channels solve as one system rather than each against its own diagonal.
    vec3 SupportForce{0};
};

// Block-constant object render state resolved before the first sample.
RenderBlock PrepareRenderBlock(SurfaceAudioState &surface, SurfaceRenderScratch &w, ModalBank &b, uint32_t o, std::span<const uint32_t> impacts, std::span<const uint32_t> voices) {
    const auto k0 = b.ModeOffset[o], stride = b.ModeCount[o], count = b.TunedModeCount[o];
    const auto shape0 = b.ShapeOffset[o];
    const auto listener_gain = std::atomic_ref{b.ListenerGain[o]}.load(std::memory_order_relaxed);
    const auto out_gain = std::atomic_ref{b.OutGain[o]}.load(std::memory_order_relaxed) * listener_gain;
    const auto sample_rate = b.SampleRate;
    const auto coupling = surface.Coupling.load(std::memory_order_relaxed) * b.DeflectionScale[o];
    const auto sustain = surface.SustainLevel.load(std::memory_order_relaxed);
    const auto sustain_level = sustain / sample_rate;

    // Distributed-excitation rows follow the voices' own rows and the impact rows, one per bin per spring voice, gathered as impact rows are.
    // A voice without bins reserves nothing.
    const uint32_t extra_base = uint32_t(voices.size()) * VoiceDrives + uint32_t(impacts.size());
    w.BinRowBase.resize(voices.size());
    uint32_t bin_rows = 0;
    for (size_t t = 0; t < voices.size(); ++t) {
        w.BinRowBase[t] = extra_base + bin_rows;
        bin_rows += std::min(surface.VoiceState[voices[t]].SpringBins, MaxSpringBins);
    }
    // The transverse frictional rows follow the bin rows, one per spring voice with a junction.
    w.TransverseRow.resize(voices.size());
    uint32_t transverse_rows = 0;
    for (size_t t = 0; t < voices.size(); ++t) {
        const auto &st = surface.VoiceState[voices[t]];
        const bool candidate = st.SpringIndex >= 0 && st.Friction > 0 && numeric::Dot(st.SlipDir, st.SlipDir) > 0;
        w.TransverseRow[t] = candidate ? extra_base + bin_rows + transverse_rows++ : NoTransverseRow;
    }
    const uint32_t drives = extra_base + bin_rows + transverse_rows;
    // Locate the read gains for bin `bb` of voice `t`.
    const auto bin_read_at = [&w, extra_base, count](size_t t, uint32_t bb) { return size_t(w.BinRowBase[t] - extra_base + bb) * count; };
    w.DriveGains.resize(size_t(drives) * count);
    w.BinRead.Resize(size_t(bin_rows) * count);
    w.SolvedForce.resize(voices.size());
    w.PointRead.Resize(voices.size() * count);
    w.Forces.resize(drives);
    w.Excite.resize(count);
    w.SweepExcite.resize(count);
    const size_t nv = voices.size();
    // Past MaxCoupledVoices only each voice's own diagonal is built and solved against.
    const bool coupled = nv <= MaxCoupledVoices;
    w.ChannelBase.resize(nv);
    w.ChannelCount.resize(nv);
    uint32_t nch = 0;
    for (size_t t = 0; t < nv; ++t) {
        const uint32_t nb = coupled ? std::min(surface.VoiceState[voices[t]].SpringBins, MaxSpringBins) : 0;
        w.ChannelBase[t] = nch;
        w.ChannelCount[t] = nb > 0 ? nb : 1;
        nch += w.ChannelCount[t];
    }
    w.NormalShape.resize(size_t(nch) * count);
    w.QuadCross.resize(coupled ? size_t(nch) * nch : nch);
    w.QuadMat.resize(coupled ? size_t(nch) * nch : 0);
    w.TangentRead.Resize(nv * count);
    w.TransverseRead.Resize(nv * count);
    w.QuadSlope.resize(2 * nv);
    for (auto *col : {&w.ChSupport, &w.QuadFeDefl, &w.QuadPsi, &w.QuadG, &w.QuadGamma, &w.QuadFree, &w.QuadPrevW, &w.QuadS, &w.QuadHx, &w.QuadGNominal}) col->resize(nch);
    for (auto *col : {&w.QuadDefl, &w.QuadSlam, &w.QuadCn, &w.QuadCt, &w.QuadCnt, &w.QuadFeT, &w.QuadRhsT, &w.QuadDeflT, &w.QuadCtTr, &w.QuadCntTr, &w.QuadRhsTr, &w.QuadDeflTr}) col->resize(nv);
    for (auto *col : {&w.QuadTangent, &w.QuadTransverse}) col->resize(nv);
    w.QuadRhs.resize(nch);
    w.QuadContact.resize(nch);
    w.TransverseDir.resize(nv);
    const auto drive_row = [&](size_t i) { return &w.DriveGains[i * count]; };
    // Voice state is fixed for the block, so each voice's surface tracks resolve once here.
    // The load orders against the reader generation, so an ended generation means no track read here is still in use.
    w.Tracks.resize(voices.size() * SustainedState::TrackCount);
    w.TurnoverTracks.resize(voices.size());
    w.SpringSets.resize(voices.size());
    w.SweepSets.resize(voices.size());
    w.SweepConformity.resize(voices.size() * count);
    w.VoiceBlocks.resize(voices.size());
    const float dtb = 1.f / sample_rate;
    for (size_t t = 0; t < voices.size(); ++t) {
        const auto &st = surface.VoiceState[voices[t]];
        for (uint32_t i = 0; i < SustainedState::TrackCount; ++i) {
            const auto index = st.Tracks[i].Index;
            w.Tracks[t * SustainedState::TrackCount + i] =
                index < 0 ? nullptr : surface.SurfaceTracks[uint32_t(index)].Live.load(std::memory_order_seq_cst);
        }
        w.TurnoverTracks[t] = st.Turnover.Index < 0 ? nullptr : surface.SurfaceTracks[uint32_t(st.Turnover.Index)].Live.load(std::memory_order_seq_cst);
        w.SpringSets[t] = st.SpringIndex < 0 ? nullptr : surface.ContactSprings[uint32_t(st.SpringIndex)].Live.load(std::memory_order_seq_cst);
        w.SweepSets[t] = st.SweepIndex < 0 ? nullptr : surface.SweepTables[uint32_t(st.SweepIndex)].Live.load(std::memory_order_seq_cst);
        w.VoiceBlocks[t] = BlockConstants(st, w.SpringSets[t], w.TurnoverTracks[t], sample_rate);
        if (const auto *sweep = w.SweepSets[t]) {
            auto *conformity = &w.SweepConformity[t * count];
            const auto rows = std::min(count, sweep->Modes);
            const auto *stiffness = sweep->ModeStiffness.size() >= rows ? sweep->ModeStiffness.data() : nullptr;
            const float defl_scale = b.DeflectionScale[o];
            for (uint32_t k = 0; k < rows; ++k) {
                const float inv_w = b.DeflectionGain[k0 + k] * b.RadiationGain[k0 + k];
                conformity[k] = stiffness ? 1.f / (1.f + stiffness[k] * defl_scale * inv_w * inv_w) : 1.f;
            }
        }
        auto *gain_n = drive_row(t * VoiceDrives);
        auto *gain_geo0 = drive_row(t * VoiceDrives + 1), *gain_geo1 = drive_row(t * VoiceDrives + 2);
        auto *gain_fric = drive_row(t * VoiceDrives + 3);
        const auto base0 = shape0 + st.Blend.Points[0] * stride, base1 = shape0 + st.Blend.Points[1] * stride, base2 = shape0 + st.Blend.Points[2] * stride;
        const float w0 = st.Blend.Weights.x, w1 = st.Blend.Weights.y, w2 = st.Blend.Weights.z;
        // A tangent direction marks the junction: the slip direction while sliding, the resting basis of a spring contact at rest.
        // Voices without one read a zero direction.
        const bool tangent_candidate = st.Friction > 0 && numeric::Dot(st.SlipDir, st.SlipDir) > 0;
        const vec3 transverse_raw = numeric::Cross(st.N, st.SlipDir);
        const float transverse_len2 = numeric::Dot(transverse_raw, transverse_raw);
        const bool transverse_candidate = w.TransverseRow[t] != NoTransverseRow && transverse_len2 > 1e-12f;
        const vec3 transverse_dir = transverse_candidate ? transverse_raw / std::sqrt(transverse_len2) : vec3{0};
        auto *gain_transverse = w.TransverseRow[t] != NoTransverseRow ? drive_row(w.TransverseRow[t]) : nullptr;
        if (gain_transverse && !transverse_candidate) std::fill_n(gain_transverse, count, 0.f);
        const auto ch0 = w.ChannelBase[t];
        const bool binned = w.ChannelCount[t] > 1;
        float cn_modal = 0, ct_modal = 0, cnt_modal = 0, ct_tr_modal = 0, cnt_tr_modal = 0;
        for (uint32_t k = 0; k < count; ++k) {
            const auto shape = BlendedShape(b, base0, base1, base2, {w0, w1, w2}, k);
            // Every drive takes the radiation gain here, so the sample loop does not.
            const float radiation = b.RadiationGain[k0 + k];
            const float normal = numeric::Dot(shape, st.N);
            gain_n[k] = radiation * normal;
            cn_modal += normal * normal * b.QuadCompliance[k0 + k];
            // Each surface's geometric force acts along the contact's travel over it, and the frictional one along the slip.
            gain_geo0[k] = radiation * numeric::Dot(shape, st.SweepDir[0]);
            gain_geo1[k] = radiation * numeric::Dot(shape, st.SweepDir[1]);
            gain_fric[k] = radiation * numeric::Dot(shape, st.SlipDir);
            const float read = coupling * normal * b.DeflectionGain[k0 + k];
            const float c_re = b.CoeffRe[k0 + k], c_im = b.CoeffIm[k0 + k];
            w.PointRead.Fill(t * count + k, read, c_re, c_im);
            if (!binned) w.NormalShape[size_t(ch0) * count + k] = normal;
            if (tangent_candidate) {
                const float slip = numeric::Dot(shape, st.SlipDir);
                w.TangentRead.Fill(t * count + k, coupling * slip * b.DeflectionGain[k0 + k], c_re, c_im);
                ct_modal += slip * slip * b.QuadCompliance[k0 + k];
                cnt_modal += normal * slip * b.QuadCompliance[k0 + k];
            }
            if (transverse_candidate) {
                const float across = numeric::Dot(shape, transverse_dir);
                w.TransverseRead.Fill(t * count + k, coupling * across * b.DeflectionGain[k0 + k], c_re, c_im);
                ct_tr_modal += across * across * b.QuadCompliance[k0 + k];
                cnt_tr_modal += normal * across * b.QuadCompliance[k0 + k];
                gain_transverse[k] = radiation * across;
            }
            const float scale = b.QuadDriveScale[k0 + k];
            gain_n[k] *= scale;
            gain_geo0[k] *= scale;
            gain_geo1[k] *= scale;
            gain_fric[k] *= scale;
            if (transverse_candidate) gain_transverse[k] *= scale;
        }
        for (uint32_t bb = 0, nb = std::min(st.SpringBins, MaxSpringBins); bb < nb; ++bb) {
            auto *gain_bin = drive_row(w.BinRowBase[t] + bb);
            const auto rel = bin_read_at(t, bb);
            const auto &bl = st.SpringBinBlend[bb];
            const auto bin0 = shape0 + bl.Points[0] * stride, bin1 = shape0 + bl.Points[1] * stride, bin2 = shape0 + bl.Points[2] * stride;
            const float bw0 = bl.Weights.x, bw1 = bl.Weights.y, bw2 = bl.Weights.z;
            for (uint32_t k = 0; k < count; ++k) {
                const auto shape = BlendedShape(b, bin0, bin1, bin2, {bw0, bw1, bw2}, k);
                const float normal_bin = numeric::Dot(shape, st.N);
                gain_bin[k] = b.RadiationGain[k0 + k] * normal_bin * b.QuadDriveScale[k0 + k];
                const float read_bin = coupling * normal_bin * b.DeflectionGain[k0 + k];
                w.BinRead.Fill(rel + k, read_bin, b.CoeffRe[k0 + k], b.CoeffIm[k0 + k]);
                if (binned && bb < w.ChannelCount[t]) w.NormalShape[size_t(ch0 + bb) * count + k] = normal_bin;
            }
        }
        const float body_ct = dtb * dtb * b.RigidInvMass[o];
        // The junction's normal displacement response per newton, modal plus body, for sizing the slam closure's impulse.
        w.QuadCn[t] = coupling * sustain_level * cn_modal + body_ct;
        const float ct = coupling * sustain_level * ct_modal + body_ct;
        w.QuadTangent[t] = tangent_candidate && ct > 0 ? 1 : 0;
        w.QuadCt[t] = ct;
        w.QuadCnt[t] = coupling * sustain_level * cnt_modal + body_ct * numeric::Dot(st.N, st.SlipDir);
        w.QuadFeT[t] = -ct * st.SolverFriction;
        const float ct_tr = coupling * sustain_level * ct_tr_modal + body_ct;
        w.QuadTransverse[t] = transverse_candidate && ct_tr > 0 ? 1 : 0;
        w.QuadCtTr[t] = ct_tr;
        w.QuadCntTr[t] = coupling * sustain_level * cnt_tr_modal + body_ct * numeric::Dot(st.N, transverse_dir);
        w.TransverseDir[t] = transverse_dir;
    }
    for (size_t t = 0; t < impacts.size(); ++t) {
        ImpactGainRow(b, impacts[t], shape0, stride, k0, 0, count, drive_row(voices.size() * VoiceDrives + t));
    }
    vec3 support_force{0};
    {
        const float dtq = 1.f / sample_rate;
        const float inv_mass_q = b.RigidInvMass[o];
        for (size_t t = 0; t < nv; ++t) {
            const auto &stv = surface.VoiceState[voices[t]];
            const auto base = w.ChannelBase[t];
            const auto nchv = w.ChannelCount[t];
            if (nchv == 1) {
                w.ChSupport[base] = stv.NormalForce;
            } else {
                const auto &vc = surface.VoiceCarry[voices[t]];
                float ssum = 0;
                for (uint32_t c = 0; c < nchv; ++c) ssum += vc.SpringShareMean[c];
                for (uint32_t c = 0; c < nchv; ++c) w.ChSupport[base + c] = stv.NormalForce * (ssum > 0.5f ? vc.SpringShareMean[c] / ssum : 1.f / float(nchv));
            }
            support_force += stv.NormalForce * stv.N;
        }
        for (size_t v = 0; v < nv; ++v) {
            const auto &sv = surface.VoiceState[voices[v]];
            for (uint32_t cv = 0; cv < w.ChannelCount[v]; ++cv) {
                const auto chv = size_t(w.ChannelBase[v]) + cv;
                float fe_defl = 0;
                for (size_t u = coupled ? 0 : v, u_end = coupled ? nv : v + 1; u < u_end; ++u) {
                    const auto &su = surface.VoiceState[voices[u]];
                    const float body = inv_mass_q > 0 ? dtq * dtq * inv_mass_q * numeric::Dot(sv.N, su.N) : 0.f;
                    for (uint32_t cu = 0; cu < (coupled ? w.ChannelCount[u] : 1u); ++cu) {
                        const auto chu = size_t(w.ChannelBase[u]) + cu;
                        float cross = 0;
                        const auto *nsv = &w.NormalShape[chv * count], *nsu = &w.NormalShape[chu * count];
                        for (uint32_t k = 0; k < count; ++k) cross += nsv[k] * nsu[k] * b.QuadCompliance[k0 + k];
                        const float modal = coupling * sustain_level * cross;
                        fe_defl -= modal * w.ChSupport[chu];
                        w.QuadCross[coupled ? chv * nch + chu : chv] = sv.SurfaceCompliance * su.SurfaceCompliance * (modal + body);
                    }
                }
                w.QuadFeDefl[chv] = fe_defl;
            }
        }
    }
    return {
        .K0 = k0,
        .Count = count,
        .OutGain = out_gain,
        .ListenerGain = listener_gain,
        .SustainLevel = sustain_level,
        .ExtraBase = extra_base,
        .Drives = drives,
        .Channels = nch,
        .Coupled = coupled,
        .SupportForce = support_force,
    };
}

void RenderObjectCoupled(
    ModalAudio &m, SurfaceAudioState &surface, SurfaceRenderScratch &w, ModalBank &b, uint32_t o,
    std::span<const uint32_t> impacts, std::span<const uint32_t> voices,
    float *out, uint32_t frame_count
) {
    const auto block = PrepareRenderBlock(surface, w, b, o, impacts, voices);
    const auto k0 = block.K0, count = block.Count, drives = block.Drives, nch = block.Channels;
    const auto out_gain = block.OutGain, listener_gain = block.ListenerGain, sustain_level = block.SustainLevel;
    const auto sample_rate = b.SampleRate;
    const auto support_force = block.SupportForce;
    const size_t nv = voices.size();
    const bool coupled = block.Coupled;
    const auto bin_read_at = [&w, base = block.ExtraBase, count](size_t t, uint32_t bb) { return size_t(w.BinRowBase[t] - base + bb) * count; };

    auto *state_re = &b.StateRe[k0];
    auto *state_im = &b.StateIm[k0];
    const auto *coeff_re = &b.CoeffRe[k0];
    const auto *coeff_im = &b.CoeffIm[k0];
    auto *gains = w.DriveGains.data();
    auto *forces = w.Forces.data();
    auto *excite = w.Excite.data();
    const bool gather = drives > VoiceDrives;
    if (!gather) std::fill_n(excite, count, 0.f);
    const auto inv_mass = b.RigidInvMass[o];
    const bool mobile = inv_mass > 0;
    auto rigid_vel = b.RigidVel[o];
    const float dt = 1.f / sample_rate;
    const auto accel_noise_gain = surface.AccelNoiseGain.load(std::memory_order_relaxed) * listener_gain;
    const auto rad_b0 = b.RadiatorB0[o];
    const auto air_b0 = b.AirB0[o], air_b1 = b.AirB1[o], air_b2 = b.AirB2[o];
    const auto recoil_a1 = b.RecoilA1[o], recoil_a2 = b.RecoilA2[o];
    vec3 normal_sum{0};
    for (const auto v : voices) normal_sum += surface.VoiceState[v].N;
    const float normal_len = numeric::Length(normal_sum);
    const vec3 accel_axis = normal_len > 0 ? normal_sum / normal_len : vec3{0};
    auto rad_z1 = b.RadiatorZ1[o], rad_z2 = b.RadiatorZ2[o];
    auto air_z1 = b.AirZ1[o], air_z2 = b.AirZ2[o];
    {
        // The diagonal sits at stride nch+1 in the full matrix, and the uncoupled fallback stores only that.
        const size_t diag_stride = coupled ? size_t(nch) + 1 : 1;
        const float drive_level = sustain_level;
        // Each drive row silences on its own, so a feedback loop through the modes can be attributed to a row.
        // The body feed stays live.
        const float geo_level = surface.MuteGeometricDrive.load(std::memory_order_relaxed) ? 0.f : drive_level;
        const float fric_level = surface.MuteFrictionDrive.load(std::memory_order_relaxed) ? 0.f : drive_level;
        for (uint32_t s = 0; s < frame_count; ++s) {
            // Impact rows are known before the voices' solve, so they fill and gather first and each voice's free prediction includes their arrival.
            const bool impact_drive = !impacts.empty();
            for (size_t t = 0; t < impacts.size(); ++t) forces[nv * VoiceDrives + t] = m.ForceScratch[size_t(impacts[t]) * frame_count + s];
            if (impact_drive) {
                std::fill(forces + VoiceDrives, forces + nv * VoiceDrives, 0.f);
                std::fill(forces + nv * VoiceDrives + impacts.size(), forces + drives, 0.f);
                GatherExcitation(gains, forces, uint32_t(nv) * VoiceDrives, drives, excite, count);
            }
            float air_force = 0.f;
            vec3 ext_force{0};
            if (mobile) {
                const float air_in = numeric::Dot(rigid_vel, accel_axis);
                air_force = air_b0 * air_in + air_z1;
                air_z1 = air_b1 * air_in - recoil_a1 * air_force + air_z2;
                air_z2 = air_b2 * air_in - recoil_a2 * air_force;
                ext_force = -air_force * accel_axis - support_force;
            }
            bool sweep_live = false;
            {
                const float sweep_level = drive_level;
                auto *sweep_excite = w.SweepExcite.data();
                for (size_t t = 0; sweep_level > 0 && t < nv; ++t) {
                    const auto *sweep = w.SweepSets[t];
                    if (!sweep || sweep->Positions == 0 || sweep->ForceTotal <= 0) continue;
                    if (!sweep_live) {
                        std::fill_n(sweep_excite, count, 0.f);
                        sweep_live = true;
                    }
                    const double p = std::fmod(surface.VoiceCarry[voices[t]].PrevSpringPos, double(sweep->Positions));
                    const auto P = sweep->Positions;
                    const auto i0q = std::min(uint32_t(p), P - 1);
                    const auto i1q = i0q + 1 < P ? i0q + 1 : 0;
                    const auto imq = i0q > 0 ? i0q - 1 : P - 1;
                    const auto i2q = i1q + 1 < P ? i1q + 1 : 0;
                    const float frq = float(p - double(i0q));
                    const float wm = frq * (-0.5f + frq * (1.f - 0.5f * frq));
                    const float w0 = 1.f + frq * frq * (-2.5f + 1.5f * frq);
                    const float w1 = frq * (0.5f + frq * (2.f - 1.5f * frq));
                    const float w2 = frq * frq * (0.5f * frq - 0.5f);
                    const auto &stv = surface.VoiceState[voices[t]];
                    const float fscale = w.SolvedForce[t] > 0 ? sweep_level * stv.NormalForce / sweep->ForceTotal : 0.f;
                    const auto rows = std::min(count, sweep->Modes);
                    const auto *table = sweep->Table.data();
                    const auto *conformity = &w.SweepConformity[t * count];
                    for (uint32_t k = 0; k < rows; ++k) {
                        const auto *row = table + size_t(k) * P;
                        const float q = wm * row[imq] + w0 * row[i0q] + w1 * row[i1q] + w2 * row[i2q];
                        sweep_excite[k] += fscale * conformity[k] * q * b.RadiationGain[k0 + k] * b.QuadDriveScale[k0 + k];
                    }
                }
            }
            // The arrivals every read row takes this sample.
            // The junction rows take the impacts alone, the sweep driving the normal channel.
            const float *const impact_arrival = impact_drive ? excite : nullptr;
            const float *const sweep_arrival = sweep_live ? w.SweepExcite.data() : nullptr;
            for (size_t t = 0; t < nv; ++t) {
                const auto v = voices[t];
                const auto &stv = surface.VoiceState[v];
                auto &carry = surface.VoiceCarry[v];
                const auto point = ReadRow(w.PointRead, t * count, state_im, state_re, count, impact_arrival, sweep_arrival);
                const float defl1 = point.Now, defl2 = point.Next;
                const auto ch0 = w.ChannelBase[t];
                const uint32_t nchv = w.ChannelCount[t];
                const bool binned = nchv > 1;
                std::array<float, MaxSpringBins> defl_bins{};
                std::array<float, MaxSpringBins> defl2_bins{};
                std::array<float, MaxSpringBins> impact_bins{};
                const uint32_t nb_defl = binned ? nchv : 0;
                for (uint32_t bb = 0; bb < nb_defl; ++bb) {
                    const auto bin = ReadRow(w.BinRead, bin_read_at(t, bb), state_im, state_re, count, impact_arrival, sweep_arrival);
                    defl_bins[bb] = bin.Now;
                    defl2_bins[bb] = bin.Next;
                    impact_bins[bb] = bin.Arrival;
                }
                const auto q = StepVoiceQuad(stv, &w.Tracks[t * SustainedState::TrackCount], w.TurnoverTracks[t], w.SpringSets[t], w.VoiceBlocks[t], carry, defl1, std::span{defl_bins.data(), nb_defl}, sample_rate);
                const float sc = stv.SurfaceCompliance;
                const float dprev = q.Rebased ? 0.f : (carry.RigidNormal - carry.PrevRigidNormal) + (defl1 - carry.PrevDeflection);
                const float body_free = mobile ? dt * numeric::Dot(rigid_vel, stv.N) + dt * dt * inv_mass * numeric::Dot(ext_force, stv.N) : 0.f;
                const float defl_free = (defl2 - defl1) + w.QuadFeDefl[ch0] + point.Arrival;
                float rhs_t = 0, defl_t1 = 0;
                if (w.QuadTangent[t] != 0) {
                    const auto tangent = ReadRow(w.TangentRead, t * count, state_im, state_re, count, impact_arrival, nullptr);
                    defl_t1 = tangent.Now;
                    const float slip_dt = stv.SlipSpeed * dt;
                    const float dprev_t = q.Rebased ? 0.f : (carry.RigidTangent - carry.PrevRigidTangent) + (defl_t1 - carry.PrevTangentDefl) - slip_dt;
                    const float body_free_t = mobile ? dt * numeric::Dot(rigid_vel, stv.SlipDir) + dt * dt * inv_mass * numeric::Dot(ext_force, stv.SlipDir) : 0.f;
                    rhs_t = dprev_t + (tangent.Next - defl_t1) + body_free_t + w.QuadFeT[t] + tangent.Arrival - slip_dt;
                }
                w.QuadRhsT[t] = rhs_t;
                w.QuadDeflT[t] = defl_t1;
                // The transverse contact point's free step, the tangential trio along the second axis.
                // The countersurface slides along the slip direction alone, so the transverse junction rests on a still surface and no surface travel enters.
                float rhs_tr = 0, defl_tr1 = 0;
                if (w.QuadTransverse[t] != 0) {
                    const auto across = ReadRow(w.TransverseRead, t * count, state_im, state_re, count, impact_arrival, nullptr);
                    defl_tr1 = across.Now;
                    const vec3 tr = w.TransverseDir[t];
                    const float dprev_tr = q.Rebased ? 0.f : (carry.RigidTransverse - carry.PrevRigidTransverse) + (defl_tr1 - carry.PrevTransverseDefl);
                    const float body_free_tr = mobile ? dt * numeric::Dot(rigid_vel, tr) + dt * dt * inv_mass * numeric::Dot(ext_force, tr) : 0.f;
                    rhs_tr = dprev_tr + (across.Next - defl_tr1) + body_free_tr + across.Arrival;
                }
                w.QuadRhsTr[t] = rhs_tr;
                w.QuadDeflTr[t] = defl_tr1;
                if (binned) {
                    // Each bin channel's free prediction: its engagement advanced by the shared body motion and its local deflection history.
                    for (uint32_t bb = 0; bb < nchv; ++bb) {
                        const auto ch = ch0 + bb;
                        const float dprev_b = q.Rebased ? 0.f : (carry.RigidNormal - carry.PrevRigidNormal) + (defl_bins[bb] - carry.PrevBinDefl[bb]);
                        const float defl_free_b = (defl2_bins[bb] - defl_bins[bb]) + w.QuadFeDefl[ch] + impact_bins[bb];
                        w.QuadPrevW[ch] = q.ChW[bb] + sc * dprev_b;
                        w.QuadFree[ch] = q.ChW[bb] - sc * (body_free + defl_free_b);
                        w.QuadPsi[ch] = q.ChPsi[bb];
                        w.QuadGamma[ch] = q.ChGamma[bb];
                        w.QuadHx[ch] = q.ChHx[bb];
                        w.QuadGNominal[ch] = q.ChGNominal[bb];
                        w.QuadContact[ch] = (q.ChContact >> bb) & 1u;
                        carry.PrevBinDefl[bb] = defl_bins[bb];
                    }
                } else {
                    w.QuadPrevW[ch0] = q.WNew + sc * dprev;
                    w.QuadFree[ch0] = q.WNew - sc * (body_free + defl_free);
                    w.QuadPsi[ch0] = q.PsiTilde;
                    w.QuadGamma[ch0] = q.Gamma;
                    w.QuadHx[ch0] = q.Hx;
                    w.QuadGNominal[ch0] = q.GNominal;
                    w.QuadContact[ch0] = q.InContact ? 1 : 0;
                }
                w.QuadDefl[t] = defl1;
                w.QuadSlope[2 * t] = q.Slope[0];
                w.QuadSlope[2 * t + 1] = q.Slope[1];
                w.QuadSlam[t] = q.SlamSpeed;
                if (q.SlamSpeed > 0) {
                    for (uint32_t c = 0; c < nchv; ++c) {
                        const auto ch = size_t(ch0) + c;
                        w.QuadPsi[ch] = 0.f;
                        w.QuadGamma[ch] = 0.f;
                        w.QuadGNominal[ch] = 0.f;
                        w.QuadContact[ch] = 0;
                    }
                }
            }
            for (size_t t = 0; t < nv; ++t) {
                auto &vc = surface.VoiceCarry[voices[t]];
                const uint32_t nchv = w.ChannelCount[t];
                for (uint32_t c = 0; c < nchv; ++c) {
                    const auto ch = size_t(w.ChannelBase[t]) + c;
                    const float C = w.QuadCross[ch * diag_stride];
                    const float psi = w.QuadPsi[ch];
                    const float lam = -0.5f * (w.QuadFree[ch] - w.QuadPrevW[ch]) - w.QuadGamma[ch] * C * psi;
                    const float denom = std::sqrt(std::numeric_limits<float>::epsilon() * C * lam * lam + C * psi * C * psi);
                    const float gp = denom > 0 ? 2 * (std::sqrt(lam * lam + C * psi * psi) - lam) / denom : 4 / std::sqrt(std::numeric_limits<float>::epsilon() * std::max(C, 1e-30f));
                    const float gn = w.QuadGNominal[ch];
                    const bool bearing = w.QuadContact[ch] != 0 && gn > 0;
                    if (bearing) vc.LastBearingG[c] = std::min(gn, gp);
                    const bool draining = w.QuadContact[ch] == 0 && psi > 0;
                    w.QuadG[ch] = bearing ? std::min(gn, gp) : (draining ? std::min(vc.LastBearingG[c], gp) : 0.f);
                }
            }
            if (!coupled) {
                for (size_t t = 0; t < nv; ++t) {
                    const float C = w.QuadCross[t];
                    const float rhs = w.QuadFree[t] - w.QuadPrevW[t];
                    const float psi = w.QuadPsi[t];
                    const auto solve = [&](float g) { return SolveChannelStep(C, rhs, psi, w.QuadGamma[t], g); };
                    float g = w.QuadG[t];
                    float s = solve(g);
                    if (psi + 0.5f * g * s < 0.f) {
                        g = s < 0.f ? std::min(g, 2 * psi / -s) : 0.f;
                        s = solve(g);
                        if (psi + 0.5f * g * s < 0.f) {
                            g = 0.f;
                            s = solve(0.f);
                        }
                    }
                    w.QuadG[t] = g;
                    w.QuadS[t] = s;
                }
            }
            // The channels' shared step, re-solved with any violator's gradient reduced until every psi stays non-negative.
            // Zero gradients are always feasible, so the loop ends.
            for (size_t attempt = 0; coupled; ++attempt) {
                if (nch == 1) {
                    w.QuadS[0] = SolveChannelStep(w.QuadCross[0], w.QuadFree[0] - w.QuadPrevW[0], w.QuadPsi[0], w.QuadGamma[0], w.QuadG[0]);
                } else {
                    for (size_t v = 0; v < nch; ++v) {
                        for (size_t u = 0; u < nch; ++u) {
                            const double g = w.QuadG[u];
                            w.QuadMat[v * nch + u] = (v == u ? 1.0 : 0.0) + double(w.QuadCross[v * nch + u]) * (double(w.QuadGamma[u]) * g + 0.25 * g * g);
                        }
                        double rhs = double(w.QuadFree[v]) - double(w.QuadPrevW[v]);
                        for (size_t u = 0; u < nch; ++u) rhs -= double(w.QuadCross[v * nch + u]) * double(w.QuadPsi[u]) * double(w.QuadG[u]);
                        w.QuadRhs[v] = rhs;
                    }
                    for (size_t p = 0; p < nch; ++p) {
                        size_t pivot = p;
                        for (size_t r = p + 1; r < nch; ++r) {
                            if (std::abs(w.QuadMat[r * nch + p]) > std::abs(w.QuadMat[pivot * nch + p])) pivot = r;
                        }
                        if (pivot != p) {
                            for (size_t c = p; c < nch; ++c) std::swap(w.QuadMat[p * nch + c], w.QuadMat[pivot * nch + c]);
                            std::swap(w.QuadRhs[p], w.QuadRhs[pivot]);
                        }
                        const double inv = 1.0 / w.QuadMat[p * nch + p];
                        for (size_t r = p + 1; r < nch; ++r) {
                            const double f = w.QuadMat[r * nch + p] * inv;
                            if (f == 0.0) continue;
                            for (size_t c = p; c < nch; ++c) w.QuadMat[r * nch + c] -= f * w.QuadMat[p * nch + c];
                            w.QuadRhs[r] -= f * w.QuadRhs[p];
                        }
                    }
                    for (size_t r = nch; r-- > 0;) {
                        double acc = w.QuadRhs[r];
                        for (size_t c = r + 1; c < nch; ++c) acc -= w.QuadMat[r * nch + c] * double(w.QuadS[c]);
                        w.QuadS[r] = float(acc / w.QuadMat[r * nch + r]);
                    }
                }
                bool feasible = true;
                for (size_t t = 0; t < nch; ++t) {
                    if (w.QuadPsi[t] + 0.5f * w.QuadG[t] * w.QuadS[t] < 0.f) {
                        feasible = false;
                        w.QuadG[t] = attempt < nch && w.QuadS[t] < 0.f ? std::min(w.QuadG[t], 2 * w.QuadPsi[t] / -w.QuadS[t]) : 0.f;
                    }
                }
                if (feasible || attempt > nch) break;
            }
            vec3 rigid_force{0};
            for (size_t t = 0; t < nv; ++t) {
                const auto v = voices[t];
                const auto &stv = surface.VoiceState[v];
                auto &carry = surface.VoiceCarry[v];
                const auto ch0 = w.ChannelBase[t];
                const uint32_t nchv = w.ChannelCount[t];
                const TangentInputs tan{
                    .Rhs = w.QuadRhsT[t],
                    .Ct = w.QuadCt[t],
                    .Cnt = w.QuadCnt[t],
                    .Defl = w.QuadDeflT[t],
                    .RhsTr = w.QuadRhsTr[t],
                    .CtTr = w.QuadCtTr[t],
                    .CntTr = w.QuadCntTr[t],
                    .DeflTr = w.QuadDeflTr[t],
                    .Active = w.QuadTangent[t] != 0,
                    .ActiveTr = w.QuadTransverse[t] != 0,
                };
                VoiceForce f;
                std::array<float, MaxSpringBins> bin_reaction{};
                if (nchv > 1) {
                    uint32_t contact_mask = 0;
                    for (uint32_t c = 0; c < nchv; ++c) contact_mask |= uint32_t(w.QuadContact[ch0 + c] != 0) << c;
                    f = FinishBinnedVoiceQuad(
                        stv, carry, nchv, contact_mask,
                        &w.QuadPsi[ch0], &w.QuadHx[ch0], &w.QuadG[ch0], &w.QuadS[ch0], &w.ChSupport[ch0], bin_reaction.data(),
                        w.QuadCn[t], w.QuadSlope[2 * t], w.QuadSlope[2 * t + 1], sample_rate, tan
                    );
                } else {
                    f = FinishVoiceQuad(stv, carry, w.QuadPsi[ch0], w.QuadHx[ch0], w.QuadContact[ch0] != 0, w.QuadG[ch0], w.QuadS[ch0], w.QuadCn[t], w.QuadSlope[2 * t], w.QuadSlope[2 * t + 1], sample_rate, tan);
                }
                carry.PrevDeflection = w.QuadDefl[t];
                if (mobile) {
                    rigid_force += f.Normal * stv.N;
                    if (w.QuadTangent[t] != 0) rigid_force += f.Frictional * stv.SlipDir;
                    if (w.QuadTransverse[t] != 0) rigid_force += f.Transverse * w.TransverseDir[t];
                }
                forces[t * VoiceDrives] = nchv > 1 ? 0.f : drive_level * f.Normal;
                forces[t * VoiceDrives + 1] = geo_level * f.Geometric[0];
                forces[t * VoiceDrives + 2] = geo_level * f.Geometric[1];
                forces[t * VoiceDrives + 3] = fric_level * f.Frictional;
                if (w.TransverseRow[t] != NoTransverseRow) forces[w.TransverseRow[t]] = fric_level * f.Transverse;
                if (const uint32_t nb = std::min(stv.SpringBins, MaxSpringBins); nb > 0) {
                    for (uint32_t bb = 0; bb < nb; ++bb) forces[w.BinRowBase[t] + bb] = nchv > 1 && bb < nchv ? drive_level * bin_reaction[bb] : 0.f;
                    float moment = 0;
                    if (stv.InverseAngularInertia > 0 && stv.SpringHalfExtent > 0 && nchv > 1) {
                        for (uint32_t bb = 0; bb < nb && bb < nchv; ++bb) {
                            moment -= bin_reaction[bb] * SpringBinOffset(bb, stv.SpringBins, stv.SpringHalfExtent);
                        }
                    }
                    carry.TiltMoment = moment;
                }
                w.SolvedForce[t] = f.Normal + stv.NormalForce;
                if (w.QuadSlam[t] > 0 && w.QuadCn[t] > 0 && stv.SurfaceCompliance > 0) {
                    const float v_in = w.QuadSlam[t];
                    const float e = std::max(1.f - (2.f / 3.f) * carry.EngagementDamping * v_in, 0.f);
                    const float close_force = (1.f + e) * (v_in / stv.SurfaceCompliance) * dt / w.QuadCn[t];
                    forces[t * VoiceDrives] += drive_level * close_force;
                    if (mobile) rigid_force += close_force * stv.N;
                }
            }
            if (mobile) {
                rigid_vel += dt * inv_mass * (rigid_force - air_force * accel_axis);
                for (size_t t = 0; t < nv; ++t) {
                    auto &carry = surface.VoiceCarry[voices[t]];
                    carry.RigidNormal += numeric::Dot(rigid_vel, surface.VoiceState[voices[t]].N) * dt;
                    if (w.QuadTangent[t] != 0) carry.RigidTangent += numeric::Dot(rigid_vel, surface.VoiceState[voices[t]].SlipDir) * dt;
                    if (w.QuadTransverse[t] != 0) carry.RigidTransverse += numeric::Dot(rigid_vel, w.TransverseDir[t]) * dt;
                }
                const float rad_in = numeric::Dot(rigid_vel, accel_axis);
                const float pressure = rad_b0 * rad_in + rad_z1;
                rad_z1 = -2 * rad_b0 * rad_in - recoil_a1 * pressure + rad_z2;
                rad_z2 = rad_b0 * rad_in - recoil_a2 * pressure;
                out[s] += accel_noise_gain * pressure;
            }
            if (gather) GatherExcitation(gains, forces, uint32_t(nv) * VoiceDrives, drives, excite, count);
            if (sweep_live) {
                if (!gather) std::fill_n(excite, count, 0.f);
                const float *se = w.SweepExcite.data();
                for (uint32_t k = 0; k < count; ++k) excite[k] += se[k];
            }
            out[s] += AdvanceModes(state_re, state_im, coeff_re, coeff_im, gains, forces, excite, &b.OutPhaseIm[k0], &b.OutPhaseRe[k0], count) * out_gain;
        }
        b.RigidVel[o] = rigid_vel;
        b.RadiatorZ1[o] = rad_z1;
        b.RadiatorZ2[o] = rad_z2;
        b.AirZ1[o] = air_z1;
        b.AirZ2[o] = air_z2;
        b.Ringing[o] = 1;
        b.LiveModeCount[o] = count;
    }
}

} // namespace

/***** Core modal interface from audio/SurfaceContact.h *****/

void SurfaceAudioStateDelete::operator()(SurfaceAudioState *s) const { delete s; }
void SurfaceRenderScratchDelete::operator()(SurfaceRenderScratch *s) const { delete s; }

SurfaceAudioStatePtr MakeSurfaceAudioState() { return SurfaceAudioStatePtr{new SurfaceAudioState}; }

uint32_t SurfaceVoiceCount(const ModalAudio &m, uint32_t object) {
    return uint32_t(std::ranges::count(Surface(m).VoiceObject, object));
}

uint32_t SurfaceActiveVoices(const ModalAudio &m) { return uint32_t(Surface(m).VoiceId.size()); }

void SurfaceSilenceObject(ModalAudio &m, uint32_t object) {
    auto &surface = Surface(m);
    for (uint32_t i = uint32_t(surface.VoiceId.size()); i-- > 0;) {
        if (surface.VoiceObject[i] == object) RemoveVoice(surface, i);
    }
}

bool SurfaceRenderObject(ModalAudio &m, ModalRenderScratch &mw, ModalBank &b, uint32_t object, std::span<const uint32_t> impacts, float *out, uint32_t frame_count) {
    auto &surface = Surface(m);
    if (!mw.Surface) mw.Surface = SurfaceRenderScratchPtr{new SurfaceRenderScratch};
    auto &w = *mw.Surface;
    w.Voices.clear();
    for (uint32_t v = 0; v < uint32_t(surface.VoiceId.size()); ++v) {
        if (surface.VoiceObject[v] == object) w.Voices.push_back(v);
    }
    if (w.Voices.empty()) return false;
    RenderObjectCoupled(m, surface, w, b, object, impacts, w.Voices, out, frame_count);
    if (ReportContacts) {
        const auto samples = Report.Samples.load(std::memory_order_relaxed);
        const auto cell_samples = Report.CellSamples.load(std::memory_order_relaxed);
        const auto seconds = Report.SampleRate > 0 ? double(samples) / Report.SampleRate : 0.0;
        surface.ContactShockRate.store(seconds > 0 ? float(double(Report.Shocks.load(std::memory_order_relaxed)) / seconds) : 0.f, std::memory_order_relaxed);
        surface.ContactFreeShare.store(samples > 0 ? float(double(Report.Free.load(std::memory_order_relaxed)) / double(samples)) : 0.f, std::memory_order_relaxed);
        surface.ContactBearingShare.store(cell_samples > 0 ? float(double(Report.CellBearing.load(std::memory_order_relaxed)) / double(cell_samples)) : 0.f, std::memory_order_relaxed);
    }
    return true;
}

void SurfaceInstallBank(ModalAudio &m) {
    auto &surface = Surface(m);
    surface.VoiceId.clear();
    surface.VoiceObject.clear();
    surface.VoiceState.clear();
    surface.VoiceCarry.clear();
    ResetPool(surface.SurfaceTracks, surface.SurfaceTrackSlotByKey, surface.VoiceTrackMask, surface.ReusableSlots);
    ResetPool(surface.ContactSprings, surface.ContactSpringSlotByKey, surface.VoiceSpringMask, surface.ReusableSpringSlots);
    ResetPool(surface.SweepTables, surface.SweepTableSlotByKey, surface.VoiceSweepMask, surface.ReusableSweepSlots);
    // Every published set addresses the replaced bank's object slots.
    surface.PublishedVoices.store(nullptr, std::memory_order_relaxed);
    for (auto &set : surface.VoiceSets) set.Voices.clear();
}

VoiceSet &NextVoiceSet(SurfaceAudioState &s) {
    // Two publishes back, which no callback can still be reading by the time a third frame comes round.
    s.VoiceSetWrite = (s.VoiceSetWrite + 1) % uint32_t(s.VoiceSets.size());
    auto &set = s.VoiceSets[s.VoiceSetWrite];
    set.Voices.clear();
    return set;
}

void PublishVoiceSet(SurfaceAudioState &s) {
    auto &set = s.VoiceSets[s.VoiceSetWrite];
    set.Frame = ++s.VoiceFrame;
    s.PublishedVoices.store(&set, std::memory_order_release);
}
