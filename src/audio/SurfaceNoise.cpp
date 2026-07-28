#include "SurfaceNoise.h"

#include <fftw3.h>

#include <algorithm>
#include <numbers>
#include <numeric>

namespace {
uint64_t SplitMix64(uint64_t &state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
float NextPhase(uint64_t &state) {
    return float(SplitMix64(state) >> 40) / float(1 << 24) * 2 * std::numbers::pi_v<float>;
}

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
    t.Sum.resize(n + 1);
    t.Sum[0] = 0;
    std::partial_sum(t.Heights.begin(), t.Heights.end(), t.Sum.begin() + 1);
    return rms;
}
} // namespace

RoughnessTrack SynthesizeRoughness(float correlation_length, float spectral_slope, float spacing, uint32_t count) {
    RoughnessTrack track;
    track.Spacing = spacing;
    track.Heights.assign(count, 0.f);
    if (count < 2 || spacing <= 0) {
        Finish(track);
        return track;
    }

    const uint32_t bins = count / 2 + 1;
    auto *spectrum = fftwf_alloc_complex(bins);
    const float q0 = 1.f / std::max(correlation_length, 1e-9f);
    const float dq = 1.f / (float(count) * spacing);
    uint64_t state = HashParams(0x517cc1b727220a95ull, correlation_length, spectral_slope, spacing);
    spectrum[0][0] = 0.f; // Zero mean.
    spectrum[0][1] = 0.f;
    for (uint32_t i = 1; i < bins; ++i) {
        const float q = float(i) * dq;
        const float amplitude = q > q0 ? std::pow(q / q0, spectral_slope * 0.5f) : 1.f;
        const float phase = NextPhase(state);
        spectrum[i][0] = amplitude * std::cos(phase);
        spectrum[i][1] = amplitude * std::sin(phase);
    }
    auto *plan = fftwf_plan_dft_c2r_1d(int(count), spectrum, track.Heights.data(), FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    fftwf_free(spectrum);

    Finish(track);
    return track;
}

RoughnessTrack MakeProfileTrack(std::span<const float> heights, float spacing) {
    RoughnessTrack track;
    track.Spacing = spacing;
    track.Heights.assign(heights.begin(), heights.end());
    track.Rms = Finish(track);
    return track;
}
