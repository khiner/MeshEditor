#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

// A track is read cyclically, so at a micron-scale spacing a contact crosses about 0.2 m of surface before repeating.
constexpr uint32_t TrackSamples{32768};

uint64_t HashParams(uint64_t seed, auto... values) {
    const auto combine = [&seed](double v) { seed ^= std::hash<double>{}(v) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2); };
    (combine(double(values)), ...);
    return seed;
}

// A track of surface heights a contact rides over, indexed by distance along the surface and traversed cyclically.
struct RoughnessTrack {
    std::vector<float> Heights; // Zero-mean, unit root-mean-square.
    std::vector<float> Sum; // Running integral, one entry longer than Heights, so a smoothed read costs two lookups.
    float Spacing{0}; // Distance along the surface between samples, m.
    float Rms{1}; // Root-mean-square height of the source, m. A synthesized track leaves this to the surface's roughness.
};

// Synthesize a self-affine roughness track, flat below the spatial frequency the correlation length sets and falling as q^p above it.
// Deterministic in its arguments, so only the surface parameters persist.
RoughnessTrack SynthesizeRoughness(float correlation_length, float spectral_slope, float spacing, uint32_t count);

RoughnessTrack MakeProfileTrack(std::span<const float> heights, float spacing);

// A read position wrapped into the track: the sample, the fraction past it, and whole traversals from the start.
struct TrackPos {
    size_t Index;
    float Frac;
    double Wraps;
};
inline TrackPos WrapTrackPos(const RoughnessTrack &t, double pos) {
    const double n = double(t.Heights.size());
    const double wraps = std::floor(pos / n);
    const double f = std::max(pos - wraps * n, 0.0);
    // Rounding can put the remainder a hair outside [0, n), which would index off the end of the track.
    const auto i = std::min(size_t(f), t.Heights.size() - 1);
    return {i, float(f - double(i)), wraps};
}

// The running integral at a fractional sample position, extended cyclically.
inline float TrackIntegral(const RoughnessTrack &t, double pos) {
    const auto [i, frac, wraps] = WrapTrackPos(t, pos);
    return t.Sum[i] + frac * t.Heights[i] + float(wraps) * t.Sum.back();
}

// Mean height over `window` samples centered on `pos`.
// The window is the contact filter: a patch of radius a resolves down to roughly 2a, whatever the speed.
inline float ReadTrack(const RoughnessTrack &t, double pos, float window) {
    if (window <= 1.f) {
        const auto [i, frac, _] = WrapTrackPos(t, pos);
        const auto j = i + 1 < t.Heights.size() ? i + 1 : 0;
        return t.Heights[i] + frac * (t.Heights[j] - t.Heights[i]);
    }
    const double half = 0.5 * window;
    return (TrackIntegral(t, pos + half) - TrackIntegral(t, pos - half)) / window;
}
