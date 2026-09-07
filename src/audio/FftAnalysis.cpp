#include "audio/FftAnalysis.h"
#include "audio/Fft.h"
#include <algorithm>
#include <cmath>
using std::ranges::nth_element;
std::optional<float> EstimateFundamentalFrequency(const FFTData &fft, uint32_t sample_rate) {
    const size_t n_bins = fft.Bins.size();

    std::vector<float> mag_db(n_bins);
    for (size_t i = 0; i < n_bins; ++i) {
        mag_db[i] = 10.f * std::log10f(std::max(std::norm(fft.Bins[i]), 1e-20f));
    }

    // Noise floor from upper half median
    std::vector<float> upper(mag_db.begin() + n_bins / 2, mag_db.end());
    nth_element(upper, upper.begin() + upper.size() / 2);
    const float threshold = upper[upper.size() / 2] + 15.f;

    constexpr size_t W{15}; // Prominence window
    const size_t min_bin = 50 * fft.NumReal / sample_rate;
    for (size_t i = std::max(min_bin, W); i < n_bins - W; ++i) {
        if (mag_db[i] <= mag_db[i - 1] || mag_db[i] <= mag_db[i + 1] || mag_db[i] < threshold) continue;

        constexpr float ProminenceThresholdDb{10.f};
        // Prominence check: peak must be above the local mean by ProminenceThresholdDb
        float local_sum = 0;
        for (size_t j = i - W; j <= i + W; ++j) local_sum += mag_db[j];
        const float local_mean = local_sum / (2 * W + 1);
        if (mag_db[i] - local_mean >= ProminenceThresholdDb) return i * sample_rate / fft.NumReal;
    }
    return std::nullopt;
}

// Capture a short audio segment shortly after the impact for FFT.
FFTData ComputeFft(const std::vector<float> &frames, uint32_t sample_rate) {
    constexpr uint32_t FftStartFrame = 30;
    const uint32_t FftEndFrame = sample_rate / 16;
    const uint32_t n = FftEndFrame - FftStartFrame;
    constexpr float coefficients[]{0.35875, -0.48829, 0.14128, -0.01168};
    std::vector<float> windowed(n);
    for (uint32_t i = 0; i < n; ++i) {
        float weight = 0.f;
        for (uint32_t j = 0; j < std::size(coefficients); ++j) weight += coefficients[j] * __cospi(float(2 * i * j) / float(n));
        windowed[i] = (n == 1 ? 1.f : weight) * frames[FftStartFrame + i];
    }
    return {fft::RealToComplex(windowed), windowed.size()};
}
