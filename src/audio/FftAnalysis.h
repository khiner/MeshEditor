#pragma once
#include <complex>
#include <optional>
#include <span>
#include <vector>
struct FFTData {
    std::vector<std::complex<float>> Bins;
    size_t NumReal;
};

std::optional<float> EstimateFundamentalFrequency(const FFTData &, uint32_t);
FFTData ComputeFft(std::span<const float>, uint32_t);
