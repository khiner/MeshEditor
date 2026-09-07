#pragma once
#include <complex>
#include <cstdint>
#include <optional>
#include <vector>
struct FFTData {
    std::vector<std::complex<float>> Bins;
    size_t NumReal;
};

std::optional<float> EstimateFundamentalFrequency(const FFTData &, uint32_t);
FFTData ComputeFft(const std::vector<float> &, uint32_t);
