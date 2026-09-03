#pragma once

#include <complex>
#include <cstdint>
#include <span>
#include <vector>

// Discrete Fourier transforms of real signals, run on vDSP.
// Every transform is unnormalized, so a forward followed by an inverse scales by the sample count.
namespace fft {
// The `n / 2 + 1` bins an `n`-sample real signal determines: `X[k] = sum(x[j] * e^(-2*pi*i*j*k/n), 0 <= j < n)`.
std::vector<std::complex<float>> RealToComplex(std::span<const float> in);

void ComplexToReal(std::span<const std::complex<float>> in, std::span<float> out);

// The same inverse over a `columns` by `rows` field, reading `columns * (rows / 2 + 1)` bins in row-major order.
void ComplexToReal2d(std::span<const std::complex<float>> in, uint32_t columns, uint32_t rows, std::span<float> out);

// The smallest length at or above `n` that every transform here runs directly rather than through a chirp convolution:
// a power of two, or 3, 5 or 15 times one from sixteen up.
uint32_t DirectLength(uint32_t n);
} // namespace fft
