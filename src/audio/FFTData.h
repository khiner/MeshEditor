#pragma once

#include <fftw3.h>

#include <vector>

struct FFTData {
    fftwf_complex *Complex;
    fftwf_plan Plan;
    size_t NumReal;

    FFTData(std::vector<float> &&frames)
        : Complex(fftwf_alloc_complex(frames.size() / 2 + 1)),
          Plan(fftwf_plan_dft_r2c_1d(frames.size(), frames.data(), Complex, FFTW_MEASURE)),
          NumReal(frames.size()) {
        fftwf_execute(Plan);
    }
    FFTData(FFTData &&other) noexcept
        : Complex(std::exchange(other.Complex, nullptr)),
          Plan(std::exchange(other.Plan, nullptr)),
          NumReal(other.NumReal) {}

    const FFTData &operator=(FFTData &&other) {
        if (this != &other) {
            Complex = std::exchange(other.Complex, nullptr);
            Plan = std::exchange(other.Plan, nullptr);
            NumReal = other.NumReal;
        }
        return *this;
    }

    ~FFTData() {
        fftwf_destroy_plan(Plan);
        fftwf_free(Complex);
    }
};
